"""GTP (Go Text Protocol) engine wrapper for a trained ResNetModel Go agent.

Loads an Orbax checkpoint produced by train_selfplay.py and exposes the model
as a GTP v2 engine that can be used with GoGui, Sabaki, or any GTP client.

Usage:
    python gtp_engine.py --checkpoint path/to/checkpoints/000150
    python gtp_engine.py --checkpoint path/to/run_dir  # picks latest checkpoint
    python gtp_engine.py --checkpoint path/to/checkpoints/000150 --stochastic
    python gtp_engine.py --checkpoint path/to/checkpoints/000150 --num-filters 128 --num-blocks 6

GTP protocol reference: https://www.lysator.liu.se/~gunnar/gtp/
"""

from __future__ import annotations

import argparse
import logging

# Suppress JAX/XLA startup noise on stderr so it doesn't corrupt GTP output
import os
import sys
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pgx  # type: ignore

from jaxpot.agents.base_rollout_actor import BaseRolloutActor
from jaxpot.agents.mcts_actor import MCTSActor
from jaxpot.agents.policy_actor import PolicyActor
from jaxpot.alphazero.mcts import MCTSConfig
from jaxpot.models.resnet_model import ResNetModel


def _setup_logging(log_file: str | None) -> logging.Logger:
    log = logging.getLogger("gtp_engine")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    # Always log to stderr (doesn't corrupt GTP stdout)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


_log = logging.getLogger("gtp_engine")


# ---------------------------------------------------------------------------
# GTP coordinate helpers
# ---------------------------------------------------------------------------

# GTP uses letters A–T (skipping I) for columns, numbers 1–N for rows
# from the bottom.  pgx uses row-major order: action = row * size + col
# where row 0 is the top of the board.

_GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # 19 letters, no I


def gtp_vertex_to_action(vertex: str, size: int) -> int | None:
    """Convert a GTP vertex string (e.g. 'D4', 'pass') to a pgx action index.

    Returns None if the vertex is invalid.
    """
    v = vertex.strip().upper()
    if v == "PASS":
        return size * size  # pass action

    if len(v) < 2:
        return None

    col_char = v[0]
    if col_char not in _GTP_COLS:
        return None
    col = _GTP_COLS.index(col_char)

    try:
        # GTP row 1 = bottom row; pgx row 0 = top row
        gtp_row = int(v[1:])
    except ValueError:
        return None

    if gtp_row < 1 or gtp_row > size:
        return None
    if col >= size:
        return None

    pgx_row = size - gtp_row
    return pgx_row * size + col


def action_to_gtp_vertex(action: int, size: int) -> str:
    """Convert a pgx action index to a GTP vertex string."""
    if action == size * size:
        return "pass"
    pgx_row = action // size
    col = action % size
    gtp_row = size - pgx_row
    return f"{_GTP_COLS[col]}{gtp_row}"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _resolve_checkpoint_path(path_str: str) -> Path:
    """Resolve a checkpoint path to a numbered directory."""
    p = Path(path_str).expanduser().resolve()
    if p.is_dir() and not p.name.isdigit():
        dirs = [d for d in p.iterdir() if d.is_dir() and d.name.isdigit()]
        if not dirs:
            raise ValueError(f"No completed checkpoint found in {p}")
        dirs.sort(key=lambda d: int(d.name))
        return dirs[-1].resolve()
    return p


def load_model(checkpoint_path: str, num_filters: int, num_blocks: int, size: int) -> ResNetModel:
    """Load a ResNetModel from an Orbax checkpoint."""
    ckpt_dir = _resolve_checkpoint_path(checkpoint_path)
    state_path = ckpt_dir / "state"

    if not state_path.exists():
        raise FileNotFoundError(
            f"No 'state' directory found at {ckpt_dir}. "
            "Make sure the path points to a numbered checkpoint directory."
        )

    obs_shape = (size, size, 17)
    action_dim = size * size + 1

    # Build the model with concrete values (same pattern as eval_go.py) so that
    # jnp operations in __init__ are not traced — avoids ConcretizationTypeError.
    model_template = ResNetModel(
        rngs=nnx.Rngs(0),
        obs_shape=obs_shape,
        action_dim=action_dim,
        num_filters=num_filters,
        num_blocks=num_blocks,
    )
    graphdef, abstract_state = nnx.split(model_template)

    checkpointer = ocp.StandardCheckpointer()
    try:
        fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        payload = checkpointer.restore(
            state_path,
            args=ocp.args.StandardRestore(fallback_sharding=fallback_sharding),
        )
    except TypeError:
        _log.warning("Using legacy restore method")
        payload = checkpointer.restore(state_path)

    nnx.replace_by_pure_dict(abstract_state, payload["model"])
    model = nnx.merge(graphdef, abstract_state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# GTP Engine
# ---------------------------------------------------------------------------

SUPPORTED_COMMANDS = [
    "protocol_version",
    "name",
    "version",
    "known_command",
    "list_commands",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
    "quit",
    "showboard",
    "time_settings",
    "time_left",
    "final_score",
]


class GTPEngine:
    def __init__(
        self,
        rollout_actor: BaseRolloutActor,
        size: int = 9,
        stochastic: bool = False,
        log_illegal_moves: bool = False,
    ):
        self.rollout_actor = rollout_actor
        self.size = size
        self.stochastic = stochastic
        self.log_illegal_moves = log_illegal_moves
        self.komi = 6.5
        self._move_num = 0

        self.env = pgx.make(f"go_{size}x{size}")
        self._reset()

    def _reset(self) -> None:
        key = jax.random.PRNGKey(0)
        self.state = self.env.init(key)
        self._move_num = 0
        _log.info("Board reset")

    def _infer(self) -> int:
        """Run rollout-actor inference and return the chosen action."""
        legal_mask = self.state.legal_action_mask
        num_legal = int(jnp.sum(legal_mask))
        _log.debug(
            "Inference: move=%d current_player=%d legal_moves=%d terminated=%s",
            self._move_num,
            int(self.state.current_player),
            num_legal,
            bool(self.state.terminated),
        )

        # Rollout agents operate on batched states. GTP is single-position,
        # so we add/remove a leading batch dimension.
        batched_state = jax.tree.map(lambda x: x[None, ...], self.state)
        key = jax.random.PRNGKey(self._move_num + 1)
        output = self.rollout_actor.sample_actions(batched_state, key)
        sampled_action = int(output.actions[0])
        logits = output.policy_logits[0]
        action = sampled_action if self.stochastic else int(jnp.argmax(logits))

        _log.debug(
            "Chosen action=%d (%s) legal=%s",
            action,
            action_to_gtp_vertex(action, self.size),
            bool(legal_mask[action]),
        )
        return action

    def _apply_action(self, action: int) -> None:
        prev_terminated = bool(self.state.terminated)
        self.state = self.env.step(self.state, action)
        self._move_num += 1
        _log.debug(
            "Applied action=%d (%s) -> terminated=%s rewards=%s",
            action,
            action_to_gtp_vertex(action, self.size),
            bool(self.state.terminated),
            self.state.rewards if not prev_terminated else "N/A",
        )
        _log.debug("Position after move %d:\n%s", self._move_num, self._render_board())

    # ------------------------------------------------------------------
    # GTP response helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ok(cmd_id: str, msg: str = "") -> str:
        if msg:
            return f"={cmd_id} {msg}\n\n"
        return f"={cmd_id}\n\n"

    @staticmethod
    def _err(cmd_id: str, msg: str) -> str:
        return f"?{cmd_id} {msg}\n\n"

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def handle(self, line: str) -> str | None:
        """Parse one GTP line and return the response string, or None for empty lines."""
        line = line.strip()
        # Strip inline comments
        if "#" in line:
            line = line[: line.index("#")].strip()
        if not line:
            return None

        parts = line.split()
        # Optional numeric ID prefix
        cmd_id = ""
        if parts[0].isdigit():
            cmd_id = parts[0]
            parts = parts[1:]

        if not parts:
            return None

        cmd = parts[0].lower()
        args = parts[1:]

        _log.debug("CMD: %s %s", cmd, " ".join(args))
        try:
            return self._dispatch(cmd_id, cmd, args)
        except Exception:
            tb = traceback.format_exc()
            _log.error("Unhandled exception in command '%s':\n%s", cmd, tb)
            return self._err(cmd_id, f"internal error: {tb.splitlines()[-1]}")

    def _dispatch(self, cmd_id: str, cmd: str, args: list[str]) -> str | None:
        if cmd == "protocol_version":
            return self._ok(cmd_id, "2")

        elif cmd == "name":
            return self._ok(cmd_id, "Jaxpot-Go")

        elif cmd == "version":
            return self._ok(cmd_id, "1.0")

        elif cmd == "known_command":
            if not args:
                return self._err(cmd_id, "missing argument")
            known = "true" if args[0].lower() in SUPPORTED_COMMANDS else "false"
            return self._ok(cmd_id, known)

        elif cmd == "list_commands":
            return self._ok(cmd_id, "\n".join(SUPPORTED_COMMANDS))

        elif cmd == "boardsize":
            if not args:
                return self._err(cmd_id, "missing argument")
            try:
                n = int(args[0])
            except ValueError:
                return self._err(cmd_id, "boardsize not an integer")
            if n != self.size:
                return self._err(
                    cmd_id, f"unacceptable size (only {self.size}x{self.size} supported)"
                )
            self._reset()
            return self._ok(cmd_id)

        elif cmd == "clear_board":
            self._reset()
            return self._ok(cmd_id)

        elif cmd == "komi":
            if not args:
                return self._err(cmd_id, "missing argument")
            try:
                self.komi = float(args[0])
            except ValueError:
                return self._err(cmd_id, "komi not a number")
            return self._ok(cmd_id)

        elif cmd in ("time_settings", "time_left"):
            return self._ok(cmd_id)

        elif cmd == "play":
            if len(args) < 2:
                return self._err(cmd_id, "syntax error: play COLOR VERTEX")
            color = args[0].lower()
            if color not in ("black", "white", "b", "w"):
                return self._err(cmd_id, f"invalid color: {args[0]}")
            action = gtp_vertex_to_action(args[1], self.size)
            if action is None:
                return self._err(cmd_id, f"invalid vertex: {args[1]}")
            if not bool(self.state.legal_action_mask[action]):
                board_str = self._render_board()
                _log.error(
                        "Illegal move attempted: color=%s vertex=%s action=%d\n%s",
                        color,
                        args[1],
                        action,
                        board_str,
                    )
                if not self.log_illegal_moves:
                    return self._err(cmd_id, "illegal move")
            self._apply_action(action)
            return self._ok(cmd_id)

        elif cmd == "genmove":
            if not args:
                return self._err(cmd_id, "missing color argument")
            color = args[0].lower()
            if color not in ("black", "white", "b", "w"):
                return self._err(cmd_id, f"invalid color: {args[0]}")

            if bool(self.state.terminated):
                return self._ok(cmd_id, "resign")

            action = self._infer()
            self._apply_action(action)
            vertex = action_to_gtp_vertex(action, self.size)
            return self._ok(cmd_id, vertex)

        elif cmd == "final_score":
            return self._ok(cmd_id, self._compute_final_score())

        elif cmd == "showboard":
            board_str = self._render_board()
            return self._ok(cmd_id, "\n" + board_str)

        elif cmd == "quit":
            return self._ok(cmd_id)

        else:
            return self._err(cmd_id, f"unknown command: {cmd}")

    def _compute_final_score(self) -> str:
        """Return GTP final_score string, e.g. 'B+1.5' or 'W+6.5'.

        pgx rewards are from the perspective of each player:
          rewards[0] > 0  => black won
          rewards[1] > 0  => white won
        We report a margin of 1 (the reward magnitude) when the game has
        terminated, since pgx does not expose the exact territory count.
        If the game is not yet finished we still return the current leader.
        """
        rewards = self.state.rewards
        black_reward = float(rewards[0])
        white_reward = float(rewards[1])
        if black_reward > 0:
            margin = black_reward
            return f"B+{margin:.1f}"
        elif white_reward > 0:
            margin = white_reward
            return f"W+{margin:.1f}"
        else:
            return "0"

    def _render_board(self) -> str:
        """Render a simple ASCII board for the showboard command."""
        size = self.size
        # pgx observation planes: plane 0 = current player stones,
        # plane 1 = opponent stones (AlphaZero-style, most recent history first)
        obs = self.state.observation
        current_stones = obs[:, :, 0]  # current player
        opponent_stones = obs[:, :, 1]  # opponent

        # Determine who is current player (pgx: current_player 0=black, 1=white)
        current = int(self.state.current_player)

        lines = []
        col_labels = "  " + " ".join(_GTP_COLS[:size])
        lines.append(col_labels)
        for pgx_row in range(size):
            gtp_row = size - pgx_row
            row_str = f"{gtp_row:2d}"
            for col in range(size):
                if current_stones[pgx_row, col]:
                    stone = "X" if current == 0 else "O"
                elif opponent_stones[pgx_row, col]:
                    stone = "O" if current == 0 else "X"
                else:
                    stone = "."
                row_str += f" {stone}"
            lines.append(row_str)
        lines.append(col_labels)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_gtp_loop(engine: GTPEngine) -> None:
    """Read GTP commands from stdin and write responses to stdout."""
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            break

        if not line:
            # EOF
            break

        response = engine.handle(line)
        if response is None:
            continue

        sys.stdout.write(response)
        sys.stdout.flush()

        # Detect quit after flushing the acknowledgement
        stripped = line.strip()
        if "#" in stripped:
            stripped = stripped[: stripped.index("#")].strip()
        parts = stripped.split()
        # Skip optional numeric ID
        if parts and parts[0].isdigit():
            parts = parts[1:]
        if parts and parts[0].lower() == "quit":
            break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GTP engine wrapper for a trained ResNetModel Go agent."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to a numbered checkpoint directory (e.g. checkpoints/000150) "
            "or a run directory (picks the latest numbered checkpoint inside)."
        ),
    )
    parser.add_argument(
        "--size",
        type=int,
        default=9,
        help="Board size (default: 9). Only 9x9 is supported by the default config.",
    )
    parser.add_argument(
        "--num-filters",
        type=int,
        default=128,
        help="Number of ResNet filters (must match the checkpoint, default: 128).",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=6,
        help="Number of ResNet blocks (must match the checkpoint, default: 6).",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=("policy", "mcts"),
        default="policy",
        help=(
            "Rollout agent used for genmove. "
            "'policy' uses direct policy logits; 'mcts' uses AlphaZero-style search."
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help=(
            "Use the sampled move from rollout_actor.sample_actions. "
            "Default is deterministic argmax over returned policy logits."
        ),
    )
    parser.add_argument(
        "--mcts-num-simulations",
        type=int,
        default=50,
        help="MCTS simulations per move when --agent mcts (default: 50).",
    )
    parser.add_argument(
        "--mcts-max-considered-actions",
        type=int,
        default=16,
        help=(
            "Max considered actions at root for Gumbel MuZero when --agent mcts "
            "(default: 16)."
        ),
    )
    parser.add_argument(
        "--mcts-dirichlet-alpha",
        type=float,
        default=0.3,
        help="Dirichlet alpha at root when --agent mcts (default: 0.3).",
    )
    parser.add_argument(
        "--mcts-exploration-fraction",
        type=float,
        default=0.25,
        help="Root exploration fraction when --agent mcts (default: 0.25).",
    )
    parser.add_argument(
        "--mcts-gamma",
        type=float,
        default=1.0,
        help="Discount factor for MCTS recurrent function when --agent mcts (default: 1.0).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write debug logs (e.g. gtp_engine.log). "
        "Logs always go to stderr; this adds a file copy.",
    )
    parser.add_argument(
        "--log-illegal-moves",
        action="store_true",
        help="Log an error (including board state) whenever the opponent plays an illegal move.",
    )
    args = parser.parse_args()

    _setup_logging(args.log_file)
    _log.info("Loading checkpoint from %s ...", args.checkpoint)
    model = load_model(args.checkpoint, args.num_filters, args.num_blocks, args.size)
    _log.info("Model loaded.")

    if args.agent == "mcts":
        mcts_config = MCTSConfig(
            num_simulations=args.mcts_num_simulations,
            max_num_considered_actions=args.mcts_max_considered_actions,
            gamma=args.mcts_gamma,
        )
        env = pgx.make(f"go_{args.size}x{args.size}")
        rollout_actor = MCTSActor(model=model, mcts_config=mcts_config).setup(
            step_fn=jax.vmap(env.step),
            no_auto_reset_step_fn=jax.vmap(env.step),
        )
    else:
        rollout_actor = PolicyActor(model=model)

    _log.info("Starting GTP engine with agent=%s", args.agent)

    engine = GTPEngine(
        rollout_actor,
        size=args.size,
        stochastic=args.stochastic,
        log_illegal_moves=args.log_illegal_moves,
    )
    run_gtp_loop(engine)


if __name__ == "__main__":
    main()
