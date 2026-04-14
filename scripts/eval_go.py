"""Evaluate Go models in selfplay, baseline, and human modes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import distrax  # type: ignore
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp  # type: ignore
import pgx  # type: ignore
from flax import nnx
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from jaxpot.models.pgx_baseline import PGXBaselineModel
from loguru import logger
from omegaconf import OmegaConf

from jaxpot.models.utils import BaseModel
from jaxpot.utils.checkpoints import CheckpointManager


def _resolve_checkpoint_dir(path_str: str) -> Path:
    p = Path(path_str).expanduser().resolve()
    if CheckpointManager.is_checkpoint(p):
        return p
    latest = CheckpointManager.find_latest_checkpoint_dir(p)
    if latest is None:
        raise ValueError(f"No checkpoint found in {p}")
    return Path(latest)


def _load_model_from_checkpoint(train_cfg, env, checkpoint_path: str) -> BaseModel:
    ckpt_dir = _resolve_checkpoint_dir(checkpoint_path)
    model_template = instantiate(train_cfg.model, _partial_=True)(
        rngs=nnx.Rngs(0),
        obs_shape=env.observation_shape,
        action_dim=env.num_actions,
    )
    graphdef, abstract_state = nnx.split(model_template)

    checkpointer = ocp.StandardCheckpointer()
    state_path = ckpt_dir / "state"
    try:
        fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        payload = checkpointer.restore(
            state_path,
            args=ocp.args.StandardRestore(fallback_sharding=fallback_sharding),
        )
    except TypeError:
        payload = checkpointer.restore(state_path)
    nnx.replace_by_pure_dict(abstract_state, payload["model"])  # type: ignore[index]
    model = nnx.merge(graphdef, abstract_state)
    model.eval()
    logger.info(f"Loaded model weights from checkpoint {ckpt_dir}")
    return model


def _find_hydra_config_root(cfg_path: Path) -> Path | None:
    """Directory that contains Hydra groups (logger/, trainer/, …), or None."""
    for d in [cfg_path.parent, *cfg_path.parents]:
        if (d / "logger").is_dir() and (d / "trainer").is_dir():
            return d
    return None


def _build_random_model(train_cfg, env, seed: int) -> BaseModel:
    model = instantiate(train_cfg.model, _partial_=True)(
        rngs=nnx.Rngs(seed),
        obs_shape=env.observation_shape,
        action_dim=env.num_actions,
    )
    model.eval()
    logger.info("Using randomly initialized model")
    return model


def _load_train_cfg(train_config_path: str):
    cfg_path = Path(train_config_path).expanduser().resolve()
    raw_cfg = OmegaConf.load(cfg_path)

    # Plain OmegaConf load does not apply Hydra defaults; compose when needed.
    if "defaults" in raw_cfg:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        config_root = _find_hydra_config_root(cfg_path)
        rel_name: str | None = None
        if config_root is not None:
            try:
                rel = cfg_path.relative_to(config_root)
            except ValueError:
                pass
            else:
                rel_name = str(rel.with_suffix("")).replace("\\", "/")
        if rel_name is not None:
            with initialize_config_dir(version_base=None, config_dir=str(config_root)):
                composed = compose(config_name=rel_name)
        else:
            with initialize_config_dir(version_base=None, config_dir=str(cfg_path.parent)):
                composed = compose(config_name=cfg_path.stem)
        return composed
    return raw_cfg


def select_model_action(
    model: BaseModel,
    obs: jax.Array,
    legal_action_mask: jax.Array,
    key: jax.Array,
    deterministic: bool,
) -> int:
    obs_batch = obs[None, ...]
    model_output = model(obs_batch)
    logits = model_output.policy_logits[0]
    masked_logits = jnp.where(legal_action_mask, logits, -1e9)
    if deterministic:
        return int(jnp.argmax(masked_logits))
    return int(distrax.Categorical(logits=masked_logits).sample(seed=key))


def _go_board_from_state(state) -> tuple[list[list[int]], int]:
    flat = jax.device_get(state._x.board)
    size = int(jax.device_get(state._size))
    board = []
    for r in range(size):
        row = [int(flat[r * size + c]) for c in range(size)]
        board.append(row)
    return board, size


def _action_to_coord(action: int, size: int) -> str:
    if action == size * size:
        return "PASS"
    row_from_top = action // size
    col = action % size
    col_char = chr(ord("A") + col)
    row_human = size - row_from_top
    return f"{col_char}{row_human}"


def _coord_to_action(token: str, size: int) -> int | None:
    token = token.strip().upper()
    if token in {"PASS", "P"}:
        return size * size
    if len(token) < 2:
        return None
    col_char = token[0]
    row_part = token[1:]
    if not ("A" <= col_char <= chr(ord("A") + size - 1)):
        return None
    if not row_part.isdigit():
        return None
    row_human = int(row_part)
    if row_human < 1 or row_human > size:
        return None
    col = ord(col_char) - ord("A")
    row_from_top = size - row_human
    return row_from_top * size + col


def _render_ascii_board(state, step_idx: int, last_action: int | None) -> str:
    board, size = _go_board_from_state(state)
    current_player = int(jax.device_get(state.current_player))
    legal_mask = jax.device_get(state.legal_action_mask)
    legal_count = int(sum(bool(x) for x in legal_mask))
    player_label = "Black (X)" if current_player == 0 else "White (O)"

    lines = []
    lines.append("=" * 44)
    lines.append(f"Step {step_idx} | Turn: {player_label} | Legal actions: {legal_count}")
    if last_action is None:
        lines.append("Last move: (none)")
    else:
        lines.append(f"Last move: {_action_to_coord(last_action, size)} [action={last_action}]")
    lines.append("")
    cols = " ".join(chr(ord("A") + c) for c in range(size))
    lines.append(f"   {cols}")
    for r in range(size):
        row_human = size - r
        symbols = []
        for c in range(size):
            v = board[r][c]
            if v > 0:
                symbols.append("X")
            elif v < 0:
                symbols.append("O")
            else:
                symbols.append(".")
        lines.append(f"{row_human:2d} " + " ".join(symbols))
    lines.append("")
    pass_idx = size * size
    pass_legal = bool(legal_mask[pass_idx])
    lines.append(
        f"Enter move as coordinate (e.g. D4), 'pass', or action index. PASS legal: {pass_legal}"
    )
    return "\n".join(lines)


def _parse_human_move(user_in: str, size: int) -> int | None:
    token = user_in.strip()
    if token == "":
        return None
    coord_action = _coord_to_action(token, size)
    if coord_action is not None:
        return coord_action
    if token.isdigit():
        return int(token)
    return None


def select_human_action(state, step_idx: int, last_action: int | None) -> int:
    legal_mask = jax.device_get(state.legal_action_mask)
    _board, size = _go_board_from_state(state)
    legal_actions = {int(i) for i, ok in enumerate(legal_mask) if bool(ok)}
    while True:
        print("\n" + _render_ascii_board(state, step_idx, last_action))
        user_in = input("action> ").strip()
        action = _parse_human_move(user_in, size)
        if action is None:
            print(f"Invalid input '{user_in}'. Use e.g. D4, pass, or action index.")
            continue
        if action not in legal_actions:
            if action == size * size:
                move_str = "PASS"
            else:
                move_str = _action_to_coord(action, size)
            print(f"Illegal move '{move_str}' (action={action}). Choose a legal move.")
            continue
        return action


def _evaluate_reward(state, model_seat: int) -> float:
    terminated = bool(jax.device_get(state.terminated))
    if not terminated:
        return 0.0
    rewards = jax.device_get(state.rewards)
    return float(rewards[model_seat])


def play_single_game(
    env,
    key: jax.Array,
    max_steps: int,
    action_fn_player0: Callable[[object, jax.Array, int, int | None], int],
    action_fn_player1: Callable[[object, jax.Array, int, int | None], int],
    model_seat: int,
) -> tuple[list, dict]:
    key, init_key = jax.random.split(key)
    state = env.init(init_key)
    states = [state]

    last_action: int | None = None
    for step_idx in range(max_steps):
        if bool(jax.device_get(state.terminated)) or bool(jax.device_get(state.truncated)):
            break
        key, action_key = jax.random.split(key)
        current_player = int(jax.device_get(state.current_player))
        if current_player == 0:
            action = action_fn_player0(state, action_key, step_idx, last_action)
        else:
            action = action_fn_player1(state, action_key, step_idx, last_action)
        key, step_key = jax.random.split(key)
        state = env.step(state, int(action), step_key)
        last_action = int(action)
        states.append(state)

    reward = _evaluate_reward(state, model_seat)
    metrics = {
        "num_steps": len(states) - 1,
        "terminated": bool(jax.device_get(state.terminated)),
        "truncated": bool(jax.device_get(state.truncated)),
        "reward": reward,
        "win": reward > 0.0,
        "loss": reward < 0.0,
        "draw": reward == 0.0 and bool(jax.device_get(state.terminated)),
    }
    return states, metrics


def _mode_seat(game_idx: int, seat_mode: str) -> int:
    if seat_mode == "alternate":
        return game_idx % 2
    if seat_mode == "model0":
        return 0
    if seat_mode == "model1":
        return 1
    raise ValueError(f"Unsupported seat_mode={seat_mode}")


def _save_game_svgs(
    states: list,
    output_dir: Path,
    mode: str,
    game_idx: int,
    result_tag: str,
    color_theme: str,
    scale: float,
    frame_duration_ms: int,
    save_final_svg: bool,
) -> tuple[Path | None, Path]:
    game_stem = f"{mode}_game_{game_idx:03d}_{result_tag}"
    final_path = output_dir / f"{game_stem}_final.svg" if save_final_svg else None
    anim_path = output_dir / f"{game_stem}_anim.svg"
    if final_path is not None:
        pgx.save_svg(
            states[-1],
            final_path,
            color_theme=color_theme,
            scale=scale,
        )
    pgx.save_svg_animation(
        states,
        anim_path,
        color_theme=color_theme,
        scale=scale,
        frame_duration_seconds=float(frame_duration_ms) / 1000.0,
    )
    return final_path, anim_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Go model in three modes.")
    parser.add_argument(
        "--train-config",
        type=str,
        default="config/experiment/go_9x9/go_9_config.yaml",
        help="Training config used to rebuild model for checkpoint restore.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint dir or run dir containing numeric checkpoint dirs.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use randomly initialized model instead of checkpoint.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="selfplay",
        choices=["selfplay", "baseline", "human"],
        help="Evaluation mode.",
    )
    parser.add_argument("--num-games", type=int, default=3, help="Number of games.")
    parser.add_argument("--max-steps-per-game", type=int, default=256, help="Max steps per game.")
    parser.add_argument(
        "--seat-mode",
        type=str,
        default="alternate",
        choices=["alternate", "model0", "model1"],
        help="Seat assignment strategy for model in selfplay/baseline.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax policy for main model.",
    )
    parser.add_argument(
        "--deterministic-opponent",
        action="store_true",
        help="In selfplay, use deterministic policy for opponent seat.",
    )
    parser.add_argument(
        "--baseline-model-id",
        type=str,
        default="go_9x9_v0",
        help="PGX baseline model ID for baseline mode.",
    )
    parser.add_argument(
        "--baseline-deterministic",
        action="store_true",
        help="Use argmax for baseline model in baseline mode.",
    )
    parser.add_argument("--human-seat", type=int, default=0, choices=[0, 1], help="Seat for human.")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    parser.add_argument("--scale", type=float, default=1.5, help="SVG scale.")
    parser.add_argument(
        "--color-theme",
        type=str,
        default="light",
        choices=["light", "dark"],
        help="SVG color theme.",
    )
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=400,
        help="Animation frame duration in ms.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_go",
        help="Directory where SVG files are saved (animated SVG by default).",
    )
    parser.add_argument(
        "--save-final-svg",
        action="store_true",
        help="Also save final static SVG in addition to animated SVG.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    svg_dir = output_dir / "svgs"
    svg_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = _load_train_cfg(args.train_config)
    env = instantiate(train_cfg.env)
    mode = str(args.mode)

    if bool(args.random_init):
        model = _build_random_model(train_cfg, env, int(args.seed))
    else:
        if args.checkpoint is None:
            raise SystemExit("Provide --checkpoint or set --random-init")
        checkpoint_value = str(args.checkpoint).strip()
        model = _load_model_from_checkpoint(train_cfg, env, checkpoint_value)

    baseline_model = None
    if mode == "baseline":
        baseline_model = PGXBaselineModel(str(args.baseline_model_id), is_eval=True)
        logger.info(f"Loaded baseline model: {args.baseline_model_id}")

    key = jax.random.key(int(args.seed))
    total_wins, total_losses, total_draws = 0, 0, 0
    total_rewards = []
    total_steps = []

    logger.info("=" * 60)
    logger.info(f"Go evaluation mode={mode}, games={int(args.num_games)}")
    logger.info("=" * 60)

    for game_idx in range(int(args.num_games)):
        key, game_key = jax.random.split(key)
        model_seat = _mode_seat(game_idx, str(args.seat_mode))

        def model_action_fn(
            state, action_key: jax.Array, _step_idx: int, _last_action: int | None
        ) -> int:
            return select_model_action(
                model=model,
                obs=state.observation,
                legal_action_mask=state.legal_action_mask,
                key=action_key,
                deterministic=bool(args.deterministic),
            )

        def baseline_action_fn(
            state, action_key: jax.Array, _step_idx: int, _last_action: int | None
        ) -> int:
            if baseline_model is None:
                raise RuntimeError("baseline model not initialized")
            return select_model_action(
                model=baseline_model,
                obs=state.observation,
                legal_action_mask=state.legal_action_mask,
                key=action_key,
                deterministic=bool(args.baseline_deterministic),
            )

        def human_action_fn(
            state, _action_key: jax.Array, step_idx: int, last_action: int | None
        ) -> int:
            return select_human_action(state, step_idx, last_action)

        if mode == "selfplay":
            action_fn_player0 = model_action_fn
            action_fn_player1 = (
                model_action_fn
                if not bool(args.deterministic_opponent)
                else lambda state, action_key, _step_idx, _last_action: select_model_action(
                    model=model,
                    obs=state.observation,
                    legal_action_mask=state.legal_action_mask,
                    key=action_key,
                    deterministic=True,
                )
            )
        elif mode == "baseline":
            if model_seat == 0:
                action_fn_player0 = model_action_fn
                action_fn_player1 = baseline_action_fn
            else:
                action_fn_player0 = baseline_action_fn
                action_fn_player1 = model_action_fn
        else:  # human
            human_seat = int(args.human_seat)
            if human_seat not in (0, 1):
                raise SystemExit(f"human_seat must be 0 or 1, got {human_seat}")
            model_seat = 1 - human_seat
            if human_seat == 0:
                action_fn_player0 = human_action_fn
                action_fn_player1 = model_action_fn
            else:
                action_fn_player0 = model_action_fn
                action_fn_player1 = human_action_fn

        states, metrics = play_single_game(
            env=env,
            key=game_key,
            max_steps=int(args.max_steps_per_game),
            action_fn_player0=action_fn_player0,
            action_fn_player1=action_fn_player1,
            model_seat=model_seat,
        )
        total_rewards.append(metrics["reward"])
        total_steps.append(metrics["num_steps"])
        if metrics["win"]:
            total_wins += 1
            result_tag = "win"
        elif metrics["loss"]:
            total_losses += 1
            result_tag = "loss"
        else:
            total_draws += 1
            result_tag = "draw"

        final_svg, anim_svg = _save_game_svgs(
            states,
            svg_dir,
            mode,
            game_idx,
            result_tag,
            color_theme=str(args.color_theme),
            scale=float(args.scale),
            frame_duration_ms=int(args.frame_duration_ms),
            save_final_svg=bool(args.save_final_svg),
        )
        if final_svg is None:
            logger.info(
                f"game={game_idx + 1}/{int(args.num_games)} result={result_tag.upper()} "
                f"steps={metrics['num_steps']} reward={metrics['reward']:+.3f} "
                f"anim_svg={anim_svg.name}"
            )
        else:
            logger.info(
                f"game={game_idx + 1}/{int(args.num_games)} result={result_tag.upper()} "
                f"steps={metrics['num_steps']} reward={metrics['reward']:+.3f} "
                f"final_svg={final_svg.name} anim_svg={anim_svg.name}"
            )

    n = int(args.num_games)
    logger.info("-" * 60)
    logger.info(f"wins={total_wins} ({total_wins / n:.1%})")
    logger.info(f"losses={total_losses} ({total_losses / n:.1%})")
    logger.info(f"draws={total_draws} ({total_draws / n:.1%})")
    logger.info(f"avg_reward={sum(total_rewards) / max(1, len(total_rewards)):+.4f}")
    logger.info(f"avg_steps={sum(total_steps) / max(1, len(total_steps)):.2f}")
    logger.info(f"svgs={svg_dir}")


if __name__ == "__main__":
    main()
