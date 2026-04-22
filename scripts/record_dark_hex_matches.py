#!/usr/bin/env python3
"""Record Dark Hex games between a latest checkpoint and prior checkpoints.

The output is a compact JSON replay file consumed by
``visualizations/dark_hex_match_viewer.html``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


DEFAULT_RUN_DIR = Path("outputs/2026-04-21/train_selfplay.yaml/dark_hex_fast_14-48-41")
DEFAULT_OUTPUT = Path("visualizations/dark_hex_matches.json")


@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    iteration: int

    @property
    def label(self) -> str:
        return f"iter {self.iteration}"


def _as_int(value: Any) -> int:
    return int(np.asarray(value).item())


def _as_bool(value: Any) -> bool:
    return bool(np.asarray(value).item())


def _as_list(value: Any) -> list[Any]:
    return np.asarray(value).tolist()


def _read_iteration(path: Path) -> int:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing checkpoint metadata: {metadata_path}")
    with metadata_path.open() as f:
        return int(json.load(f)["iter"])


def _find_checkpoint_dirs(root: Path) -> list[CheckpointRef]:
    checkpoint_root = root / "checkpoints" if (root / "checkpoints").exists() else root
    candidates: list[Path] = []
    candidates.extend(p for p in checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit())

    milestone_root = checkpoint_root / "milestones"
    if milestone_root.exists():
        candidates.extend(p for p in milestone_root.iterdir() if p.is_dir() and p.name.isdigit())

    refs_by_iter: dict[int, CheckpointRef] = {}
    for path in candidates:
        metadata = path / "metadata.json"
        state = path / "state"
        if not metadata.exists() or not state.exists():
            continue
        ref = CheckpointRef(path=path.resolve(), iteration=_read_iteration(path))
        refs_by_iter[ref.iteration] = ref

    return sorted(refs_by_iter.values(), key=lambda ref: ref.iteration)


def _load_cfg(run_dir: Path) -> DictConfig:
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find Hydra config at {config_path}")
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected DictConfig from {config_path}")
    return cfg


def _load_model(checkpoint_path: Path, cfg: DictConfig, env) -> Any:
    raw_actions = env.num_actions
    action_dim = tuple(int(d) for d in raw_actions) if isinstance(raw_actions, tuple) else int(raw_actions)

    abstract_model = nnx.eval_shape(
        lambda: instantiate(cfg.model, _partial_=True)(
            rngs=nnx.Rngs(0),
            obs_shape=env.observation_shape,
            action_dim=action_dim,
        )
    )
    graphdef, abstract_state = nnx.split(abstract_model)

    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    payload = checkpointer.restore(
        checkpoint_path / "state",
        args=ocp.args.StandardRestore(fallback_sharding=fallback_sharding),
    )
    nnx.replace_by_pure_dict(abstract_state, payload["model"])
    model = nnx.merge(graphdef, abstract_state)
    model.eval()
    return model


def _snapshot(state) -> dict[str, Any]:
    x = state._x
    return {
        "board": _as_list(x.board),
        "black_view": _as_list(x.black_view),
        "white_view": _as_list(x.white_view),
        "current_player": _as_int(state.current_player),
        "current_color": _as_int(x.color),
        "legal_action_mask": _as_list(state.legal_action_mask),
        "move_succeeded": _as_bool(x.move_succeeded),
        "winner_color": _as_int(x.winner),
        "terminated": _as_bool(state.terminated),
        "truncated": _as_bool(state.truncated),
        "rewards": _as_list(state.rewards),
        "step_count": _as_int(state._step_count),
    }


def _color_name(color: int) -> str:
    return "black" if color == 0 else "white"


def _action_name(action: int, num_cols: int) -> str:
    row = action // num_cols
    col = action % num_cols
    return f"{chr(ord('a') + col)}{row + 1}"


def _select_action(
    model,
    state,
    key,
    deterministic: bool,
) -> tuple[int, list[float], list[float], float, float]:
    output = model(state.observation)
    masked_logits = jnp.where(state.legal_action_mask, output.policy_logits, -1e9)
    probs = jax.nn.softmax(masked_logits)
    if deterministic:
        action = jnp.argmax(masked_logits)
    else:
        action = jax.random.categorical(key, masked_logits)
    action_int = int(action)
    value = float(np.asarray(output.value).reshape(-1)[0])
    return (
        action_int,
        _as_list(probs),
        _as_list(masked_logits),
        float(np.asarray(probs[action_int])),
        value,
    )


def _record_game(
    *,
    env,
    latest_model,
    opponent_model,
    latest_ref: CheckpointRef,
    opponent_ref: CheckpointRef,
    game_index: int,
    seed: int,
    max_steps: int,
    deterministic: bool,
) -> dict[str, Any]:
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    state = env.init(init_key)

    player_order = _as_list(state._player_order)
    player_labels = {
        "0": f"latest {latest_ref.label}",
        "1": f"opponent {opponent_ref.label}",
    }
    player_checkpoints = {
        "0": str(latest_ref.path),
        "1": str(opponent_ref.path),
    }

    moves: list[dict[str, Any]] = []
    snapshots = [_snapshot(state)]

    for ply in range(max_steps):
        if _as_bool(state.terminated) or _as_bool(state.truncated):
            break

        current_player = _as_int(state.current_player)
        current_color = _as_int(state._x.color)
        actor = "latest" if current_player == 0 else "opponent"
        model = latest_model if current_player == 0 else opponent_model

        key, action_key = jax.random.split(key)
        before = _snapshot(state)
        action, policy_probs, policy_logits, selected_prob, value_estimate = _select_action(
            model,
            state,
            action_key,
            deterministic,
        )
        next_state = env.step(state, jnp.int32(action))
        after = _snapshot(next_state)
        black_edge_value = value_estimate if current_color == 0 else -value_estimate
        latest_edge_value = value_estimate if actor == "latest" else -value_estimate

        moves.append(
            {
                "ply": ply,
                "actor": actor,
                "player_id": current_player,
                "player_label": player_labels[str(current_player)],
                "color": _color_name(current_color),
                "action": action,
                "action_name": _action_name(action, env._num_cols),
                "row": action // env._num_cols,
                "col": action % env._num_cols,
                "policy": policy_probs,
                "policy_logits": policy_logits,
                "selected_policy_prob": selected_prob,
                "value_estimate": value_estimate,
                "black_edge_value": black_edge_value,
                "latest_edge_value": latest_edge_value,
                "move_succeeded": after["move_succeeded"],
                "before": before,
                "after": after,
            }
        )
        snapshots.append(after)
        state = next_state

    final = _snapshot(state)
    winner_color = final["winner_color"]
    winner_player = None
    winner_label = None
    if winner_color >= 0:
        winner_player = int(player_order[winner_color])
        winner_label = player_labels[str(winner_player)]

    return {
        "id": f"game-{game_index + 1}",
        "game_index": game_index,
        "seed": seed,
        "deterministic": deterministic,
        "latest_checkpoint": {
            "iteration": latest_ref.iteration,
            "path": str(latest_ref.path),
        },
        "opponent_checkpoint": {
            "iteration": opponent_ref.iteration,
            "path": str(opponent_ref.path),
        },
        "player_labels": player_labels,
        "player_checkpoints": player_checkpoints,
        "player_order": player_order,
        "color_players": {
            "black": int(player_order[0]),
            "white": int(player_order[1]),
        },
        "moves": moves,
        "snapshots": snapshots,
        "result": {
            "winner_color": winner_color,
            "winner_player": winner_player,
            "winner_label": winner_label,
            "rewards": final["rewards"],
            "terminated": final["terminated"],
            "truncated": final["truncated"],
            "num_plies": len(moves),
        },
    }


def _parse_checkpoint_paths(paths: list[str]) -> list[CheckpointRef]:
    refs = []
    for raw in paths:
        path = Path(raw).resolve()
        refs.append(CheckpointRef(path=path, iteration=_read_iteration(path)))
    return sorted(refs, key=lambda ref: ref.iteration)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--latest-checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", action="append", default=[])
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260421)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cfg = _load_cfg(run_dir)
    env = instantiate(cfg.env)

    if args.latest_checkpoint:
        latest_ref = CheckpointRef(
            path=args.latest_checkpoint.resolve(),
            iteration=_read_iteration(args.latest_checkpoint),
        )
    else:
        checkpoints = _find_checkpoint_dirs(run_dir)
        if not checkpoints:
            raise RuntimeError(f"No checkpoints found below {run_dir}")
        latest_ref = checkpoints[-1]

    if args.opponent_checkpoint:
        opponent_refs = _parse_checkpoint_paths(args.opponent_checkpoint)
    else:
        opponent_refs = [ref for ref in _find_checkpoint_dirs(run_dir) if ref.iteration < latest_ref.iteration]

    if not opponent_refs:
        raise RuntimeError("Need at least one prior checkpoint to record matches.")

    opponent_refs = sorted(opponent_refs, key=lambda ref: ref.iteration, reverse=True)
    latest_model = _load_model(latest_ref.path, cfg, env)
    opponent_models = {
        ref.iteration: _load_model(ref.path, cfg, env)
        for ref in opponent_refs
    }

    games = []
    for game_index in range(args.games):
        opponent_ref = opponent_refs[game_index % len(opponent_refs)]
        game = _record_game(
            env=env,
            latest_model=latest_model,
            opponent_model=opponent_models[opponent_ref.iteration],
            latest_ref=latest_ref,
            opponent_ref=opponent_ref,
            game_index=game_index,
            seed=args.seed + game_index,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )
        games.append(game)

    payload = {
        "schema_version": 1,
        "game": "dark_hex",
        "run_dir": str(run_dir),
        "num_rows": int(env._num_rows),
        "num_cols": int(env._num_cols),
        "latest_iteration": latest_ref.iteration,
        "opponent_iterations": [ref.iteration for ref in opponent_refs],
        "games": games,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Recorded {len(games)} games to {args.output}")
    for game in games:
        result = game["result"]
        opponent = game["opponent_checkpoint"]["iteration"]
        print(
            f"  {game['id']}: latest {latest_ref.iteration} vs {opponent}, "
            f"plies={result['num_plies']}, winner={result['winner_label']}"
        )


if __name__ == "__main__":
    main()
