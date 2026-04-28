#!/usr/bin/env python3
"""Run a small checkpoint round-robin and plot Dark Hex Elo estimates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

from record_dark_hex_matches import (  # noqa: E402
    CheckpointRef,
    _find_checkpoint_dirs,
    _load_cfg,
    _load_model,
)


DEFAULT_RUN_DIR = Path("outputs/2026-04-21/train_selfplay.yaml/dark_hex_7x7_fast_17-46-56")
DEFAULT_OUTPUT_DIR = Path("visualizations/article_charts")

PINK = "#e31b91"
BLACK = "#111111"
INK = "#24212c"
MUTED = "#706a7b"
GRID = "#e8e2d4"
PAPER = "#fffaf0"
TEAL = "#1f9d91"


@dataclass
class GameResult:
    checkpoint_a: int
    checkpoint_b: int
    a_color: str
    winner: str
    plies: int

    def to_json(self) -> dict[str, Any]:
        return {
            "checkpoint_a": self.checkpoint_a,
            "checkpoint_b": self.checkpoint_b,
            "a_color": self.a_color,
            "winner": self.winner,
            "plies": self.plies,
        }


def select_action(model, state, key, deterministic: bool) -> int:
    output = model(state.observation)
    masked_logits = jnp.where(state.legal_action_mask, output.policy_logits, -1e9)
    if deterministic:
        return int(jnp.argmax(masked_logits))
    return int(jax.random.categorical(key, masked_logits))


def force_colors(env, state, *, a_is_black: bool):
    # Player id 0 is checkpoint A, player id 1 is checkpoint B.
    player_order = jnp.array([0, 1], dtype=jnp.int32) if a_is_black else jnp.array([1, 0], dtype=jnp.int32)
    state = state.replace(
        _player_order=player_order,
        current_player=player_order[state._x.color],
    )
    return state.replace(observation=env.observe(state))


def play_game(
    *,
    env,
    model_a,
    model_b,
    checkpoint_a: int,
    checkpoint_b: int,
    a_is_black: bool,
    seed: int,
    max_steps: int,
    deterministic: bool,
) -> GameResult:
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    state = force_colors(env, env.init(init_key), a_is_black=a_is_black)
    models = {0: model_a, 1: model_b}

    plies = 0
    while not bool(np.asarray(state.terminated)) and not bool(np.asarray(state.truncated)) and plies < max_steps:
        player_id = int(np.asarray(state.current_player))
        key, action_key = jax.random.split(key)
        action = select_action(models[player_id], state, action_key, deterministic)
        state = env.step(state, jnp.int32(action))
        plies += 1

    rewards = np.asarray(state.rewards)
    if rewards[0] > rewards[1]:
        winner = "a"
    elif rewards[1] > rewards[0]:
        winner = "b"
    else:
        winner = "draw"

    return GameResult(
        checkpoint_a=checkpoint_a,
        checkpoint_b=checkpoint_b,
        a_color="black" if a_is_black else "white",
        winner=winner,
        plies=plies,
    )


def run_round_robin(
    *,
    run_dir: Path,
    games_per_color: int,
    seed: int,
    max_steps: int,
    deterministic: bool,
) -> dict[str, Any]:
    cfg = _load_cfg(run_dir)
    env = __import__("hydra.utils").utils.instantiate(cfg.env)
    refs = _find_checkpoint_dirs(run_dir)
    if len(refs) < 2:
        raise RuntimeError(f"Need at least two checkpoints below {run_dir}")

    models = {}
    for ref in refs:
        print(f"Loading checkpoint {ref.iteration} from {ref.path}", flush=True)
        models[ref.iteration] = _load_model(ref.path, cfg, env)

    results: list[GameResult] = []
    game_seed = seed
    for ref_a, ref_b in itertools.combinations(refs, 2):
        for a_is_black in (True, False):
            for _ in range(games_per_color):
                result = play_game(
                    env=env,
                    model_a=models[ref_a.iteration],
                    model_b=models[ref_b.iteration],
                    checkpoint_a=ref_a.iteration,
                    checkpoint_b=ref_b.iteration,
                    a_is_black=a_is_black,
                    seed=game_seed,
                    max_steps=max_steps,
                    deterministic=deterministic,
                )
                results.append(result)
                game_seed += 1
        wins_a = sum(1 for r in results if r.checkpoint_a == ref_a.iteration and r.checkpoint_b == ref_b.iteration and r.winner == "a")
        wins_b = sum(1 for r in results if r.checkpoint_a == ref_a.iteration and r.checkpoint_b == ref_b.iteration and r.winner == "b")
        draws = sum(1 for r in results if r.checkpoint_a == ref_a.iteration and r.checkpoint_b == ref_b.iteration and r.winner == "draw")
        print(f"{ref_a.iteration} vs {ref_b.iteration}: {wins_a}-{wins_b}-{draws}", flush=True)

    return {
        "run_dir": str(run_dir),
        "checkpoints": [ref.iteration for ref in refs],
        "games_per_color": games_per_color,
        "seed": seed,
        "max_steps": max_steps,
        "deterministic": deterministic,
        "results": [result.to_json() for result in results],
    }


def estimate_elos(payload: dict[str, Any], *, prior_scale: float = 500.0) -> dict[int, dict[str, float]]:
    checkpoints = [int(x) for x in payload["checkpoints"]]
    index = {checkpoint: i for i, checkpoint in enumerate(checkpoints)}
    results = payload["results"]

    def neg_log_posterior(params: np.ndarray) -> float:
        elos = np.concatenate([[0.0], params])
        loss = 0.5 * float(np.sum((elos / prior_scale) ** 2))
        for result in results:
            a = index[int(result["checkpoint_a"])]
            b = index[int(result["checkpoint_b"])]
            diff = elos[a] - elos[b]
            p_a = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
            p_a = min(max(p_a, 1e-8), 1.0 - 1e-8)
            if result["winner"] == "a":
                loss -= math.log(p_a)
            elif result["winner"] == "b":
                loss -= math.log(1.0 - p_a)
            else:
                loss -= 0.5 * (math.log(p_a) + math.log(1.0 - p_a))
        return loss

    res = minimize(neg_log_posterior, np.zeros(len(checkpoints) - 1), method="BFGS")
    rel = np.concatenate([[0.0], res.x])
    rel = rel - np.mean(rel)
    base = 1000.0

    if getattr(res, "hess_inv", None) is not None and hasattr(res.hess_inv, "shape"):
        # First checkpoint is anchored in the optimization, so report only a rough shared uncertainty.
        std = float(np.sqrt(np.mean(np.diag(res.hess_inv)))) if res.hess_inv.size else float("nan")
    else:
        std = float("nan")

    return {
        checkpoint: {
            "elo": float(base + rel[index[checkpoint]]),
            "relative_elo": float(rel[index[checkpoint]]),
            "rough_std": std,
        }
        for checkpoint in checkpoints
    }


def matchup_table(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for a, b in itertools.combinations(payload["checkpoints"], 2):
        subset = [
            result
            for result in payload["results"]
            if int(result["checkpoint_a"]) == int(a) and int(result["checkpoint_b"]) == int(b)
        ]
        wins_a = sum(1 for result in subset if result["winner"] == "a")
        wins_b = sum(1 for result in subset if result["winner"] == "b")
        draws = sum(1 for result in subset if result["winner"] == "draw")
        rows.append(
            {
                "checkpoint_a": int(a),
                "checkpoint_b": int(b),
                "games": len(subset),
                "wins_a": wins_a,
                "wins_b": wins_b,
                "draws": draws,
                "score_a": (wins_a + 0.5 * draws) / max(len(subset), 1),
            }
        )
    return rows


def plot_elo(payload: dict[str, Any], elos: dict[int, dict[str, float]], output_dir: Path) -> Path:
    checkpoints = [int(x) for x in payload["checkpoints"]]
    values = np.asarray([elos[checkpoint]["elo"] for checkpoint in checkpoints])
    rel = values - np.mean(values)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PAPER)
    colors = [BLACK if checkpoint != max(checkpoints) else PINK for checkpoint in checkpoints]
    bars = ax.bar([str(x) for x in checkpoints], values, color=colors, edgecolor=BLACK, linewidth=1.1)
    ax.plot([str(x) for x in checkpoints], values, color=PINK, linewidth=2.2, marker="o", zorder=3)
    for bar, value, relative in zip(bars, values, rel, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{value:.0f}\n{relative:+.0f}",
            ha="center",
            va="bottom",
            color=INK,
            fontweight="bold",
        )
    ax.set_facecolor(PAPER)
    ax.grid(True, axis="y", color=GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bdb5a5")
    ax.spines["bottom"].set_color("#bdb5a5")
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("checkpoint iteration", color=INK, fontweight="bold")
    ax.set_ylabel("estimated Elo", color=INK, fontweight="bold")
    ax.set_title("Dark Hex 7x7 Checkpoint Elo Estimate", color=INK, fontweight="bold", pad=14)
    ax.text(
        0.02,
        0.04,
        f"Round-robin: {payload['games_per_color']} games per color per pair, "
        f"{'greedy' if payload['deterministic'] else 'sampled'} policy.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
        fontweight="bold",
    )
    path = output_dir / "dark_hex_checkpoint_elo.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_relative_elo(payload: dict[str, Any], elos: dict[int, dict[str, float]], output_dir: Path) -> Path:
    checkpoints = [int(x) for x in payload["checkpoints"]]
    relative = np.asarray([elos[checkpoint]["relative_elo"] for checkpoint in checkpoints])
    rough_std = float(np.nanmean([elos[checkpoint]["rough_std"] for checkpoint in checkpoints]))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PAPER)
    ax.axhline(0, color=BLACK, linewidth=1.3, alpha=0.75)
    ax.fill_between(
        checkpoints,
        relative - rough_std,
        relative + rough_std,
        color=PINK,
        alpha=0.12,
        label=f"rough +/- {rough_std:.0f} Elo",
    )
    ax.plot(checkpoints, relative, color=PINK, linewidth=3.0, marker="o", markersize=7)
    ax.scatter(checkpoints[-1:], relative[-1:], color=PINK, edgecolor=BLACK, linewidth=1.4, s=110, zorder=4)

    for checkpoint, value in zip(checkpoints, relative, strict=True):
        ax.text(
            checkpoint,
            value + (7 if value >= 0 else -10),
            f"{value:+.0f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            color=INK,
            fontweight="bold",
            fontsize=10,
        )

    ax.set_facecolor(PAPER)
    ax.grid(True, color=GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bdb5a5")
    ax.spines["bottom"].set_color("#bdb5a5")
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("checkpoint iteration", color=INK, fontweight="bold")
    ax.set_ylabel("relative Elo vs checkpoint pool", color=INK, fontweight="bold")
    ax.set_title("Dark Hex 7x7 Checkpoint Strength Over Training", color=INK, fontweight="bold", pad=14)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#d8d0c0", loc="upper right")
    ax.text(
        0.02,
        0.035,
        f"Round-robin: {payload['games_per_color']} games per color per pair, "
        f"{'greedy' if payload['deterministic'] else 'sampled'} policy.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
        fontweight="bold",
    )

    y_min = min(-80.0, float(np.min(relative - rough_std)) - 10.0)
    y_max = max(80.0, float(np.max(relative + rough_std)) + 10.0)
    ax.set_ylim(y_min, y_max)
    path = output_dir / "dark_hex_checkpoint_relative_elo.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--games-per-color", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--reuse-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "dark_hex_checkpoint_round_robin.json"
    elo_path = args.output_dir / "dark_hex_checkpoint_elos.json"

    if args.reuse_results and results_path.exists():
        with results_path.open() as f:
            payload = json.load(f)
    else:
        payload = run_round_robin(
            run_dir=args.run_dir.resolve(),
            games_per_color=args.games_per_color,
            seed=args.seed,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
        )
        with results_path.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")

    elos = estimate_elos(payload)
    output = {
        "elos": elos,
        "matchups": matchup_table(payload),
        "note": "Elo is estimated from this local post-hoc round-robin, not from W&B training logs.",
    }
    with elo_path.open("w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    chart_path = plot_elo(payload, elos, args.output_dir)
    relative_chart_path = plot_relative_elo(payload, elos, args.output_dir)
    print(results_path)
    print(elo_path)
    print(chart_path)
    print(relative_chart_path)


if __name__ == "__main__":
    main()
