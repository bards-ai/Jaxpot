#!/usr/bin/env python3
"""Generate article charts for the Dark Hex 7x7 self-play run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN_DIR = Path("outputs/2026-04-21/train_selfplay.yaml/dark_hex_7x7_fast_17-46-56")
DEFAULT_REPLAYS = Path("visualizations/dark_hex_7x7_matches.json")
DEFAULT_OUTPUT_DIR = Path("visualizations/article_charts")

PINK = "#e31b91"
BLACK = "#111111"
INK = "#24212c"
MUTED = "#706a7b"
GRID = "#e8e2d4"
TEAL = "#1f9d91"
AMBER = "#d99b2b"
PAPER = "#fffaf0"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric_series(rows: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        if metric in row and "iteration" in row:
            xs.append(float(row["iteration"]))
            ys.append(float(row[metric]))
    return np.asarray(xs), np.asarray(ys)


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window) / window
    padded = np.pad(y, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def style_axes(ax, *, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(PAPER)
    ax.grid(True, color=GRID, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bdb5a5")
    ax.spines["bottom"].set_color("#bdb5a5")
    ax.tick_params(colors=MUTED)
    ax.set_xlabel(xlabel, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK, fontweight="bold")


def save(fig, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_random_saturation(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    x_win, y_win = metric_series(rows, "eval_vs_random/win_rate")
    x_det, y_det = metric_series(rows, "eval_vs_random_deterministic/win_rate")
    x_loss, y_loss = metric_series(rows, "eval_vs_random/lose_rate")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PAPER)
    ax.plot(x_win, y_win * 100, color=PINK, linewidth=2.8, marker="o", label="stochastic policy")
    ax.plot(x_det, y_det * 100, color=BLACK, linewidth=2.2, marker="s", label="greedy policy")
    ax.plot(x_loss, y_loss * 100, color=AMBER, linewidth=1.8, linestyle="--", label="loss rate vs random")
    ax.axvline(500, color=TEAL, linewidth=1.8, linestyle=":", alpha=0.9)
    ax.text(510, 11, "random is already solved", color=TEAL, fontweight="bold", va="bottom")
    ax.set_ylim(-2, 104)
    ax.set_title("Dark Hex 7x7: Random Baseline Saturates Early", color=INK, fontweight="bold", pad=14)
    style_axes(ax, xlabel="training iteration", ylabel="rate (%)")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#d8d0c0")
    return save(fig, output_dir, "dark_hex_random_saturation.png")


def plot_learning_after_500(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    x_total, y_total = metric_series(rows, "total_loss")
    x_value, y_value = metric_series(rows, "value_loss")
    x_policy, y_policy = metric_series(rows, "policy_loss")

    mask_total = x_total >= 500
    mask_value = x_value >= 500
    mask_policy = x_policy >= 500

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PAPER)
    ax.plot(x_total[mask_total], y_total[mask_total], color=BLACK, linewidth=0.9, alpha=0.16)
    ax.plot(x_value[mask_value], y_value[mask_value], color=PINK, linewidth=0.9, alpha=0.16)
    ax.plot(
        x_total[mask_total],
        smooth(y_total[mask_total], 75),
        color=BLACK,
        linewidth=2.6,
        label="total loss, moving average",
    )
    ax.plot(
        x_value[mask_value],
        smooth(y_value[mask_value], 75),
        color=PINK,
        linewidth=2.6,
        label="value loss, moving average",
    )
    ax.plot(
        x_policy[mask_policy],
        smooth(y_policy[mask_policy], 75),
        color=TEAL,
        linewidth=1.8,
        alpha=0.8,
        label="policy loss, moving average",
    )
    for xs, ys, color in (
        (x_total[mask_total], y_total[mask_total], BLACK),
        (x_value[mask_value], y_value[mask_value], PINK),
    ):
        if len(xs) > 1:
            m, b = np.polyfit(xs, ys, 1)
            ax.plot(xs, m * xs + b, color=color, linewidth=1.6, linestyle=":", alpha=0.85)
    ax.set_title("Learning Continues After the Random Baseline Stops Helping", color=INK, fontweight="bold", pad=14)
    style_axes(ax, xlabel="training iteration", ylabel="loss")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#d8d0c0")
    return save(fig, output_dir, "dark_hex_learning_after_500.png")


def plot_value_loss_with_random_context(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    x_value, y_value = metric_series(rows, "value_loss")
    x_win, y_win = metric_series(rows, "eval_vs_random/win_rate")

    fig, ax1 = plt.subplots(figsize=(9, 5), facecolor=PAPER)
    ax1.plot(x_value, y_value, color=PINK, linewidth=0.9, alpha=0.14)
    ax1.plot(x_value, smooth(y_value, 75), color=PINK, linewidth=2.8, label="value loss moving average")
    ax1.axvspan(500, max(x_value), color=PINK, alpha=0.06)
    style_axes(ax1, xlabel="training iteration", ylabel="value loss")

    ax2 = ax1.twinx()
    ax2.plot(x_win, y_win * 100, color=BLACK, linewidth=2.2, marker="o", label="win rate vs random")
    ax2.set_ylabel("random win rate (%)", color=INK, fontweight="bold")
    ax2.tick_params(colors=MUTED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#bdb5a5")
    ax2.set_ylim(45, 103)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, frameon=True, facecolor="#ffffff", edgecolor="#d8d0c0", loc="upper right")
    ax1.set_title("Value Head Keeps Improving While Random Win Rate Is Flat", color=INK, fontweight="bold", pad=14)
    return save(fig, output_dir, "dark_hex_value_loss_vs_random.png")


def plot_sparse_checkpoint_results(replay_path: Path, output_dir: Path) -> Path | None:
    if not replay_path.exists():
        return None
    with replay_path.open() as f:
        payload = json.load(f)

    latest_iter = int(payload["latest_iteration"])
    games = payload.get("games", [])
    if not games:
        return None

    by_opponent: dict[int, list[int]] = defaultdict(list)
    for game in games:
        opponent = int(game["opponent_checkpoint"]["iteration"])
        winner = game["result"].get("winner_label") or ""
        by_opponent[opponent].append(1 if f"latest iter {latest_iter}" in winner else 0)

    opponents = sorted(by_opponent)
    scores = np.asarray([np.mean(by_opponent[opponent]) * 100 for opponent in opponents])
    counts = np.asarray([len(by_opponent[opponent]) for opponent in opponents])

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PAPER)
    ax.bar([str(opponent) for opponent in opponents], scores, color=PINK, edgecolor=BLACK, linewidth=1.2)
    for idx, (score, count) in enumerate(zip(scores, counts, strict=True)):
        ax.text(idx, score + 3, f"{score:.0f}%\n{count} games", ha="center", color=INK, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_title("Sparse Checkpoint Matchups from Recorded Replays", color=INK, fontweight="bold", pad=14)
    style_axes(ax, xlabel="opponent checkpoint iteration", ylabel=f"latest {latest_iter} win rate (%)")
    ax.text(
        0.02,
        0.04,
        "This is real replay data, but too sparse to call Elo.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
        fontweight="bold",
    )
    return save(fig, output_dir, "dark_hex_sparse_checkpoint_results.png")


def plot_article_panel(rows: list[dict[str, Any]], replay_path: Path, output_dir: Path) -> Path:
    x_win, y_win = metric_series(rows, "eval_vs_random/win_rate")
    x_value, y_value = metric_series(rows, "value_loss")
    x_total, y_total = metric_series(rows, "total_loss")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PAPER)

    axes[0].plot(x_win, y_win * 100, color=PINK, linewidth=2.8, marker="o")
    axes[0].axvline(500, color=TEAL, linewidth=1.8, linestyle=":")
    axes[0].set_ylim(45, 103)
    axes[0].set_title("Random Evaluation Plateaus", color=INK, fontweight="bold", pad=12)
    style_axes(axes[0], xlabel="training iteration", ylabel="win rate vs random (%)")

    mask_value = x_value >= 500
    mask_total = x_total >= 500
    axes[1].plot(x_value[mask_value], y_value[mask_value], color=PINK, linewidth=0.8, alpha=0.12)
    axes[1].plot(x_total[mask_total], y_total[mask_total], color=BLACK, linewidth=0.8, alpha=0.12)
    axes[1].plot(x_value[mask_value], smooth(y_value[mask_value], 75), color=PINK, linewidth=2.8, label="value loss")
    axes[1].plot(x_total[mask_total], smooth(y_total[mask_total], 75), color=BLACK, linewidth=2.4, label="total loss")
    for xs, ys, color in (
        (x_value[mask_value], y_value[mask_value], PINK),
        (x_total[mask_total], y_total[mask_total], BLACK),
    ):
        if len(xs) > 1:
            m, b = np.polyfit(xs, ys, 1)
            axes[1].plot(xs, m * xs + b, color=color, linewidth=1.5, linestyle=":", alpha=0.85)
    axes[1].set_title("Training Signal Keeps Moving", color=INK, fontweight="bold", pad=12)
    style_axes(axes[1], xlabel="training iteration", ylabel="loss after step 500")
    axes[1].legend(frameon=True, facecolor="#ffffff", edgecolor="#d8d0c0")

    fig.suptitle("Dark Hex 7x7 Self-Play: Random Is Solved Before Learning Is Done", color=INK, fontweight="bold", fontsize=16)
    fig.tight_layout()
    return save(fig, output_dir, "dark_hex_article_panel.png")


def write_summary(rows: list[dict[str, Any]], replay_path: Path, output_dir: Path) -> Path:
    x_win, y_win = metric_series(rows, "eval_vs_random/win_rate")
    x_loss, y_loss = metric_series(rows, "total_loss")
    x_value, y_value = metric_series(rows, "value_loss")

    def at_or_after(xs: np.ndarray, ys: np.ndarray, target: int) -> float | None:
        indices = np.where(xs >= target)[0]
        if len(indices) == 0:
            return None
        return float(ys[indices[0]])

    summary = {
        "run": str(DEFAULT_RUN_DIR),
        "random_win_rate_at_500": at_or_after(x_win, y_win, 500),
        "random_win_rate_last_eval": float(y_win[-1]) if len(y_win) else None,
        "total_loss_at_500": at_or_after(x_loss, y_loss, 500),
        "total_loss_last": float(y_loss[-1]) if len(y_loss) else None,
        "value_loss_at_500": at_or_after(x_value, y_value, 500),
        "value_loss_last": float(y_value[-1]) if len(y_value) else None,
        "replay_file": str(replay_path),
        "note": "Checkpoint Elo needs a larger round-robin or more saved milestones. These charts use real logged metrics and sparse replay outcomes only.",
    }
    path = output_dir / "dark_hex_chart_summary.json"
    with path.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--replays", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.run_dir / "output.jsonl")

    paths = [
        plot_random_saturation(rows, args.output_dir),
        plot_learning_after_500(rows, args.output_dir),
        plot_value_loss_with_random_context(rows, args.output_dir),
        plot_article_panel(rows, args.replays, args.output_dir),
        write_summary(rows, args.replays, args.output_dir),
    ]
    sparse_path = plot_sparse_checkpoint_results(args.replays, args.output_dir)
    if sparse_path is not None:
        paths.append(sparse_path)

    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
