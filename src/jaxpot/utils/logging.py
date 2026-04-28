from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from jaxpot.utils.timer import Timer

if TYPE_CHECKING:
    from jaxpot.league import LeagueManager

_console = Console()

# Module-level handle for the loguru stderr handler so we can remove/restore it.
_stderr_handler_id: int | None = None


def setup_loguru_logging(cfg: DictConfig) -> None:
    """Configure loguru to log to both stderr and Hydra's run directory."""
    global _stderr_handler_id

    run_dir = Path(HydraConfig.get().runtime.output_dir)

    log_file_name = cfg.loguru.get("log_file_name", "train_selfplay.log")
    log_level = cfg.loguru.get("log_level", "INFO")
    log_file_path = run_dir / str(log_file_name)

    logger.remove()
    _stderr_handler_id = logger.add(sys.stderr, level=log_level)
    logger.add(log_file_path, level=log_level)

    # Suppress noisy absl INFO logs (Orbax checkpoint spam)
    logging.getLogger("absl").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _loss_color(value: float, lo: float = 0.1, hi: float = 1.0) -> str:
    if value < lo:
        return "green"
    if value < hi:
        return "yellow"
    return "red"


def _format_eta(seconds: float) -> str:
    """Format seconds as ETA string (e.g. '12m 34s' or '1h 2m')."""
    if seconds < 0 or not (seconds < 1e9):
        return "?"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"


def _fmt(value: float, width: int = 6) -> str:
    """Auto-format a float for dashboard display."""
    abs_v = abs(value)
    if abs_v == 0:
        return "0".rjust(width)
    if abs_v < 0.001:
        return f"{value:.5f}".rjust(width)
    if abs_v < 1:
        return f"{value:.4f}".rjust(width)
    if abs_v < 100:
        return f"{value:.2f}".rjust(width)
    return f"{value:.0f}".rjust(width)


# ---------------------------------------------------------------------------
# Eval metric helpers
# ---------------------------------------------------------------------------

# Prefixes in log_payload that are NOT eval results
_NON_EVAL_PREFIXES = {"iteration", "timings", "train_vs_random"}

# Timer keys to hide from the performance column
_HIDDEN_TIMER_KEYS = {
    "total",
    "concatenate_batches",
    "transfer_metrics",
    "dump_debug_file",
    "log_league_standings",
}

# Ordered list of timer keys to show first in the performance column
_TIMER_DISPLAY_ORDER = [
    ("training", "Train"),
    ("collect_selfplay", "Selfplay"),
    ("collect_random", "Random"),
    ("collect_league", "League"),
    ("collect_archive_league", "Archive"),
]


def _is_eval_metric(key: str) -> bool:
    """Return True if *key* looks like an eval metric (has ``/`` and prefix not in exclusion set)."""
    if "/" not in key:
        return False
    prefix = key.split("/", 1)[0]
    return prefix not in _NON_EVAL_PREFIXES


# ---------------------------------------------------------------------------
# Dashboard sub-tables
# ---------------------------------------------------------------------------


def _build_training_column(metrics: dict, log_payload: dict) -> Table:
    """Left column: training losses and sample stats."""
    t = Table(show_header=True, show_edge=False, padding=(0, 2), box=None)
    t.add_column("Training", style="bold cyan", min_width=18)
    t.add_column("", justify="right", min_width=10)

    priority_keys = ("value_loss", "policy_loss")
    shown_keys: set[str] = set()

    for key in priority_keys:
        if key not in metrics:
            continue
        value = float(metrics[key])
        color = _loss_color(value)
        label = "Value Loss" if key == "value_loss" else "Policy Loss"
        t.add_row(Text(label, style=color), Text(_fmt(value), style=color))
        shown_keys.add(key)

    for key, raw_value in metrics.items():
        if key in shown_keys:
            continue
        try:
            value = float(raw_value)
            t.add_row(Text(key, style="dim"), Text(_fmt(value), style="dim"))
        except (TypeError, ValueError):
            t.add_row(Text(key, style="dim"), Text(str(raw_value), style="dim"))

    num_samples = log_payload.get("iteration/num_valid_samples")
    if num_samples is not None:
        t.add_row(Text("Samples", style="dim"), Text(f"{int(num_samples):,}", style="dim"))
    valid_pct = log_payload.get("iteration/valid_percentage")
    if valid_pct is not None:
        t.add_row(Text("Valid", style="dim"), Text(f"{valid_pct:.1%}", style="dim"))

    return t


def _build_performance_column(timings: dict) -> Table:
    """Middle column: timing breakdown."""
    t = Table(show_header=True, show_edge=False, padding=(0, 2), box=None)
    total_data = timings.get("total")
    header = "Performance"
    if total_data is not None:
        mean = total_data.get("mean", 0)
        header = f"Performance ({mean:.2f}s/iter)"
    t.add_column(header, style="bold cyan", min_width=14)
    t.add_column("", justify="right", min_width=8)

    shown_keys: set[str] = set()

    # Show core timers in fixed order
    for key, label in _TIMER_DISPLAY_ORDER:
        data = timings.get(key)
        if data is not None:
            mean = data.get("mean", 0)
            t.add_row(Text(label, style="dim"), Text(f"{mean:.2f}s", style="dim"))
            shown_keys.add(key)

    # Show remaining timer keys (eval timers, etc.) — skip hidden system keys
    for key in sorted(timings.keys()):
        if key in shown_keys or key in _HIDDEN_TIMER_KEYS:
            continue
        data = timings[key]
        mean = data.get("mean", 0)
        label = key
        t.add_row(Text(label, style="dim"), Text(f"{mean:.2f}s", style="dim"))

    return t


def _build_eval_column(log_payload: dict) -> Table:
    """Evaluation results grouped by prefix."""
    t = Table(show_header=True, show_edge=False, padding=(0, 2), box=None)
    t.add_column("Evaluation", style="bold cyan", min_width=26)
    t.add_column("", justify="right", min_width=1)

    # Dynamic eval group discovery: scan all keys with _is_eval_metric
    eval_groups: dict[str, dict[str, float]] = {}
    for k, v in log_payload.items():
        if not isinstance(v, (int, float)):
            continue
        if not _is_eval_metric(k):
            continue
        prefix, metric_name = k.split("/", 1)
        eval_groups.setdefault(prefix, {})[metric_name] = float(v)

    if not eval_groups:
        t.add_row(Text("Waiting...", style="dim"), Text(""))
        return t

    # Skip per-player sub-metrics when displaying win/lose
    _SKIP_PREFIXES = ("p0/", "p1/")

    for group, metrics_dict in sorted(eval_groups.items()):
        t.add_row(Text(f"  {group}", style="bold"), Text(""))

        # Show win/lose/draw rates prominently on one line
        win = metrics_dict.get("win_rate")
        lose = metrics_dict.get("lose_rate")
        draw = metrics_dict.get("draw_rate")
        if win is not None or lose is not None:
            parts = []
            if win is not None:
                c = "green" if win > 0.9 else "yellow" if win > 0.7 else "red"
                parts.append(f"[{c}]W {win:.3%}[/]")
            if lose is not None:
                c = "green" if lose < 0.01 else "yellow" if lose < 0.1 else "red"
                parts.append(f"[{c}]L {lose:.3%}[/]")
            if draw is not None:
                parts.append(f"[dim]D {draw:.3%}[/]")
            t.add_row(Text.from_markup("    " + "  ".join(parts)), Text(""))

        # Show remaining metrics (skip win/lose/draw already shown and per-player sub-metrics)
        shown = {"win_rate", "lose_rate", "draw_rate"}
        for mk, mv in sorted(metrics_dict.items()):
            if mk in shown or any(mk.startswith(p) for p in _SKIP_PREFIXES):
                continue
            t.add_row(Text(f"    {mk}", style="dim"), Text(_fmt(mv), style="dim"))

    return t


def _build_league_column(league: LeagueManager) -> Table:
    """League + archive standings as a separate column."""
    t = Table(show_header=True, show_edge=False, padding=(0, 2), box=None)
    t.add_column("League", style="bold cyan", min_width=20)
    t.add_column("", justify="right", min_width=8)

    if league.size() > 0:
        for entry in league.entries:
            name = entry.name or "agent"
            score = entry.score_ema
            c = "green" if score > 0 else "red" if score < 0 else "dim"
            t.add_row(Text(f"  {name}", style="dim"), Text(f"{score:+.3f}", style=c))

    if len(league.archive) > 0:
        num_active = league.num_active_archive()
        total = len(league.archive)
        t.add_row(Text(""), Text(""))
        t.add_row(Text(f"  Archive ({num_active}/{total})", style="bold"), Text(""))
        for entry in league.archive:
            name = entry.name or "agent"
            score = entry.score_ema
            c = "green" if score > 0 else "red" if score < 0 else "dim"
            active_marker = " *" if entry.active else ""
            t.add_row(
                Text(f"  {name}{active_marker}", style="dim"),
                Text(f"{score:+.3f}", style=c),
            )

    return t


def _build_dashboard(
    it: int,
    total_iters: int,
    metrics: dict,
    log_payload: dict,
    timer: Timer,
    experiment_name: str = "jaxpot",
    league: LeagueManager | None = None,
) -> Panel:
    """Build a PufferLib-style Rich Table dashboard."""
    pct = (it + 1) / total_iters * 100
    timings = timer.get_stats()
    t_total = timings.get("total", {}).get("mean", 0)

    # SPS: samples per second
    sps = log_payload.get("iteration/sps")
    sps_str = f"  SPS {sps:,}" if sps is not None else ""

    # Elapsed / ETA
    elapsed_str = ""
    eta_str = ""
    t_cumulative = timings.get("total", {}).get("cumulative", 0)
    if t_cumulative > 0:
        elapsed_str = f"  Elapsed {_format_eta(t_cumulative)}"
    if t_total > 0:
        remaining_iters = total_iters - (it + 1)
        eta_str = f"  ETA {_format_eta(remaining_iters * t_total)}"

    # Header line
    header = (
        f"[bold bright_cyan]{experiment_name}[/]  "
        f"Iter [bold cyan]{it + 1:,}[/][dim]/{total_iters:,}[/] "
        f"[dim]({pct:.1f}%)[/]"
        f"[dim]{elapsed_str}{eta_str}{sps_str}[/]"
    )

    # Progress bar — scale to terminal width (minus panel borders + padding)
    term_w = _console.width or 80
    bar_w = max(20, term_w - 8)
    filled = int(bar_w * (it + 1) / total_iters)
    progress_bar = f"[cyan]{'━' * filled}[/][dim]{'─' * (bar_w - filled)}[/]"

    # Build column layout — add league as 4th column when present
    has_league = league is not None and (league.size() > 0 or len(league.archive) > 0)
    layout = Table(show_header=False, show_edge=False, box=None, padding=(0, 1))
    layout.add_column(ratio=1)
    layout.add_column(ratio=1)
    layout.add_column(ratio=1)
    if has_league:
        layout.add_column(ratio=1)
        layout.add_row(
            _build_training_column(metrics, log_payload),
            _build_performance_column(timings),
            _build_eval_column(log_payload),
            _build_league_column(league),
        )
    else:
        layout.add_row(
            _build_training_column(metrics, log_payload),
            _build_performance_column(timings),
            _build_eval_column(log_payload),
        )

    # Outer panel — compact vertically for short terminals
    outer = Table(show_header=False, show_edge=False, box=None, padding=0)
    outer.add_column()
    outer.add_row(Text.from_markup(f"  {header}"))
    outer.add_row(Text.from_markup(f"  {progress_bar}"))
    outer.add_row(layout)

    return Panel(outer, border_style="bright_cyan", expand=True)


# ---------------------------------------------------------------------------
# TrainingProgress — Live dashboard
# ---------------------------------------------------------------------------


class TrainingProgress:
    """Live-updating training progress display (PufferLib-style dashboard)."""

    def __init__(self, total_iters: int, experiment_name: str = "jaxpot"):
        self.total_iters = total_iters
        self.experiment_name = experiment_name
        self._live = Live(console=_console, refresh_per_second=4, transient=False)
        self._last_eval_payload: dict = {}
        self._cleared = False

    def start(self) -> None:
        global _stderr_handler_id
        # Suppress loguru stderr while Live display is active (file logging continues)
        if _stderr_handler_id is not None:
            try:
                logger.remove(_stderr_handler_id)
            except ValueError:
                pass
            _stderr_handler_id = None
        _console.print("[dim]Waiting for first iteration...[/]")
        self._live.start()

    def stop(self) -> None:
        global _stderr_handler_id
        self._live.stop()
        # Restore loguru stderr handler so post-training logs still show
        if _stderr_handler_id is None:
            _stderr_handler_id = logger.add(sys.stderr, level="INFO")

    def update(
        self,
        it: int,
        metrics: dict,
        log_payload: dict,
        timer: Timer,
        league: LeagueManager | None = None,
    ) -> None:
        # Clear terminal on first real update so the dashboard replaces the waiting message
        if not self._cleared:
            _console.clear()
            self._cleared = True

        # Cache eval results so they persist between eval iterations
        for k, v in log_payload.items():
            if _is_eval_metric(k):
                self._last_eval_payload[k] = v

        # Merge cached eval data into a display payload
        display_payload = dict(log_payload)
        for k, v in self._last_eval_payload.items():
            display_payload.setdefault(k, v)

        dashboard = _build_dashboard(
            it,
            self.total_iters,
            metrics,
            display_payload,
            timer,
            self.experiment_name,
            league=league,
        )
        self._live.update(dashboard)


# ---------------------------------------------------------------------------
# JsonLinesLogger — structured log file for LLM parsing
# ---------------------------------------------------------------------------


class JsonLinesLogger:
    """Append-only JSON-lines logger. One line per training iteration."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._file = open(self._path, "a", buffering=1)  # line-buffered

    def log(
        self,
        iteration: int,
        log_payload: dict,
        metrics: dict,
        timer: Timer,
    ) -> None:
        record: dict = {
            "iteration": iteration,
            "timestamp": time.time(),
        }

        # Add training metrics (all scalars)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                record[k] = v

        # Add log_payload scalars only (skip arrays/objects)
        for k, v in log_payload.items():
            if isinstance(v, (int, float)):
                record[k] = v

        # Add timings
        timings = timer.get_stats()
        record["timings"] = {name: data["mean"] for name, data in timings.items()}

        self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
