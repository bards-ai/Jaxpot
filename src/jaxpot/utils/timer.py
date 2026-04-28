from collections import defaultdict, deque
from contextlib import contextmanager
from time import perf_counter

from loguru import logger


class Timer:
    def __init__(self, max_history: int = 1):
        """
        Timer with sliding window statistics.

        Parameters
        ----------
        max_history : int, optional
            Maximum number of recent measurements to keep per timer name.
        """
        if max_history <= 0:
            raise ValueError("max_history must be positive")

        self.max_history = int(max_history)
        self.timings: defaultdict[str, deque[float]] = defaultdict(self._make_deque)
        self._cumulative: defaultdict[str, float] = defaultdict(float)
        self._active_timers: dict[str, float] = {}

    def _make_deque(self) -> deque[float]:
        """Create a new deque for a timer name."""
        return deque(maxlen=self.max_history)

    def _record(self, name: str, elapsed: float) -> None:
        """Record a completed duration for a named operation."""
        self.timings[name].append(elapsed)
        self._cumulative[name] += elapsed

    @contextmanager
    def __call__(self, name: str):
        """Allow using timer as context manager with a name: with timer('operation'):"""
        start = perf_counter()
        try:
            yield
        finally:
            self._record(name, perf_counter() - start)

    def __enter__(self):
        raise RuntimeError("Timer must be called with a name: with timer('name'):")

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise RuntimeError("Timer must be called with a name: with timer('name'):")

    def start(self, name: str):
        """Start timing for a named operation."""
        self._active_timers[name] = perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing for a named operation."""
        if name not in self._active_timers:
            raise ValueError(f"Timer '{name}' was not started")
        elapsed = perf_counter() - self._active_timers[name]
        self._record(name, elapsed)
        del self._active_timers[name]
        return elapsed

    def reset(self):
        """Reset all timings."""
        self.timings.clear()
        self._active_timers.clear()

    def get_stats(self) -> dict[str, dict[str, float]]:
        """
        Get timing statistics over the recent history window.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionary with timing statistics for each operation computed from
            at most the last ``max_history`` recorded durations.
        """
        stats: dict[str, dict[str, float]] = {}
        for name, samples in self.timings.items():
            count = len(samples)
            total = float(sum(samples))
            stats[name] = {
                "total": total,
                "count": count,
                "mean": total / count if count > 0 else 0.0,
                "cumulative": self._cumulative[name],
            }
        return stats

    def print_stats(self):
        """Print timing statistics in a readable format."""
        stats = self.get_stats()
        if not stats:
            logger.warning("No timing data collected")
            return

        logger.info("\n" + "=" * 70)
        logger.info("Timing Statistics")
        logger.info("=" * 70)
        logger.info(f"{'Operation':<30} {'Count':>8} {'Total':>12} {'Mean':>12}")
        logger.info("-" * 70)

        for name, data in sorted(stats.items()):
            logger.info(
                f"{name:<30} {data['count']:>8} {data['total']:>12.6f}s {data['mean']:>12.6f}s"
            )
        logger.info("=" * 70 + "\n")
