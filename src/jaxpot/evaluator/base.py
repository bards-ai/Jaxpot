from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import pgx

from jaxpot.agents.base_training_agent import BaseTrainingAgent


@dataclass
class HistogramData:
    name: str
    counts_array: Any
    bins: Any


@dataclass
class EvaluationOutput:
    metrics: dict[str, Any] = field(default_factory=dict)
    histograms: list[HistogramData] = field(default_factory=list)


class BaseEvaluator:
    def __init__(
        self,
        eval_every: int,
        name: str,
        init: Callable[[jax.Array], pgx.State],
        step_fn: Callable[[pgx.State, jax.Array], pgx.State],
        agent: BaseTrainingAgent,
    ):
        self.eval_every = eval_every
        self.name = name
        self.init = init
        self.step_fn = step_fn
        self.agent = agent

    def should_eval(self, it: int) -> bool:
        """Whether to run evaluation at this iteration. Override for custom logic."""
        return self.eval_every > 0 and it % self.eval_every == 0

    def _build_metrics(self, results: dict) -> dict[str, Any]:
        """Build standardized metrics from evaluation results.

        Uses wandb-friendly ``{self.name}/...`` hierarchy:
          - ``{name}/win_rate``, ``{name}/lose_rate``, etc. for overall stats
          - ``{name}/p0/win_rate``, ``{name}/p1/lose_rate``, etc. for per-player stats
        """
        prefix = self.name
        metrics: dict[str, Any] = {}

        for key in ("avg_reward", "win_rate", "lose_rate", "draw_rate", "done_rate"):
            if key in results:
                metrics[f"{prefix}/{key}"] = float(results[key])

        for player in ("p0", "p1"):
            player_stats = results.get(player)
            if player_stats is None:
                continue
            for key in ("win_rate", "lose_rate", "draw_rate", "avg_reward"):
                if key in player_stats:
                    metrics[f"{prefix}/{player}/{key}"] = float(player_stats[key])

        return metrics

    @abstractmethod
    def eval(self, key: jax.Array) -> EvaluationOutput:
        raise NotImplementedError
