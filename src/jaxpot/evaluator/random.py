from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from jaxpot.agents.base_training_agent import BaseTrainingAgent
from jaxpot.agents.ppo_agent import PPOAgent
from jaxpot.evaluator.base import BaseEvaluator, EvaluationOutput, HistogramData
from jaxpot.evaluator.evaluate import evaluate_vs_random_jited


class RandomEvaluator(BaseEvaluator):
    def __init__(
        self,
        eval_every: int,
        num_envs: int,
        num_steps: int,
        init: Callable[..., Any],
        step_fn: Callable[..., Any],
        agent: BaseTrainingAgent,
        deterministic: bool = False,
        name: str = "random_eval",
        no_auto_reset_step_fn: Callable[..., Any] | None = None,
        **kwargs,
    ):
        super().__init__(eval_every, name, init, step_fn, agent)
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.deterministic = deterministic
        self.no_auto_reset_step_fn = no_auto_reset_step_fn

    def eval(self, key: jax.Array) -> EvaluationOutput:
        rollout_actor = self.agent.rollout_actor.setup(
            step_fn=self.step_fn, no_auto_reset_step_fn=self.no_auto_reset_step_fn
        )

        # Stochastic evaluation as seat 0
        results = evaluate_vs_random_jited(
            rollout_actor,
            key,
            self.init,
            self.step_fn,
            num_envs=self.num_envs,
            num_steps=self.num_steps,
            model_seat=0,
            deterministic=self.deterministic,
        )

        log_payload = self._build_metrics(results)

        histograms: list[HistogramData] = []
        ac_counts = results.get("action_counts")
        if ac_counts is not None:
            counts_array = np.asarray(ac_counts, dtype=np.int32)
            bins = np.arange(len(counts_array) + 1, dtype=np.float32) - 0.5
            histograms.append(
                HistogramData(
                    name=f"{self.name}/action_counts_hist",
                    counts_array=counts_array,
                    bins=bins,
                )
            )

        return EvaluationOutput(metrics=log_payload, histograms=histograms)
