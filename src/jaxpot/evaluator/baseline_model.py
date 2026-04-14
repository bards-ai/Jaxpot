from collections.abc import Callable
from typing import Any

import jax

from jaxpot.agents.base_training_agent import BaseTrainingAgent
from jaxpot.agents.policy_actor import PolicyActor
from jaxpot.evaluator.base import BaseEvaluator, EvaluationOutput
from jaxpot.evaluator.evaluate import evaluate_vs_opponent_jited
from jaxpot.evaluator.utils import calculate_elo
from jaxpot.models.base import PolicyValueModel


class BaselineModelEvaluator(BaseEvaluator):
    """Evaluate the current agent against a fixed baseline opponent.

    Parameters
    ----------
    baseline_model : PolicyValueModel
        Opponent model instantiated from Hydra config.
    eval_every : int
        Evaluation frequency in training iterations.
    num_envs : int
        Number of parallel environments to run.
    num_steps : int
        Maximum rollout length for each evaluation run.
    init : Callable[..., Any]
        Environment reset function.
    step_fn : Callable[..., Any]
        Environment transition function.
    agent : PPOAgent
        Training agent whose model is being evaluated.
    name : str, default="baseline_eval"
        Metric prefix used for logged results.
    model_seat : int, default=0
        Seat controlled by the training model during evaluation.
    deterministic : bool, default=False
        Whether to use greedy action selection for the training model.
    include_elo : bool, default=True
        Whether to append Elo computed from win and draw rates.
    """

    def __init__(
        self,
        baseline_model: PolicyValueModel,
        eval_every: int,
        num_envs: int,
        num_steps: int,
        init: Callable[..., Any],
        step_fn: Callable[..., Any],
        agent: BaseTrainingAgent,
        name: str = "baseline_eval",
        model_seat: int = 0,
        deterministic: bool = False,
        include_elo: bool = True,
        no_auto_reset_step_fn: Callable[..., Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(eval_every, name, init, step_fn, agent)
        self.baseline_model = baseline_model
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.model_seat = model_seat
        self.deterministic = deterministic
        self.include_elo = include_elo
        self.no_auto_reset_step_fn = no_auto_reset_step_fn

    def eval(self, key: jax.Array) -> EvaluationOutput:
        rollout_actor = self.agent.rollout_actor.setup(
            step_fn=self.step_fn, no_auto_reset_step_fn=self.no_auto_reset_step_fn
        )
        results = evaluate_vs_opponent_jited(
            rollout_actor,
            PolicyActor(model=self.baseline_model),
            key,
            self.init,
            self.step_fn,
            self.num_envs,
            self.num_steps,
            model_seat=self.model_seat,
            deterministic=self.deterministic,
        )

        log_payload = self._build_metrics(results)
        if self.include_elo:
            log_payload[f"{self.name}/elo"] = float(
                calculate_elo(results["win_rate"], results["draw_rate"])
            )

        return EvaluationOutput(metrics=log_payload)
