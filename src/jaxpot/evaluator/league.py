from collections.abc import Callable
from typing import Any

import jax

from jaxpot.agents.base_training_agent import BaseTrainingAgent
from jaxpot.agents.policy_actor import PolicyActor
from jaxpot.evaluator.base import BaseEvaluator, EvaluationOutput
from jaxpot.evaluator.evaluate import evaluate_vs_opponent_jited
from jaxpot.league import LeagueManager


class ArchivedLeagueEvaluator(BaseEvaluator):
    def __init__(
        self,
        eval_every: int,
        num_envs: int,
        num_steps: int,
        init: Callable[..., Any],
        step_fn: Callable[..., Any],
        agent: BaseTrainingAgent,
        league: LeagueManager,
        name: str = "forgotten_eval",
        no_auto_reset_step_fn: Callable[..., Any] | None = None,
        **kwargs,
    ):
        super().__init__(eval_every, name, init, step_fn, agent)
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.league = league
        self.no_auto_reset_step_fn = no_auto_reset_step_fn

    def should_eval(self, it: int) -> bool:
        return super().should_eval(it) and len(self.league.archive) > 0

    def eval(self, key: jax.Array) -> EvaluationOutput:
        eval_keys = jax.random.split(key, len(self.league.archive))
        rollout_actor = self.agent.rollout_actor.setup(
            step_fn=self.step_fn, no_auto_reset_step_fn=self.no_auto_reset_step_fn
        )

        for i, entry in enumerate(self.league.archive):
            res = evaluate_vs_opponent_jited(
                rollout_actor,
                PolicyActor(model=entry.model),
                eval_keys[i],
                self.init,
                self.step_fn,
                num_envs=self.num_envs,
                num_steps=self.num_steps,
                model_seat=0,
            )
            avg_r = float(res.get("avg_reward", 0.0))
            self.league.update_archive_score(i, avg_r, int(res["num_games"]))

        return EvaluationOutput()
