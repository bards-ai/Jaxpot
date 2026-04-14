from typing import NamedTuple, override

import jax.numpy as jnp

from jaxpot.agents.base_training_agent import BaseTrainingAgent
from jaxpot.agents.policy_actor import PolicyActor
from jaxpot.models.base import PolicyValueModel
from jaxpot.rl.ppo_trainer import PPOTrainer
from jaxpot.rollout.buffer import TrainingDataBuffer


class PPOUpdateMetrics(NamedTuple):
    """Metrics returned from PPO update."""

    value_loss: float
    policy_loss: float
    entropy: float
    kl_divergence: float
    equity_loss: float
    opp_card_loss: float


class PPOAgent(BaseTrainingAgent):
    """
    PPO training wrapper.

    Bundles a model with a PPOTrainer and a PolicyActor used for rollouts.
    The actor is the JIT-compatible PyTree exposed via ``self.rollout_actor``.
    """

    _trainer: PPOTrainer

    def __init__(self, model: PolicyValueModel, trainer: PPOTrainer):
        super().__init__(model, trainer, PolicyActor(model=model))

    @override
    def update(
        self,
        training_data: dict[str, jnp.ndarray] | TrainingDataBuffer,
    ) -> dict:
        return super().update(training_data)
