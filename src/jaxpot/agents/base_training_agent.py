"""
BaseTrainingAgent: abstract interface for stateful training wrappers.

Separates the two concerns that all training agents share:
  1. Providing a JAX-PyTree rollout actor for JIT-compiled data collection.
  2. Performing model updates from a TrainingDataBuffer.

Both PPOAgent and AlphaZeroAgent implement this interface, making
train_selfplay.py fully agent-agnostic: it calls `rollout_actor()` to get
whatever PyTree actor is appropriate (PolicyActor, MCTSActor, …) without
knowing which algorithm is running.
"""

from __future__ import annotations

from abc import abstractmethod

import jax
import jax.numpy as jnp

from jaxpot.agents.base_rollout_actor import AgentOutput, BaseRolloutActor
from jaxpot.models.base import PolicyValueModel
from jaxpot.rl.trainer import Trainer
from jaxpot.rollout.buffer import TrainingDataBuffer


class BaseTrainingAgent:
    """Abstract base class for stateful training agent wrappers.

    A training agent is NOT a JAX PyTree — it is a stateful Python object
    that owns a model and a trainer. For JIT-compiled rollouts, call
    `rollout_actor()` to obtain a PyTree-compatible `BaseRolloutActor`.

    Subclasses must implement:
      - `rollout_actor()` — return the PyTree actor used for data collection.
      - `update(training_data)` — perform one training update.

    The `model`, `trainer`, `train()`, and `eval()` methods are provided by
    this base class assuming the standard `_model` / `_trainer` convention.
    """

    _model: PolicyValueModel
    _trainer: Trainer
    _rollout_actor: BaseRolloutActor

    def __init__(self, model: PolicyValueModel, trainer: Trainer, rollout_actor: BaseRolloutActor):
        self._model = model
        self._trainer = trainer
        self._rollout_actor = rollout_actor

    @property
    def model(self) -> PolicyValueModel:
        """The underlying policy/value network."""
        return self._model

    @property
    def trainer(self) -> Trainer:
        """The trainer used for parameter updates."""
        return self._trainer

    @property
    def rollout_actor(self) -> BaseRolloutActor:
        """The rollout actor used for action selection."""
        return self._rollout_actor

    @abstractmethod
    def update(
        self,
        training_data: dict[str, jnp.ndarray] | TrainingDataBuffer,
    ) -> dict:
        """Perform one training update on the given data.

        Parameters
        ----------
        training_data : TrainingDataBuffer or dict
            Training batch produced by the rollout collectors.

        Returns
        -------
        dict
            Dictionary of training metrics (losses, grad norms, …).
        """
        if isinstance(training_data, dict):
            td = dict(training_data)
            td["value_loss_mask"] = td.get(
                "value_loss_mask",
                jnp.ones_like(td["valids"], dtype=jnp.float32) if "valids" in td else None,
            )
            training_batch = TrainingDataBuffer(**td)
        else:
            training_batch = training_data

        return self._trainer.train_epochs(self._model, training_batch)

    def train(self) -> None:
        """Set model to training mode."""
        self._model.train()

    def eval(self) -> None:
        """Set model to evaluation mode."""
        self._model.eval()

    def sample_actions(
        self,
        state: pgx.State,
        key: jax.Array,
        hidden_state: jnp.ndarray | None = None,
    ) -> AgentOutput:
        """Convenience wrapper: sample actions via the rollout actor.

        Not intended for JIT-compiled use. For JIT rollouts pass
        `rollout_actor()` directly into the collector functions.
        """
        return self.rollout_actor.sample_actions(state, key, hidden_state)
