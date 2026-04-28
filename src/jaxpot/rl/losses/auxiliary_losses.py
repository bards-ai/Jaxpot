"""Auxiliary loss abstractions for PPO training.

This module provides a base class for auxiliary losses that can be plugged into
the PPO training loop without modifying the core PPO implementation.

AuxiliaryLoss is generic over the model output type, so game-specific losses
can type-hint their concrete ModelOutput subclass while the training loop
stays model-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

import jax.numpy as jnp

from jaxpot.models.base import ModelOutput
from jaxpot.rollout.aux_target_hooks import AuxTargetHook

TModelOutput = TypeVar("TModelOutput", bound=ModelOutput)


class AuxiliaryLoss(ABC, Generic[TModelOutput]):
    """Abstract base class for auxiliary losses in PPO training.

    Generic over the model output type so concrete losses can access
    game-specific fields (e.g. game progress).

    All subclasses must define:
    - name: str - unique identifier for the loss
    - target_field: str - field name in TrainingDataBatch for targets
    - coef: float - loss coefficient (instance variable)
    - __call__: compute the loss
    - make_target_hook: return the rollout hook that collects targets
    """

    name: str
    target_field: str

    @abstractmethod
    def __call__(
        self,
        model_outputs: TModelOutput,
        target: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute the auxiliary loss.

        Parameters
        ----------
        model_outputs : TModelOutput
            Full model outputs (concrete type depends on the game).
        target : jnp.ndarray
            Target values for this auxiliary loss.

        Returns
        -------
        jnp.ndarray
            Scalar loss value (already scaled by coefficient).
        """
        ...

    @abstractmethod
    def make_target_hook(self) -> AuxTargetHook:
        """Return rollout target hook for this loss."""
        raise NotImplementedError


def compute_auxiliary_losses(
    auxiliary_losses: Sequence[AuxiliaryLoss],
    model_outputs: ModelOutput,
    targets: Sequence[jnp.ndarray | None],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute all auxiliary losses and return total + individual values.

    Parameters
    ----------
    auxiliary_losses : Sequence[AuxiliaryLoss]
        List of auxiliary loss objects.
    model_outputs : ModelOutput
        Full model outputs (base type; concrete losses will cast as needed).
    targets : Sequence[jnp.ndarray | None]
        Target arrays aligned with auxiliary_losses. Use None to skip a loss.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (total_aux_loss, loss_values_by_index).
    """
    if not auxiliary_losses:
        return jnp.array(0.0, dtype=jnp.float32), jnp.zeros((0,), dtype=jnp.float32)

    total = jnp.array(0.0, dtype=jnp.float32)
    loss_values = []

    for aux_loss, target in zip(auxiliary_losses, targets, strict=True):
        if target is None:
            loss_val = jnp.array(0.0, dtype=jnp.float32)
        else:
            loss_val = aux_loss(model_outputs, target)
        total = total + loss_val
        loss_values.append(loss_val)

    return total, jnp.stack(loss_values)
