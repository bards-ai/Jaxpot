"""
AlphaZero trainer.

Mirrors the structure of PPOTrainer / Trainer while keeping the
AlphaZero-specific batch loss local to this module.

The batch layout expected from TrainingDataBuffer is:
  - obs, returns, policy_logits (= log MCTS policy), legal_action_mask, valids.
  - adv, log_prob, value are present in the buffer but not used by this trainer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, override

import jax
import jax.numpy as jnp
from flax import nnx
from flax.struct import PyTreeNode
from jax.sharding import Mesh

from jaxpot.models.base import PolicyValueModel
from jaxpot.rl.losses import (
    AuxiliaryLoss,
    alphazero_policy_loss_fn,
    alphazero_value_loss_fn,
)
from jaxpot.rl.trainer import Trainer
from jaxpot.rollout.buffer import RolloutAuxTransform, RolloutBuffer


class MCTSPolicyTransform:
    """Extracts raw MCTS visit-count probabilities from RolloutBuffer.policy_logits into aux_targets["mcts_policy"]."""

    target_field = "mcts_policy"

    def __call__(self, rb: RolloutBuffer, seats: tuple[int, ...]) -> jax.Array:
        return jnp.concatenate(
            [rb.policy_logits[:, :, seat].reshape(-1, rb.policy_logits.shape[-1]) for seat in seats],
            axis=0,
        )


class AlphaZeroTrainMetrics(PyTreeNode):
    """Accumulating metrics for AlphaZero training."""

    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    grad_norm: jnp.ndarray
    grad_norm_clipped: jnp.ndarray

    @classmethod
    def zero(cls) -> "AlphaZeroTrainMetrics":
        z = jnp.zeros((), dtype=jnp.float32)
        return cls(
            total_loss=z,
            value_loss=z,
            policy_loss=z,
            grad_norm=z,
            grad_norm_clipped=z,
        )

    def merge(self, other: "AlphaZeroTrainMetrics") -> "AlphaZeroTrainMetrics":
        return AlphaZeroTrainMetrics(
            total_loss=self.total_loss + other.total_loss,
            value_loss=self.value_loss + other.value_loss,
            policy_loss=self.policy_loss + other.policy_loss,
            grad_norm=self.grad_norm + other.grad_norm,
            grad_norm_clipped=self.grad_norm_clipped + other.grad_norm_clipped,
        )

    def compute(self, count: jnp.ndarray) -> dict[str, float]:
        n = jnp.maximum(count.astype(jnp.float32), 1.0)
        values = jnp.stack(
            [
                self.total_loss / n,
                self.value_loss / n,
                self.policy_loss / n,
                self.grad_norm / n,
                self.grad_norm_clipped / n,
            ]
        )
        host = jax.device_get(values)
        return {
            "total_loss": float(host[0]),
            "value_loss": float(host[1]),
            "policy_loss": float(host[2]),
            "grad_norm": float(host[3]),
            "grad_norm_clipped": float(host[4]),
        }


def _alphazero_loss_fn(
    model: PolicyValueModel,
    batch: dict[str, jnp.ndarray],
    *,
    value_coeff: float,
    policy_coeff: float,
) -> tuple[jnp.ndarray, AlphaZeroTrainMetrics]:
    """
    Compute the AlphaZero training loss for a minibatch.

    Parameters
    ----------
    model : PolicyValueModel
        Model used to compute policy logits and value predictions.
    batch : dict[str, jnp.ndarray]
        Minibatch containing AlphaZero supervision targets.
    value_coeff : float
        Weight applied to the value loss.
    policy_coeff : float
        Weight applied to the policy loss.

    Returns
    -------
    tuple[jnp.ndarray, AlphaZeroTrainMetrics]
        Total loss and per-component metrics for the minibatch.
    """
    obs = batch["obs"]  # [B, S, *obs_shape]
    returns = batch["returns"]  # [B, S, 1]
    mcts_policy = batch["mcts_policy"]  # [B, S, A] or [B, A]
    legal_action_mask = batch["legal_action_mask"]  # [B, S, A]
    value_loss_mask = batch["value_loss_mask"]  # [B, S]
    valids = batch["valids"]  # [B, S]

    B, S = valids.shape[:2]
    obs_flat = jax.tree.map(lambda x: x.reshape(B * S, *x.shape[2:]), obs)
    legal_action_mask_flat = legal_action_mask.reshape(B * S, -1)
    mcts_policy_flat = mcts_policy.reshape(B * S, -1)
    valids_flat = valids.reshape(B * S)
    value_loss_mask_flat = value_loss_mask.reshape(B * S)
    returns_flat = returns.reshape(B * S, -1)

    model_output = model(obs_flat)
    pred_values = model_output.value
    pred_policy_logits = model_output.policy_logits

    value_mask = value_loss_mask_flat * valids_flat
    value_loss = alphazero_value_loss_fn(
        pred_values=pred_values,
        target_values=returns_flat.reshape(pred_values.shape),
        mask=value_mask,
    )
    policy_loss = alphazero_policy_loss_fn(
        policy_logits=pred_policy_logits,
        target_policy=mcts_policy_flat,
        legal_action_mask=legal_action_mask_flat,
        mask=valids_flat,
    )

    value_loss_scaled = value_coeff * value_loss
    policy_loss_scaled = policy_coeff * policy_loss
    total_loss = value_loss_scaled + policy_loss_scaled

    metrics = AlphaZeroTrainMetrics(
        total_loss=total_loss,
        value_loss=value_loss_scaled,
        policy_loss=policy_loss_scaled,
        grad_norm=jnp.zeros((), dtype=jnp.float32),
        grad_norm_clipped=jnp.zeros((), dtype=jnp.float32),
    )
    return total_loss, metrics
class AlphaZeroTrainer(Trainer):
    """
    AlphaZero trainer: value MSE + policy cross-entropy against MCTS targets.

    Expects TrainingDataBuffer where `policy_logits` contains visit-count
    probabilities as produced by MCTSAgent.

    Parameters
    ----------
    optimizer : nnx.Optimizer
        Optimizer bound to model parameters.
    seed : int
        Random seed.
    num_epochs : int
        Training epochs per update.
    batch_size : int
        Minibatch size.
    mesh : Mesh | None
        Device mesh for multi-GPU training.
    max_grad_norm : float | None
        Gradient clipping norm.
    value_coeff : float
        Weight for value MSE loss.
    policy_coeff : float
        Weight for policy cross-entropy loss.
    """

    def __init__(
        self,
        optimizer: nnx.Optimizer,
        *,
        start_iteration: int = 0,
        seed: int = 0,
        num_epochs: int = 1,
        batch_size: int = 16_384,
        mesh: Mesh | None = None,
        max_grad_norm: float | None = None,
        value_coeff: float = 1.0,
        policy_coeff: float = 1.0,
        auxiliary_losses: Sequence[AuxiliaryLoss] = (),
    ):
        super().__init__(
            optimizer,
            start_iteration=start_iteration,
            seed=seed,
            num_epochs=num_epochs,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            mesh=mesh,
            auxiliary_losses=auxiliary_losses,
        )
        self.value_coeff = value_coeff
        self.policy_coeff = policy_coeff

    @override
    def _init_metrics(self) -> None:
        self._metrics_accumulator = nnx.Variable(AlphaZeroTrainMetrics.zero())

    @override
    def _accumulate_metrics(self, metrics) -> None:
        self._metrics_accumulator.value = self._metrics_accumulator.value.merge(metrics)

    @override
    def reset_metrics(self) -> None:
        self._metrics_accumulator.value = AlphaZeroTrainMetrics.zero()

    @override
    def compute_metrics(self) -> dict[str, float]:
        return self._metrics_accumulator.value.compute(self.training_steps.value)

    @override
    def _create_loss_fn(
        self,
    ) -> Callable[[PolicyValueModel, dict[str, jnp.ndarray]], tuple[jnp.ndarray, PyTreeNode]]:
        value_coeff = self.value_coeff
        policy_coeff = self.policy_coeff

        def loss_fn(
            model: PolicyValueModel, batch: dict[str, jnp.ndarray]
        ) -> tuple[jnp.ndarray, AlphaZeroTrainMetrics]:
            return _alphazero_loss_fn(
                model,
                batch,
                value_coeff=value_coeff,
                policy_coeff=policy_coeff,
            )

        return loss_fn

    @override
    def get_rollout_transforms(self) -> tuple[RolloutAuxTransform, ...]:
        return (MCTSPolicyTransform(),)

    @override
    def _get_auxiliary_target_fields(self) -> tuple[str, ...]:
        """Include mcts_policy so Trainer._extract_batch_at_index passes it to the loss."""
        return ("mcts_policy",)

    @override
    def _create_pure_loss_fn(
        self, graphdef: Any, non_params: Any = None
    ) -> Callable[..., tuple[jnp.ndarray, AlphaZeroTrainMetrics]]:
        static_kwargs = dict(
            value_coeff=self.value_coeff,
            policy_coeff=self.policy_coeff,
        )

        def pure_loss(params, batch, dynamic_args):
            model = (
                nnx.merge(graphdef, params, non_params)
                if non_params is not None
                else nnx.merge(graphdef, params)
            )
            return _alphazero_loss_fn(model, batch, **static_kwargs)

        return pure_loss

    @override
    def _get_dynamic_loss_args(self) -> Any:
        return None

    @override
    def _create_zero_metrics(self) -> AlphaZeroTrainMetrics:
        return AlphaZeroTrainMetrics.zero()
