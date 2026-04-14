from __future__ import annotations

from collections.abc import Sequence
from typing import Any, override

import distrax  # type: ignore
import jax
import jax.numpy as jnp
from flax import nnx, struct
from flax.struct import PyTreeNode
from jax.sharding import Mesh

from jaxpot.models.base import PolicyValueModel
from jaxpot.rl.losses import (
    AuxiliaryLoss,
    compute_auxiliary_losses,
    value_loss_fn,
)
from jaxpot.rl.trainer import Trainer


def ppo_loss_fn(
    model: PolicyValueModel,
    batch: dict[str, jnp.ndarray],
    *,
    clip_value: float,
    clip_eps: float,
    value_coeff: float,
    policy_coeff: float,
    entropy_coeff: jnp.ndarray,
    kl_coef: float,
    normalize_advantages: bool,
    auxiliary_losses: tuple[AuxiliaryLoss, ...],
    auxiliary_loss_names: tuple[str, ...],
) -> tuple[jnp.ndarray, "PPOTrainMetrics"]:
    """Compute PPO loss for a minibatch of sequences.

    Uses the encode/core/decode pipeline with LSTM scan over sequence
    length.  Works for both recurrent (seq_len > 1) and non-recurrent
    (seq_len = 1) models.

    Parameters
    ----------
    model
        Model implementing encode/core/decode.
    batch
        Dict with keys:
        - obs: [B, S, *obs_shape]
        - value, returns: [B, S, 1]
        - adv, actions, log_prob: [B, S]
        - legal_action_mask: [B, S, A]
        - valids, done: [B, S]
        - hidden_state: [B, S, *state_shape] (opaque per-model recurrent state)
    """
    obs = batch["obs"]  # [B, S, *obs_shape]
    old_values = batch["value"]  # [B, S, 1]
    returns = batch["returns"]  # [B, S, 1]
    advantages = batch["adv"]  # [B, S]
    actions = batch["actions"]  # [B, S]
    old_log_prob = batch["log_prob"]  # [B, S]
    legal_action_mask = batch["legal_action_mask"]  # [B, S, A]
    valids = batch["valids"]  # [B, S]
    done = batch["done"]  # [B, S]
    # ``hidden_state`` is the per-chunk INITIAL recurrent state, shape
    # ``[B, *state_shape]`` — extract_init_hidden already collapses the
    # sequence dim (one entry per chunk start).
    hidden_state = batch["hidden_state"]  # [B, *state_shape]
    value_loss_mask = batch["value_loss_mask"]  # [B, S]

    B, S = actions.shape[:2]

    # 1. Encode all timesteps at once: [B, S, *obs] -> [B*S, D] -> [B, S, D]
    obs_flat = jax.tree.map(lambda x: x.reshape(B * S, *x.shape[2:]), obs)
    features_flat = model.encode(obs_flat)  # [B*S, D]
    D = features_flat.shape[-1]
    features = features_flat.reshape(B, S, D)

    # 2. Recurrent scan over sequence with done resets.
    # ``hidden_state`` is already the per-sample initial carry ``[B, *state_shape]``.

    def core_step(carry, inputs):
        hidden_state = carry  # [B, *state_shape]
        feat, d = inputs  # feat: [B, D], d: [B]

        # Reset hidden on done boundaries — broadcast the per-sample mask
        # across the opaque trailing state shape.
        d_mask = d.reshape(d.shape + (1,) * (hidden_state.ndim - 1))
        hidden_state = jnp.where(d_mask, 0.0, hidden_state)
        core_output, new_hidden_state = model.core(feat, hidden_state)
        return new_hidden_state, core_output  # core_output: [B, *core_output_shape]

    # done_shifted: done[t] means reset hidden at t+1
    done_shifted = jnp.concatenate([jnp.zeros((B, 1)), done[:, :-1]], axis=1)  # [B, S]

    # Transpose for scan: [B, S, D] -> [S, B, D], [B, S] -> [S, B]
    features_t = jnp.transpose(features, (1, 0, 2))  # [S, B, D]
    done_t = jnp.transpose(done_shifted, (1, 0))  # [S, B]

    _, core_outputs_t = jax.lax.scan(
        core_step, hidden_state, (features_t, done_t)
    )  # core_outputs_t: [S, B, hidden_size]

    # Transpose back: [S, B, H] -> [B*S, H]
    core_outputs = jnp.transpose(core_outputs_t, (1, 0, 2))  # [B, S, H]
    core_flat = core_outputs.reshape(B * S, -1)

    # 3. Decode all timesteps at once
    model_output = model.decode(core_flat)  # logits: [B*S, A], value: [B*S, 1]
    new_logits = model_output.policy_logits.reshape(B, S, -1)  # [B, S, A]
    new_values = model_output.value.reshape(B, S, 1)  # [B, S, 1]

    # 4. Compute losses per timestep, masked by valids
    v_loss = value_loss_fn(
        new_values.reshape(-1, 1),
        old_values.reshape(-1, 1),
        returns.reshape(-1, 1),
        clip_value=clip_value,
        mask=value_loss_mask,
    )

    # Policy distribution
    masked_logits = jnp.where(
        legal_action_mask.reshape(B * S, -1),
        new_logits.reshape(B * S, -1),
        -1e9,
    )
    pi = distrax.Categorical(logits=masked_logits)
    new_log_prob = pi.log_prob(actions.reshape(B * S))  # [B*S]
    entropy = pi.entropy()  # [B*S]

    valids_flat = valids.reshape(B * S)
    n_valid = jnp.maximum(valids_flat.sum(), 1.0)

    # Weighted entropy loss
    entropy_loss = -(entropy * valids_flat).sum() / n_valid

    # Probability ratio
    old_log_prob_flat = old_log_prob.reshape(B * S)
    log_ratio = new_log_prob - old_log_prob_flat
    log_ratio = jnp.clip(log_ratio, -20.0, 20.0)
    ratio = jnp.exp(log_ratio)

    # PPO clipped policy loss (weighted by valids)
    clip_ratio_high = 1.0 + clip_eps
    clip_ratio_low = 1.0 / clip_ratio_high

    adv_flat = advantages.reshape(B * S)
    if normalize_advantages:
        adv_mean = (adv_flat * valids_flat).sum() / n_valid
        adv_var = ((adv_flat - adv_mean) ** 2 * valids_flat).sum() / n_valid
        adv_flat = (adv_flat - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)

    unclipped = -adv_flat * ratio
    clipped = -adv_flat * jnp.clip(ratio, clip_ratio_low, clip_ratio_high)
    p_loss = (jnp.maximum(unclipped, clipped) * valids_flat).sum() / n_valid

    # KL divergence
    kl_sample = old_log_prob_flat - new_log_prob
    kl_loss = (kl_sample * valids_flat).sum() / n_valid

    # Clip fraction (diagnostic)
    clipfrac = ((jnp.abs(ratio - 1.0) > clip_eps) * valids_flat).sum() / n_valid

    # Apply coefficients
    value_loss_scaled = value_coeff * v_loss
    policy_loss_scaled = policy_coeff * p_loss
    entropy_loss_scaled = entropy_coeff * entropy_loss
    kl_loss_scaled = kl_coef * kl_loss

    # Auxiliary losses
    aux_targets = tuple(batch[al.target_field].reshape(B * S, -1) for al in auxiliary_losses)
    aux_loss_scaled, aux_loss_values = compute_auxiliary_losses(
        auxiliary_losses, model_output, aux_targets
    )

    total_loss = (
        value_loss_scaled
        + policy_loss_scaled
        + entropy_loss_scaled
        + kl_loss_scaled
        + aux_loss_scaled
    )

    metrics = PPOTrainMetrics(
        total_loss=total_loss,
        value_loss=value_loss_scaled,
        policy_loss=policy_loss_scaled,
        entropy_loss=entropy_loss_scaled,
        kl_loss=kl_loss_scaled,
        clipfrac=clipfrac,
        grad_norm=jnp.zeros((), dtype=jnp.float32),
        grad_norm_clipped=jnp.zeros((), dtype=jnp.float32),
        auxiliary_loss_values=aux_loss_values,
        auxiliary_loss_names=auxiliary_loss_names,
    )

    return total_loss, metrics


class PPOTrainMetrics(PyTreeNode):
    """Accumulating metrics for PPO training."""

    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy_loss: jnp.ndarray
    kl_loss: jnp.ndarray
    clipfrac: jnp.ndarray
    grad_norm: jnp.ndarray
    grad_norm_clipped: jnp.ndarray
    auxiliary_loss_values: jnp.ndarray
    auxiliary_loss_names: tuple[str, ...] = struct.field(pytree_node=False, default=())

    @classmethod
    def zero(cls, auxiliary_loss_names: tuple[str, ...] = ()) -> "PPOTrainMetrics":
        """Create zero-initialized metrics for accumulation."""
        z = jnp.zeros((), dtype=jnp.float32)
        return cls(
            total_loss=z,
            value_loss=z,
            policy_loss=z,
            entropy_loss=z,
            kl_loss=z,
            clipfrac=z,
            auxiliary_loss_values=jnp.zeros((len(auxiliary_loss_names),), dtype=jnp.float32),
            grad_norm=z,
            grad_norm_clipped=z,
            auxiliary_loss_names=auxiliary_loss_names,
        )

    def merge(self, other: "PPOTrainMetrics") -> "PPOTrainMetrics":
        """Merge another metrics instance into this one."""
        return PPOTrainMetrics(
            total_loss=self.total_loss + other.total_loss,
            value_loss=self.value_loss + other.value_loss,
            policy_loss=self.policy_loss + other.policy_loss,
            entropy_loss=self.entropy_loss + other.entropy_loss,
            kl_loss=self.kl_loss + other.kl_loss,
            clipfrac=self.clipfrac + other.clipfrac,
            auxiliary_loss_values=self.auxiliary_loss_values + other.auxiliary_loss_values,
            grad_norm=self.grad_norm + other.grad_norm,
            grad_norm_clipped=self.grad_norm_clipped + other.grad_norm_clipped,
            auxiliary_loss_names=self.auxiliary_loss_names,
        )

    def compute(self, count: jnp.ndarray) -> dict[str, float]:
        """Compute averaged metrics with single device sync."""

        n = jnp.maximum(count.astype(jnp.float32), 1.0)

        metric_values = jnp.stack(
            [
                self.total_loss / n,
                self.value_loss / n,
                self.policy_loss / n,
                self.entropy_loss / n,
                self.kl_loss / n,
                self.clipfrac / n,
                self.grad_norm / n,
                self.grad_norm_clipped / n,
            ]
        )

        if self.auxiliary_loss_names:
            aux_values = self.auxiliary_loss_values / n
            all_values = jnp.concatenate([metric_values, aux_values])
        else:
            all_values = metric_values

        host_values = jax.device_get(all_values)

        metrics = {
            "total_loss": float(host_values[0]),
            "value_loss": float(host_values[1]),
            "policy_loss": float(host_values[2]),
            "entropy_loss": float(host_values[3]),
            "kl_loss": float(host_values[4]),
            "clipfrac": float(host_values[5]),
            "grad_norm": float(host_values[6]),
            "grad_norm_clipped": float(host_values[7]),
        }
        if self.auxiliary_loss_names:
            for i, name in enumerate(self.auxiliary_loss_names):
                metrics[name] = float(host_values[8 + i])

        return metrics


class PPOTrainer(Trainer):
    """
    PPO trainer implementing the Proximal Policy Optimization algorithm.

    Uses the encode/core/decode pipeline with recurrent scan for all models.
    For non-recurrent models (seq_len=1), the scan has a single step and
    the identity core is optimized away by XLA.

    Parameters
    ----------
    optimizer : nnx.Optimizer
        Optimizer bound to model parameters.
    seq_len : int
        Sequence length for recurrent training.  Use 1 for non-recurrent.
    seed : int
        Random seed for internal RNG initialization.
    clip_eps : float
        Policy clipping epsilon.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Minibatch size (total timesteps, not sequences).
    mesh : Mesh | None
        Device mesh for multi-GPU training.
    max_grad_norm : float | None
        Max global gradient norm for clipped-norm diagnostics.
    clip_value : float
        Value clipping magnitude.
    value_coeff : float
        Value loss coefficient.
    policy_coeff : float
        Policy loss coefficient.
    entropy_coeff : float
        Final entropy coefficient (after decay).
    entropy_coeff_start : float, optional
        Initial entropy coefficient for decay schedule.
    entropy_decay_iterations : int
        Number of iterations over which to decay entropy coefficient.
    kl_coef : float
        KL divergence coefficient.
    auxiliary_losses : Sequence[AuxiliaryLoss], optional
        Auxiliary loss objects.
    normalize_advantages : bool
        Whether to normalize advantages.
    """

    def __init__(
        self,
        optimizer: nnx.Optimizer,
        *,
        seq_len: int = 1,
        start_iteration: int = 0,
        seed: int = 0,
        num_epochs: int = 1,
        batch_size: int = 16_384,
        mesh: Mesh | None = None,
        max_grad_norm: float | None = None,
        clip_value: float = 1.0,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        policy_coeff: float = 1.0,
        entropy_coeff: float = 0.01,
        entropy_coeff_start: float | None = None,
        entropy_decay_iterations: int = 100_000,
        kl_coef: float = 0.0,
        auxiliary_losses: Sequence[AuxiliaryLoss] = (),
        normalize_advantages: bool = True,
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
            seq_len=seq_len,
        )
        self.clip_value = clip_value
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.policy_coeff = policy_coeff
        self.entropy_coeff_end = entropy_coeff
        self.entropy_coeff_start = (
            entropy_coeff_start if entropy_coeff_start is not None else entropy_coeff
        )
        self.entropy_decay_iterations = entropy_decay_iterations
        self.kl_coef = kl_coef
        self.normalize_advantages = normalize_advantages

    @property
    def auxiliary_loss_names(self) -> tuple[str, ...]:
        """Get names of all auxiliary losses."""
        return tuple(aux_loss.name for aux_loss in self.auxiliary_losses)

    def get_entropy_coeff(self) -> jnp.ndarray:
        """Compute current entropy coefficient with cosine annealing."""
        progress = jnp.minimum(self.iterations.value / self.entropy_decay_iterations, 1.0)
        return self.entropy_coeff_end + 0.5 * (
            self.entropy_coeff_start - self.entropy_coeff_end
        ) * (1.0 + jnp.cos(jnp.pi * progress))

    def compute_metrics(self) -> dict[str, float]:
        metrics = self._metrics_accumulator.value.compute(self.training_steps.value)
        metrics["entropy_coeff"] = float(self.get_entropy_coeff())
        return metrics

    def _create_loss_fn(self):
        """Create loss function closure."""
        clip_value = self.clip_value
        clip_eps = self.clip_eps
        value_coeff = self.value_coeff
        policy_coeff = self.policy_coeff
        kl_coef = self.kl_coef
        normalize_advantages = self.normalize_advantages
        auxiliary_losses = self.auxiliary_losses
        auxiliary_loss_names = self.auxiliary_loss_names
        get_entropy_coeff = self.get_entropy_coeff

        def loss_fn(model, batch):
            return ppo_loss_fn(
                model,
                batch,
                clip_value=clip_value,
                clip_eps=clip_eps,
                value_coeff=value_coeff,
                policy_coeff=policy_coeff,
                entropy_coeff=get_entropy_coeff(),
                kl_coef=kl_coef,
                normalize_advantages=normalize_advantages,
                auxiliary_losses=auxiliary_losses,
                auxiliary_loss_names=auxiliary_loss_names,
            )

        return loss_fn

    def _create_pure_loss_fn(self, graphdef, non_params=None):
        """Create a pure loss function with entropy_coeff as explicit arg."""
        static_kwargs = dict(
            clip_value=self.clip_value,
            clip_eps=self.clip_eps,
            value_coeff=self.value_coeff,
            policy_coeff=self.policy_coeff,
            kl_coef=self.kl_coef,
            normalize_advantages=self.normalize_advantages,
            auxiliary_losses=self.auxiliary_losses,
            auxiliary_loss_names=self.auxiliary_loss_names,
        )

        def pure_loss(params, batch, dynamic_args):
            entropy_coeff = dynamic_args
            if non_params is not None:
                model = nnx.merge(graphdef, params, non_params)
            else:
                model = nnx.merge(graphdef, params)
            return ppo_loss_fn(model, batch, entropy_coeff=entropy_coeff, **static_kwargs)

        return pure_loss

    @override
    def _get_dynamic_loss_args(self) -> jnp.ndarray:
        """Return entropy coefficient as dynamic arg to avoid recompilation."""
        return self.get_entropy_coeff()

    @override
    def _create_zero_metrics(self) -> PPOTrainMetrics:
        return PPOTrainMetrics.zero(auxiliary_loss_names=self.auxiliary_loss_names)
