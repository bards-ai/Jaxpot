from __future__ import annotations

import dataclasses
from abc import abstractmethod
from collections.abc import Sequence
from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxpot.models.base import PolicyValueModel
from jaxpot.rl.losses.auxiliary_losses import AuxiliaryLoss
from jaxpot.rollout.buffer import RolloutAuxTransform, TrainingDataBuffer


class Trainer(nnx.Module):
    """
    Abstract base trainer for RL algorithms.

    Provides a Flax NNX-style training interface with JIT-compiled methods.
    Subclass and implement `loss_fn` to create trainers for different algorithms.

    Supports multi-GPU data-parallel training when a device mesh is provided.
    Parameters
    ----------
    optimizer : nnx.Optimizer
        Optimizer bound to model parameters.
    seed : int
        Random seed for internal RNG initialization.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Minibatch size. With multi-GPU, this is the total batch size across
        all devices.
    max_grad_norm : float | None
        Max global gradient norm for clipped-norm diagnostics.
    mesh : Mesh | None
        Device mesh for multi-GPU training. If None, uses single device.
    seq_len : int
        Sequence length for recurrent training.  Use 1 for non-recurrent models.
    """

    def __init__(
        self,
        optimizer: nnx.Optimizer,
        *,
        start_iteration: int = 0,
        seed: int = 0,
        num_epochs: int = 1,
        batch_size: int = 16_384,
        max_grad_norm: float | None = None,
        mesh: Mesh | None = None,
        auxiliary_losses: Sequence[AuxiliaryLoss] = (),
        seq_len: int = 1,
    ):
        self.optimizer = optimizer
        self.rngs = nnx.Rngs(seed)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.mesh = mesh
        self.seq_len = seq_len
        self.iterations = nnx.Variable(jnp.zeros((), dtype=jnp.int32) + start_iteration)
        self.training_steps = nnx.Variable(jnp.zeros((), dtype=jnp.int32))
        self.auxiliary_losses = tuple(auxiliary_losses or ())

        self._batched_data_sharding: NamedSharding | None = None
        if mesh is not None:
            num_devices = len(mesh.devices)
            if batch_size % num_devices != 0:
                raise ValueError(
                    f"batch_size ({batch_size}) must be divisible by "
                    f"number of devices ({num_devices}) for multi-GPU training"
                )
            self._batched_data_sharding = NamedSharding(mesh, P(None, "data"))

        self._fori_epoch_fn: Callable | None = None
        self._fori_fresh_opt_struct: Any = None
        self._fori_nnx_opt_struct: Any = None

        self._metrics_accumulator = nnx.Variable(self._create_zero_metrics())

    @abstractmethod
    def _create_zero_metrics(self) -> Any:
        """Create zero-initialized metrics for accumulation."""
        ...

    def reset_metrics(self) -> None:
        """Reset accumulated metrics."""
        self._metrics_accumulator.value = self._create_zero_metrics()

    @abstractmethod
    def compute_metrics(self) -> dict[str, float]:
        """Compute averaged metrics."""
        ...

    def get_rollout_transforms(self) -> tuple[RolloutAuxTransform, ...]:
        """Return transforms that derive aux_target entries from RolloutBuffer fields.

        Override in subclasses that need to populate aux_targets from existing
        rollout fields (e.g. extracting MCTS policy from policy_logits).
        """
        return ()

    def _get_auxiliary_target_fields(self) -> tuple[str, ...]:
        """Return field names needed by auxiliary losses."""
        return tuple(aux_loss.target_field for aux_loss in self.auxiliary_losses)

    @abstractmethod
    def _create_loss_fn(self):
        """Create a loss function closure with current hyperparameters."""
        ...

    def _create_pure_loss_fn(
        self, graphdef: Any, non_params: Any = None
    ) -> Callable[..., tuple[jnp.ndarray, Any]]:
        """Create a pure loss function for fori_loop training."""
        loss_fn = self._create_loss_fn()

        def pure_loss(params, batch, dynamic_args):
            if non_params is not None:
                model = nnx.merge(graphdef, params, non_params)
            else:
                model = nnx.merge(graphdef, params)
            return loss_fn(model, batch)

        return pure_loss

    def _get_dynamic_loss_args(self) -> Any:
        """Return dynamic values passed to the pure loss function each iteration."""
        return None

    def _build_fori_epoch_fn(self, graphdef: Any, non_params: Any = None) -> Callable:
        """Build a pure JIT-compiled function for the full training loop.

        Uses jax.lax.scan to compile all minibatch steps into a single
        XLA program.
        """
        tx = self.optimizer.tx
        max_grad_norm = self.max_grad_norm
        aux_fields = self._get_auxiliary_target_fields()
        num_epochs = self.num_epochs

        pure_loss_fn = self._create_pure_loss_fn(graphdef, non_params)
        grad_fn = jax.value_and_grad(pure_loss_fn, has_aux=True)

        @jax.jit
        def run_epochs(params, opt_state, batched_data, dynamic_args, zero_metrics):
            steps = jnp.zeros((), dtype=jnp.int32)

            def scan_step(carry, batch_slice):
                params, opt_state, metrics_acc, steps = carry

                batch = {
                    "obs": batch_slice["obs"],
                    "value": batch_slice["value"],
                    "returns": batch_slice["returns"],
                    "adv": batch_slice["adv"],
                    "actions": batch_slice["actions"],
                    "log_prob": batch_slice["log_prob"],
                    "legal_action_mask": batch_slice["legal_action_mask"],
                    "valids": batch_slice["valids"],
                    "done": batch_slice["done"],
                    "hidden_state": batch_slice["hidden_state"],
                    "value_loss_mask": batch_slice["value_loss_mask"],
                }
                for field_name in aux_fields:
                    batch[field_name] = batch_slice[field_name]

                (_, metrics), grads = grad_fn(params, batch, dynamic_args)
                grad_norm = optax.global_norm(grads)
                grad_norm_clipped = grad_norm
                update_grads = grads
                if max_grad_norm is not None:
                    clip_tx = optax.clip_by_global_norm(max_grad_norm)
                    clip_state = clip_tx.init(grads)
                    clipped_grads, _ = clip_tx.update(grads, clip_state)
                    grad_norm_clipped = optax.global_norm(clipped_grads)
                    update_grads = clipped_grads

                metrics = dataclasses.replace(
                    metrics, grad_norm=grad_norm, grad_norm_clipped=grad_norm_clipped
                )

                updates, new_opt_state = tx.update(update_grads, opt_state, params)
                new_params = optax.apply_updates(params, updates)
                new_metrics = metrics_acc.merge(metrics)
                return (new_params, new_opt_state, new_metrics, steps + 1), None

            scan_input = {
                "obs": batched_data.obs,
                "value": batched_data.value,
                "returns": batched_data.returns,
                "adv": batched_data.adv,
                "actions": batched_data.actions,
                "log_prob": batched_data.log_prob,
                "legal_action_mask": batched_data.legal_action_mask,
                "valids": batched_data.valids,
                "done": batched_data.done,
                "hidden_state": batched_data.hidden_state,
                "value_loss_mask": batched_data.value_loss_mask,
            }
            for field_name in aux_fields:
                scan_input[field_name] = batched_data.aux_targets[field_name]

            carry = (params, opt_state, zero_metrics, steps)
            for _ in range(num_epochs):
                carry, _ = jax.lax.scan(scan_step, carry, scan_input)
            return carry

        return run_epochs

    def _ensure_fori_epoch_fn(self, model: PolicyValueModel) -> tuple[Any, Any, Any]:
        """Ensure the fori_loop epoch function is built and cached.

        Returns ``(graphdef, params, non_params)`` from splitting the model.
        """
        graphdef, params, non_params = nnx.split(model, nnx.Param, ...)

        if self._fori_epoch_fn is None:
            self._fori_epoch_fn = self._build_fori_epoch_fn(graphdef, non_params)
            fresh_opt_state = self.optimizer.tx.init(params)
            self._fori_fresh_opt_struct = jax.tree_util.tree_structure(fresh_opt_state)
            self._fori_nnx_opt_struct = jax.tree_util.tree_structure(self.optimizer.opt_state)

        return graphdef, params, non_params

    def _run_all_epochs(
        self,
        model: PolicyValueModel,
        batched_data: TrainingDataBuffer,
    ) -> None:
        """Run training loop over all epochs using a fori_loop."""
        graphdef, params, non_params = self._ensure_fori_epoch_fn(model)

        opt_state = jax.tree_util.tree_unflatten(
            self._fori_fresh_opt_struct,
            jax.tree_util.tree_leaves(self.optimizer.opt_state),
        )

        dynamic_args = self._get_dynamic_loss_args()
        zero_metrics = self._create_zero_metrics()

        params, opt_state, final_metrics, final_steps = self._fori_epoch_fn(
            params,
            opt_state,
            batched_data,
            dynamic_args,
            zero_metrics,
        )

        nnx.update(model, nnx.State.merge(params, non_params))
        self.optimizer.opt_state = jax.tree_util.tree_unflatten(
            self._fori_nnx_opt_struct, jax.tree_util.tree_leaves(opt_state)
        )
        self._metrics_accumulator.value = final_metrics
        self.training_steps.value = final_steps

    def train_epochs(
        self,
        model: PolicyValueModel,
        training_data: TrainingDataBuffer,
    ) -> dict[str, float]:
        """Run multiple epochs on sequence-based training data.

        The batch_size refers to the number of *timesteps* per minibatch.
        For recurrent training, sequences are batched such that
        ``seq_batch_size = batch_size // seq_len``.
        """
        self.reset_metrics()
        self.training_steps.value = jnp.zeros((), dtype=jnp.int32)

        seq_batch_size = self.batch_size // self.seq_len

        if seq_batch_size < 1:
            raise ValueError(f"batch_size ({self.batch_size}) must be >= seq_len ({self.seq_len})")

        rng_key = self.rngs()

        batched_data = _prepare_training_data(training_data, rng_key, seq_batch_size)

        if self._batched_data_sharding is not None:
            batched_data = jax.device_put(batched_data, self._batched_data_sharding)

        self._run_all_epochs(model, batched_data)
        self.iterations.value = self.iterations.value + 1
        return self.compute_metrics()


def _prepare_training_data(
    training_data: TrainingDataBuffer,
    rng_key: jax.Array,
    seq_batch_size: int,
) -> TrainingDataBuffer:
    """Filter to valid sequences, shuffle, and reshape for minibatching.

    Computes the number of valid sequences (those with at least one valid
    timestep), then shuffles and reshapes into
    ``[num_minibatches, seq_batch_size, seq_len, ...]``.
    """
    num_valid_sequences = int(jax.device_get(jnp.sum(training_data.valids.sum(axis=-1) > 0)))
    num_minibatches = max(1, num_valid_sequences // seq_batch_size)
    usable_sequences = num_minibatches * seq_batch_size
    return _shuffle_and_reshape(
        training_data, rng_key, usable_sequences, num_minibatches, seq_batch_size
    )


@partial(
    jax.jit,
    static_argnames=("usable_sequences", "num_minibatches", "seq_batch_size"),
)
def _shuffle_and_reshape(
    training_data: TrainingDataBuffer,
    rng_key: jax.Array,
    usable_sequences: int,
    num_minibatches: int,
    seq_batch_size: int,
) -> TrainingDataBuffer:
    """Shuffle and reshape training data for minibatching."""
    num_sequences = training_data.adv.shape[0]

    valid_per_seq = training_data.valids.sum(axis=-1)  # [N]
    random_keys = jax.random.uniform(rng_key, (num_sequences,))
    sort_keys = jnp.where(valid_per_seq > 0, random_keys, 2.0)
    sorted_indices = jnp.argsort(sort_keys)[:usable_sequences]

    def gather_sorted(x):
        return x[sorted_indices]

    training_data = jax.tree.map(gather_sorted, training_data)

    def reshape_for_batch(x):
        return x.reshape((num_minibatches, seq_batch_size) + x.shape[1:])

    return jax.tree.map(reshape_for_batch, training_data)
