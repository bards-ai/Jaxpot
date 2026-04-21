from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from flax import nnx

from jaxpot.models.base import ComposablePolicyValueModel, ModelOutput
from jaxpot.models.utils import orthogonal_init

ACTIVATION_MAP: dict[str, Callable] = {
    "relu": nnx.relu,
    "tanh": nnx.tanh,
    "elu": nnx.elu,
    "gelu": nnx.gelu,
    "silu": nnx.silu,
}


class MLPModel(ComposablePolicyValueModel):
    """Simple MLP model for flat observations.

    Parameters
    ----------
    rngs
        Random number generators for submodules.
    action_dim
        Number of policy logits to output.
    obs_shape
        Shape of the observation (will be flattened).
    hidden_dims
        Sizes of hidden layers.
    activation
        Activation function (callable or string: "relu", "tanh", "elu", "gelu", "silu").
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        action_dim: int | tuple[int, ...],
        obs_shape: int | tuple[int, ...],
        hidden_dims: int | list[int] = 128,
        activation: Callable[[jnp.ndarray], jnp.ndarray] | str = nnx.relu,
        **kwargs,
    ):
        super().__init__(obs_shape=obs_shape, action_dim=action_dim, **kwargs)
        self.hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else list(hidden_dims)
        if isinstance(activation, str):
            activation = ACTIVATION_MAP[activation]
        self.activation = activation

        self.mlp = nnx.Sequential(
            *[
                item
                for i, h in enumerate(self.hidden_dims)
                for item in (
                    nnx.Linear(
                        in_features=self.input_dim if i == 0 else self.hidden_dims[i - 1],
                        out_features=h,
                        rngs=rngs,
                    ),
                    activation,
                )
            ]
        )

        head_in_features = self.hidden_dims[-1] if self.hidden_dims else self.input_dim
        self.policy_head = nnx.Linear(
            in_features=head_in_features,
            out_features=self.action_dim,
            kernel_init=orthogonal_init(0.01),
            rngs=rngs,
        )
        self.value_head = nnx.Linear(
            in_features=head_in_features,
            out_features=1,
            kernel_init=orthogonal_init(1.0),
            rngs=rngs,
        )

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode observation into a feature vector.

        Parameters
        ----------
        obs
            Batched observation ``[B, *obs_shape]``.

        Returns
        -------
        jnp.ndarray
            Feature vector ``[B, D]``.
        """
        if obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        return self.mlp(obs)

    def decode(self, core_output: jnp.ndarray) -> ModelOutput:
        """Compute policy logits and value from feature vector."""
        return ModelOutput(
            value=self.value_head(core_output),
            policy_logits=self.policy_head(core_output),
        )
