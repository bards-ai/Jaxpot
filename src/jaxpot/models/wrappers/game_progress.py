from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jaxpot.models.base import ComposablePolicyValueModel, GameProgressModelOutput
from jaxpot.models.utils import orthogonal_init


class GameProgressWrapper(ComposablePolicyValueModel):
    """Wrapper that adds a game-progress auxiliary head to a composable model.

    Delegates ``encode``/``core`` to the inner model and overrides ``decode``
    to append a sigmoid game-progress prediction.

    Parameters
    ----------
    inner
        Any ComposablePolicyValueModel instance.
    rngs
        Random number generators for the auxiliary head.
    """

    def __init__(self, inner: ComposablePolicyValueModel, *, rngs: nnx.Rngs):
        super().__init__(obs_shape=inner.obs_shape, action_dim=inner.action_dim)
        self.inner = inner
        self.game_progress_head = nnx.Linear(
            in_features=inner.features_dim,
            out_features=1,
            kernel_init=orthogonal_init(1.0),
            rngs=rngs,
        )

    @property
    def features_dim(self) -> int:
        return self.inner.features_dim

    @property
    def hidden_shape(self) -> tuple[int, ...]:
        return self.inner.hidden_shape

    @property
    def is_recurrent(self) -> bool:
        return self.inner.is_recurrent

    def init_state(self, batch_size: int):
        return self.inner.init_state(batch_size)

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        return self.inner.encode(obs)

    def core(self, features: jnp.ndarray, state=None):
        return self.inner.core(features, state)

    def decode(self, core_output: jnp.ndarray) -> GameProgressModelOutput:
        result = self.inner.decode(core_output)
        game_progress = jax.nn.sigmoid(self.game_progress_head(core_output))
        return GameProgressModelOutput(
            value=result.value,
            policy_logits=result.policy_logits,
            game_progress=game_progress,
        )



def make_game_progress_model(
    *,
    inner,
    rngs: nnx.Rngs,
    obs_shape,
    action_dim,
    **kwargs,
) -> GameProgressWrapper:
    """Factory for Hydra instantiation.

    Usage in config::

        _target_: jaxpot.models.wrappers.game_progress.make_game_progress_model
        inner:
          _target_: jaxpot.models.architectures.conv.ConvModel
          _partial_: true
          conv_channels: [32, 64]
          hidden_dims: 128
    """
    inner_model = inner(rngs=rngs, obs_shape=obs_shape, action_dim=action_dim)
    return GameProgressWrapper(inner_model, rngs=rngs)
