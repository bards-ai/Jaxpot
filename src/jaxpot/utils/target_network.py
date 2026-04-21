from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jaxpot.models.base import PolicyValueModel


class TargetNetwork:
    """Maintain a slowly-updated copy of model parameters."""

    def __init__(self, model: PolicyValueModel, tau: float = 0.005) -> None:
        self.tau = float(tau)
        self.target_model = self._clone_frozen(model)

    def _clone_frozen(self, model: PolicyValueModel) -> PolicyValueModel:
        graphdef, state = nnx.split(model)
        cloned = nnx.merge(graphdef, state)
        cloned.eval()
        return cloned

    def soft_update(self, online_model: PolicyValueModel) -> None:
        """
        Update target parameters using Polyak averaging.

        Parameters
        ----------
        online_model : PolicyValueModel
            Source model used for the update.
        """
        online_state = nnx.state(online_model)
        target_state = nnx.state(self.target_model)
        tau = self.tau

        def polyak_update(target, online):
            if isinstance(online, jax.Array) and jnp.issubdtype(online.dtype, jnp.inexact):
                return tau * online + (1.0 - tau) * target
            return target

        new_state = jax.tree.map(polyak_update, target_state, online_state)
        nnx.update(self.target_model, new_state)
