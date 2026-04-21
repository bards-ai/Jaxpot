from __future__ import annotations

from abc import abstractmethod
from typing import Any

import jax
import jax.numpy as jnp

from jaxpot.models.base.outputs import ModelOutput
from jaxpot.models.base.policy_value_model import PolicyValueModel


class ComposablePolicyValueModel(PolicyValueModel):
    """Policy/value model with an explicit encode-core-decode pipeline."""

    @property
    def features_dim(self) -> int:
        """Feature width consumed by the policy head."""
        policy_head = getattr(self, "policy_head", None)
        if policy_head is None:
            raise AttributeError(
                "ComposablePolicyValueModel subclasses must define policy_head or override features_dim."
            )
        return policy_head.in_features

    @abstractmethod
    def encode(self, obs: Any) -> Any:
        """Encode an observation into intermediate features."""
        raise NotImplementedError

    def core(self, features: Any, state: Any = None) -> tuple[Any, Any]:
        """Process encoded features through an optional recurrent core."""
        return features, state

    @abstractmethod
    def decode(self, core_output: Any) -> ModelOutput:
        """Decode intermediate features into policy/value outputs."""
        raise NotImplementedError

    def __call__(
        self,
        obs: Any,
        hidden_state: jnp.ndarray | None = None,
    ) -> ModelOutput:
        """Run the staged forward pipeline with optional recurrent state.

        ``hidden_state`` is treated as opaque to this base class — it is
        forwarded to ``core`` which is responsible for any internal packing
        (e.g. unpacking ``(h, c)`` from a single tensor for LSTM cores).
        """
        squeeze_output = (
            hasattr(obs, "ndim")
            and isinstance(self.obs_shape, tuple)
            and obs.ndim == len(self.obs_shape)
        )
        if squeeze_output:
            obs = obs[None, ...]
            if hidden_state is not None:
                hidden_state = hidden_state[None, ...]

        features = self.encode(obs)
        core_output, new_hidden_state = self.core(features, hidden_state)
        output = self.decode(core_output)

        if squeeze_output:
            output = jax.tree.map(
                lambda x: x.squeeze(0) if x is not None else None,
                output,
                is_leaf=lambda x: x is None,
            )
            if new_hidden_state is not None:
                new_hidden_state = new_hidden_state.squeeze(0)

        if new_hidden_state is not None:
            output = output.replace(hidden_state=new_hidden_state)
        return output
