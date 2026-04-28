from typing import override

import distrax  # type: ignore
import jax
import jax.numpy as jnp
import pgx
from flax import struct

from jaxpot.agents.base_rollout_actor import AgentOutput, BaseRolloutActor
from jaxpot.models.base import PolicyValueModel


@struct.dataclass
class PolicyActor(BaseRolloutActor):
    """
    Actor that uses a neural network policy for action selection.

    Wraps a Flax nnx.Module that outputs (value_logits, policy_logits, ...)
    and provides the standard actor interface for rollouts.

    The model is stored as a PyTree node, so this actor can be used
    efficiently in JIT-compiled functions without recompilation.
    """

    model: PolicyValueModel

    @override
    def sample_actions(
        self,
        state: pgx.State,
        key: jax.Array,
        hidden_state: jnp.ndarray | None = None,
    ) -> AgentOutput:
        """
        Sample actions from the policy network.

        Parameters
        ----------
        obs : jnp.ndarray
            Observations, shape [E, obs_dim].
        legal_action_mask : jnp.ndarray
            Boolean mask of legal actions, shape [E, num_actions].
        key : jax.Array
            PRNG key for sampling.

        Returns
        -------
        AgentOutput
            Actions, log_probs, values, and masked policy logits.
        """
        model_output = self.model(state.observation, hidden_state)
        masked = jnp.where(state.legal_action_mask, model_output.policy_logits, -1e9)
        pi = distrax.Categorical(logits=masked)
        actions = pi.sample(seed=key)
        log_probs = pi.log_prob(actions)

        # Non-recurrent models return ``hidden_state=None``; pass the input
        # through unchanged so downstream controllers can stay shape-agnostic.
        next_hidden_state = (
            model_output.hidden_state if model_output.hidden_state is not None else hidden_state
        )

        return AgentOutput(
            actions=actions,
            log_probs=log_probs,
            values=model_output.value.squeeze(),
            policy_logits=masked,
            hidden_state=next_hidden_state,
        )

    @override
    def init_hidden_state(self, num_envs: int) -> jnp.ndarray:
        """Delegate to the wrapped model's :meth:`init_state`.

        Returns a single batch-first tensor of shape ``[num_envs, *hidden_shape]``.
        """
        return self.model.init_state(num_envs)
