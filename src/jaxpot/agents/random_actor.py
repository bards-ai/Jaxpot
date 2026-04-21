import distrax
import jax
import jax.numpy as jnp
import pgx
from flax import struct

from jaxpot.agents.base_rollout_actor import AgentOutput, BaseRolloutActor


@struct.dataclass
class RandomActor(BaseRolloutActor):
    """
    Actor that selects actions uniformly at random from legal actions.

    Useful as a baseline opponent or for exploration.

    This is a JAX PyTree node with no trainable parameters,
    so it can be used efficiently in JIT-compiled functions.
    """

    def sample_actions(
        self,
        state: pgx.State,
        key: jax.Array,
        hidden_state: jnp.ndarray | None = None,
    ) -> AgentOutput:
        """
        Sample actions uniformly from legal actions.

        Parameters
        ----------
        obs : jnp.ndarray
            Observations, shape [E, obs_dim]. (Ignored by random agent)
        legal_action_mask : jnp.ndarray
            Boolean mask of legal actions, shape [E, num_actions].
        key : jax.Array
            PRNG key for sampling.

        Returns
        -------
        AgentOutput
            Random actions with uniform log_probs, zero values.
        """
        num_envs = state.observation.shape[0]
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        pi = distrax.Categorical(logits=logits)
        actions = pi.sample(seed=key)
        log_probs = pi.log_prob(actions)

        return AgentOutput(
            actions=actions,
            log_probs=log_probs,
            values=jnp.zeros(num_envs, dtype=jnp.float32),
            policy_logits=logits,
            hidden_state=hidden_state,
        )
