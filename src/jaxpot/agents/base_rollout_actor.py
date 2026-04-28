from abc import abstractmethod
from typing import NamedTuple

import jax
import jax.numpy as jnp
import pgx
from flax import struct


class AgentOutput(NamedTuple):
    """Output from an actor's ``sample_actions`` method."""

    actions: jnp.ndarray  # [E] sampled actions
    log_probs: jnp.ndarray  # [E] log probabilities of sampled actions
    values: jnp.ndarray  # [E] value estimates (can be zeros if not applicable)
    policy_logits: jnp.ndarray  # [E, A] masked policy logits
    hidden_state: jnp.ndarray | None = None  # [E, *hidden_shape] updated hidden state


class BaseRolloutActor(struct.PyTreeNode):
    """
    Abstract base class for rollout actors.

    Actors provide action selection logic during rollouts.

    All actors are JAX PyTree nodes to work efficiently with JIT compilation.
    """

    @abstractmethod
    def sample_actions(
        self,
        state: pgx.State,
        key: jax.Array,
        hidden_state: jnp.ndarray | None = None,
    ) -> AgentOutput:
        """
        Sample actions given observations and legal action mask.

        Parameters
        ----------
        state : pgx.State
            State of the environment.
        key : jax.Array
            PRNG key for sampling.

        Returns
        -------
        AgentOutput
            Named tuple containing actions, log_probs, values, and policy_logits.
        """
        raise NotImplementedError("Subclasses must implement sample_actions")

    def setup(self, *, step_fn, **kwargs):
        """Optional hook for attaching runtime rollout dependencies.

        Actors that require environment transitions during planning/search
        (e.g. MCTS) can override this and return a (typically replaced) actor
        with `step_fn` stored. Extra keyword arguments are accepted for
        compatibility with actor-specific setup parameters (e.g. MCTS-only
        arguments) and ignored by default.
        """
        return self

    def init_hidden_state(self, num_envs: int) -> jnp.ndarray:
        """Initialize the recurrent state for this actor.

        Returns a single batch-first array of shape ``[num_envs, *state_shape]``
        where ``state_shape`` is opaque to the rollout/training machinery.
        Non-recurrent actors return a dummy ``[num_envs, 1]`` placeholder.
        """
        return jnp.zeros((num_envs, 1))
