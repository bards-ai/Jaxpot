from typing import override

import jax
import jax.numpy as jnp
import pgx
from flax import struct

from jaxpot.agents.base_rollout_actor import BaseRolloutActor
from jaxpot.rollout.buffer import RolloutBuffer


class ControllerState(struct.PyTreeNode):
    """State for agent-based controllers."""

    hidden_state: jnp.ndarray


def _broadcast_per_env(mask: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Reshape a per-env mask ``[E]`` so it broadcasts over ``target``.

    The hidden-state convention is batch-first with an opaque trailing shape:
    ``[E, *state_shape]`` (per actor) or ``[2, E, *state_shape]`` (controller).
    Callers pass a target with batch on a known leading axis; this helper
    reshapes the mask to ``[E, 1, 1, ...]`` so :func:`jnp.where` broadcasts
    cleanly regardless of how many trailing axes the state carries.
    """
    return mask.reshape(mask.shape + (1,) * (target.ndim - mask.ndim))


class RolloutController(struct.PyTreeNode):
    """
    Base class for agent-based rollout controllers.
    """

    def init_state(self, *, num_envs: int, key: jax.Array):
        """Return initial controller state and (optionally updated) rng key."""
        raise NotImplementedError

    def select_actions(
        self,
        *,
        controller_state,
        state: pgx.State,
        key: jax.Array,
    ):
        """
        Select actions using agents.

        Returns
        -------
        tuple
            (actions, log_probs, values, policy_logits, valids, new_state)
        """
        raise NotImplementedError

    def post_step(
        self,
        *,
        controller_state,
        next_state,
        done_next: jnp.ndarray,
        current_player: jnp.ndarray,
        key: jax.Array,
    ):
        """Update controller state after the environment step."""
        raise NotImplementedError

    def finalize(self, *, rollout_buffer: RolloutBuffer, controller_state):
        """Optional final processing; returns `(rollout_buffer, outputs)`."""
        raise NotImplementedError


class SelfPlayController(RolloutController):
    """Self-play controller using encode/core/decode pipeline.

    Both players use the same agent. Per-player hidden states are tracked
    separately and reset to zero on game termination. Works for both
    recurrent and non-recurrent models — the controller treats hidden state
    as an opaque batch-first tensor of shape ``[E, *state_shape]``.

    The agent must be a JAX PyTree node (e.g., :class:`PolicyActor`,
    :class:`RandomActor`).
    """

    agent: BaseRolloutActor

    @override
    def init_state(self, *, num_envs: int, key: jax.Array):
        hidden_state = self.agent.init_hidden_state(num_envs)  # [E, *state_shape]
        hidden_state_both = jnp.stack(
            [hidden_state, hidden_state], axis=0
        )  # [2, E, *state_shape]
        return ControllerState(hidden_state=hidden_state_both), key

    @override
    def select_actions(
        self,
        *,
        controller_state: ControllerState,
        state: pgx.State,
        key: jax.Array,
    ):
        hidden_state = controller_state.hidden_state  # [2, E, *state_shape]
        is_p0 = _broadcast_per_env(state.current_player == 0, hidden_state[0])
        hidden_state_curr = jnp.where(is_p0, hidden_state[0], hidden_state[1])  # [E, *state]

        agent_output = self.agent.sample_actions(state, key, hidden_state_curr)
        new_hidden_state_p0 = jnp.where(is_p0, agent_output.hidden_state, hidden_state[0])
        new_hidden_state_p1 = jnp.where(is_p0, hidden_state[1], agent_output.hidden_state)

        new_state = ControllerState(
            hidden_state=jnp.stack([new_hidden_state_p0, new_hidden_state_p1], axis=0)
        )

        valids_flag = jnp.ones(state.current_player.shape, dtype=bool)
        return (
            agent_output.actions,
            agent_output.log_probs,
            agent_output.values,
            agent_output.policy_logits,
            valids_flag,
            new_state,
        )

    @override
    def post_step(
        self,
        *,
        controller_state: ControllerState,
        next_state,
        done_next: jnp.ndarray,
        current_player: jnp.ndarray,
        key: jax.Array,
    ):
        # done_next: [E] -> broadcast over [2, E, *state_shape]
        done_mask = _broadcast_per_env(done_next, controller_state.hidden_state[0])[None]
        return ControllerState(
            hidden_state=jnp.where(done_mask, 0.0, controller_state.hidden_state),
        )

    @override
    def finalize(self, *, rollout_buffer: RolloutBuffer, controller_state):
        return rollout_buffer, None


class OpponentControllerState(struct.PyTreeNode):
    """State for the agent-vs-opponent controller.

    Tracks two heterogeneously-shaped hidden states: the main agent's state
    (in ``hidden_state``, exposed to the rollout buffer) and the opponent's
    state (in ``opp_hidden_state``, used only inside the controller).

    ``hidden_state`` has shape ``[2, E, *main_state_shape]`` where index 0 is
    the main agent's actual state and index 1 is a zero placeholder (opponent
    steps are masked out by ``valids`` during training anyway).
    """

    hidden_state: jnp.ndarray      # [2, E, *main_state_shape]
    opp_hidden_state: jnp.ndarray  # [E, *opp_state_shape]


@struct.dataclass
class AgentOpponentRolloutController(RolloutController):
    """
    Controller where the main agent plays against a fixed opponent agent.

    The main agent's seat is specified, and only its actions are marked valid.
    Both agents must be JAX PyTree nodes.

    The two agents may have **different** hidden-state shapes (e.g. a
    recurrent main agent vs. a non-recurrent random opponent). The main
    agent's state is stored in :pyattr:`OpponentControllerState.hidden_state`
    (which the rollout buffer records), while the opponent's is tracked
    separately in :pyattr:`OpponentControllerState.opp_hidden_state`.
    """

    main_agent: BaseRolloutActor
    opponent_agent: BaseRolloutActor
    main_seat: int = struct.field(pytree_node=False)

    @override
    def init_state(self, *, num_envs: int, key: jax.Array):
        hidden_main = self.main_agent.init_hidden_state(num_envs)  # [E, *main_state]
        hidden_opp = self.opponent_agent.init_hidden_state(num_envs)  # [E, *opp_state]
        # The buffer needs [2, E, *main_state]; slot 1 is a zero placeholder.
        hidden_both = jnp.stack(
            [hidden_main, jnp.zeros_like(hidden_main)], axis=0
        )
        return OpponentControllerState(
            hidden_state=hidden_both, opp_hidden_state=hidden_opp
        ), key

    @override
    def select_actions(
        self,
        *,
        controller_state: OpponentControllerState,
        state: pgx.State,
        key: jax.Array,
    ):
        k_main, k_opp = jax.random.split(key)

        hidden_main = controller_state.hidden_state[0]        # [E, *main_state]
        hidden_opp = controller_state.opp_hidden_state         # [E, *opp_state]

        main_output = self.main_agent.sample_actions(state, k_main, hidden_main)
        opp_output = self.opponent_agent.sample_actions(state, k_opp, hidden_opp)

        is_main_turn = state.current_player == self.main_seat  # [E]

        actions = jnp.where(is_main_turn, main_output.actions, opp_output.actions)
        log_probs = jnp.where(
            is_main_turn, main_output.log_probs, jnp.zeros_like(main_output.log_probs)
        )
        values = main_output.values
        policy_logits = main_output.policy_logits
        valids_flag = is_main_turn

        # Update main state only when main acted, opponent state only when opp acted.
        is_main = _broadcast_per_env(is_main_turn, hidden_main)
        new_hidden_main = jnp.where(is_main, main_output.hidden_state, hidden_main)

        is_opp = _broadcast_per_env(~is_main_turn, hidden_opp)
        new_hidden_opp = jnp.where(is_opp, opp_output.hidden_state, hidden_opp)

        # Slot 1 stays zeros; only slot 0 carries the main agent's live state.
        new_hidden_both = controller_state.hidden_state.at[0].set(new_hidden_main)

        new_state = OpponentControllerState(
            hidden_state=new_hidden_both, opp_hidden_state=new_hidden_opp
        )
        return (actions, log_probs, values, policy_logits, valids_flag, new_state)

    @override
    def post_step(
        self,
        *,
        controller_state: OpponentControllerState,
        next_state,
        done_next: jnp.ndarray,
        current_player: jnp.ndarray,
        key: jax.Array,
    ):
        done_main = _broadcast_per_env(done_next, controller_state.hidden_state[0])[None]
        done_opp = _broadcast_per_env(done_next, controller_state.opp_hidden_state)
        return OpponentControllerState(
            hidden_state=jnp.where(done_main, 0.0, controller_state.hidden_state),
            opp_hidden_state=jnp.where(done_opp, 0.0, controller_state.opp_hidden_state),
        )

    @override
    def finalize(self, *, rollout_buffer: RolloutBuffer, controller_state):
        return rollout_buffer, None
