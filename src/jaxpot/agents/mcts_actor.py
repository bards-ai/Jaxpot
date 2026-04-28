"""
MCTSActor: a BaseRolloutActor backed by mctx.gumbel_muzero_policy.

Gumbel MuZero guarantees policy improvement (given correct value estimates)
and is more simulation-efficient than PUCT-based search.

The policy_logits field of AgentOutput stores visit-count probability
distribution (action_weights) from MCTS as the training target
for the AlphaZero policy loss.
"""

from __future__ import annotations

from typing import Callable, override

import jax
import jax.numpy as jnp
import mctx
import pgx
from flax import struct

from jaxpot.agents.base_rollout_actor import AgentOutput, BaseRolloutActor
from jaxpot.alphazero.mcts import MCTSConfig, make_root_and_recurrent_fn
from jaxpot.models.base import PolicyValueModel


@struct.dataclass
class MCTSActor(BaseRolloutActor):
    """Actor that uses mctx.muzero_policy for action selection.

    JAX PyTree compatible with JIT-compiled rollout controllers.

    Parameters
    ----------
    model : BaseModel
        Policy/value network returning ModelOutput(.policy_logits, .value).
    mcts_config : MCTSConfig
        Search hyperparameters.
    """

    model: PolicyValueModel
    mcts_config: MCTSConfig = struct.field(pytree_node=False)
    step_fn: Callable[[pgx.State, jnp.ndarray, jax.Array], pgx.State] | None = struct.field(
        default=None, pytree_node=False
    )
    no_auto_reset_step_fn: Callable[
        [pgx.State, jnp.ndarray, jax.Array], pgx.State
    ] | None = struct.field(default=None, pytree_node=False)

    def setup(
        self,
        *,
        step_fn: Callable[[pgx.State, jnp.ndarray, jax.Array], pgx.State],
        no_auto_reset_step_fn: Callable[[pgx.State, jnp.ndarray, jax.Array], pgx.State],
    ) -> "MCTSActor":
        """Attach env transition functions.

        Parameters
        ----------
        step_fn:
            Rollout step function (typically with auto_reset).
        no_auto_reset_step_fn:
            Step function used exclusively inside MCTS tree search.  Should be
            a plain ``env.step`` without auto_reset so terminal states are
            correctly propagated as dead nodes in the search tree.
        """
        return self.replace(step_fn=step_fn, no_auto_reset_step_fn=no_auto_reset_step_fn)

    @override
    def sample_actions(
        self,
        state: pgx.State,
        key: jax.Array,
        hidden_state: jnp.ndarray | None = None,
    ) -> AgentOutput:
        # MCTS does not maintain a recurrent hidden state across calls; the
        # argument is accepted for interface uniformity and passed through.
        passthrough_hidden = hidden_state
        key, k_search = jax.random.split(key, 2)

        if self.step_fn is None or self.no_auto_reset_step_fn is None:
            raise ValueError(
                "MCTSActor requires both step_fn and no_auto_reset_step_fn. "
                "Call actor.setup(step_fn=..., no_auto_reset_step_fn=...) before "
                "using MCTSActor in rollouts."
            )

        root, recurrent_fn = make_root_and_recurrent_fn(
            self.model, state, self.no_auto_reset_step_fn, self.mcts_config.gamma
        )

        policy_output: mctx.PolicyOutput = mctx.gumbel_muzero_policy(
            params=None,
            rng_key=k_search,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=self.mcts_config.num_simulations,
            max_num_considered_actions=self.mcts_config.max_num_considered_actions,
            invalid_actions=~state.legal_action_mask,
        )

        root_values = policy_output.search_tree.node_values[:, mctx.Tree.ROOT_INDEX]

        return AgentOutput(
            actions=policy_output.action,
            log_probs=jnp.zeros_like(
                policy_output.action, dtype=policy_output.action_weights.dtype
            ),  # not used
            values=root_values,
            policy_logits=jnp.where(
                state.legal_action_mask, policy_output.action_weights, 0.0
            ),  # policy_tgt
            hidden_state=passthrough_hidden,
        )
