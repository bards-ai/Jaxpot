"""
AlphaZero MCTS via mctx.

Provides MCTSConfig and a helper that builds the mctx root + recurrent_fn
from a BaseModel, so MCTSActor can call mctx.gumbel_muzero_policy directly.

gumbel_muzero_policy is preferred over muzero_policy because it guarantees
policy improvement (given correct value estimates) and is more
simulation-efficient via sequential halving rather than PUCT.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import mctx
import pgx

from jaxpot.models.base import PolicyValueModel


class MCTSConfig(NamedTuple):
    """Hyperparameters for Gumbel MuZero search.

    Parameters
    ----------
    num_simulations : int
        Number of MCTS simulations per root position.
        Gumbel MuZero is efficient even with small budgets (e.g. 32–64).
    max_num_considered_actions : int
        Number of actions considered at the root before sequential halving.
        Typically set to num_actions or a smaller subset for large action spaces.
    gamma : float
        Discount factor. 1.0 = undiscounted (standard AlphaZero).
    """

    num_simulations: int = 32
    max_num_considered_actions: int = 16
    gamma: float = 1.0


def make_root_and_recurrent_fn(
    model: PolicyValueModel,
    state: pgx.State,
    step_fn: Callable[[pgx.State, jnp.ndarray, jax.Array], pgx.State],
    gamma: float,
) -> tuple[mctx.RootFnOutput, mctx.RecurrentFn]:
    """Build the mctx root output and recurrent_fn from a model + env transition.

    Parameters
    ----------
    model : PolicyValueModel
        Policy/value network. Called as model(obs_batch) -> ModelOutput.
    state : pgx.State
        Batched PGX environment state. Used as the mctx embedding so the search
        can advance by applying actions.
    step_fn : Callable
        Vectorized environment step: step_fn(state, actions, keys) -> next_state.
    gamma : float
        Discount factor. For 2-player zero-sum alternating-move games we pass
        `discount = -gamma` for non-terminal transitions to flip the value
        perspective after each ply (matching the pgx AlphaZero example).

    Returns
    -------
    root : mctx.RootFnOutput
        Root prior logits, values, and state embedding for mctx.
    recurrent_fn : mctx.RecurrentFn
        Node-expansion function for mctx simulations.
    """
    root_out = model(state.observation)
    root_logits = root_out.policy_logits
    root_logits = jnp.where(state.legal_action_mask, root_logits, jnp.finfo(root_logits.dtype).min)
    root_values = root_out.value.squeeze(-1)

    root = mctx.RootFnOutput(
        prior_logits=root_logits,
        value=root_values,
        embedding=state,
    )

    def recurrent_fn(params, rng: jax.Array, action: jnp.ndarray, embedding: pgx.State):
        del params
        batch_size = embedding.observation.shape[0]
        keys = jax.random.split(rng, batch_size)
        actor = embedding.current_player
        next_state = step_fn(embedding, action, keys)
        done = jnp.logical_or(next_state.terminated, next_state.truncated)

        out = model(next_state.observation)
        logits = out.policy_logits
        logits = jnp.where(next_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        values = out.value.squeeze(-1)
        values = jnp.where(done, 0.0, values)

        reward = next_state.rewards[jnp.arange(batch_size), actor]
        discount = jnp.where(done, 0.0, -gamma)

        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=values,
        ), next_state

    return root, recurrent_fn
