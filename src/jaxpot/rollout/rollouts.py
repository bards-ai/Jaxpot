"""
Actor-based rollout functions.

These functions use the rollout actor interface for action selection,
providing a cleaner abstraction over model-based rollouts.
"""

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from jaxpot.agents.base_rollout_actor import BaseRolloutActor
from jaxpot.rollout.aux_target_hooks import AuxTargetHook
from jaxpot.rollout.buffer import RolloutBuffer, init_aux_target_buffers
from jaxpot.rollout.controllers import (
    AgentOpponentRolloutController,
    ControllerState,
    RolloutController,
    SelfPlayController,
)


def _rollout_core_impl(
    init,
    step_fn,
    num_envs: int,
    num_steps: int,
    key: jax.Array,
    action_shape: tuple[int, ...],
    *,
    controller: RolloutController,
    aux_target_hooks: tuple[AuxTargetHook, ...] = (),
) -> tuple[RolloutBuffer, Any]:
    """
    Core rollout implementation using agent-based controllers.

    Always stores per-step hidden states in the RolloutBuffer. The hidden
    state shape is derived from the controller's initial state, so callers do
    not need to pass it explicitly — non-recurrent actors yield a dummy
    ``(1, 1)`` shape via :py:meth:`BaseRolloutActor.init_hidden_state`.
    """
    keys = jax.random.split(key, num_envs)
    state = init(keys)

    controller_state, key = controller.init_state(num_envs=num_envs, key=key)

    # controller_state.hidden_state has shape [2, L, E, H]; derive (L, H) for the buffer.
    state_shape = controller_state.hidden_state.shape[2:]

    obs_buffer = jax.tree.map(
        lambda x: jnp.zeros((num_envs, num_steps, 2, *x.shape[1:]), dtype=x.dtype),
        state.observation,
    )

    rollout_buffer = RolloutBuffer(
        obs=obs_buffer,
        value=jnp.zeros((num_envs, num_steps, 2)),
        policy_logits=jnp.zeros((num_envs, num_steps, 2, *action_shape)),
        done=jnp.zeros((num_envs, num_steps, 2), dtype=bool),
        terminated=jnp.zeros((num_envs, num_steps, 2), dtype=bool),
        actions=jnp.zeros((num_envs, num_steps, 2), dtype=jnp.int32),
        legal_action_mask=jnp.zeros((num_envs, num_steps, 2, *action_shape), dtype=bool),
        rewards=jnp.zeros((num_envs, num_steps, 2), dtype=jnp.float32),
        log_prob=jnp.zeros((num_envs, num_steps, 2), dtype=jnp.float32),
        valids=jnp.zeros((num_envs, num_steps, 2), dtype=bool),
        aux_targets=init_aux_target_buffers(aux_target_hooks, num_envs, num_steps),
        hidden_state=jnp.zeros((num_envs, num_steps, 2, *state_shape)),
    )

    player_step_indices = jnp.zeros((num_envs, 2), dtype=jnp.int32)
    batch_indexing = jnp.arange(num_envs)

    def body(carry, _):
        state_c, rb_c, p_step_idx_c, key_c, controller_state_c = carry
        key_c, k_action, k_step, k_post = jax.random.split(key_c, 4)

        obs_curr = state_c.observation
        current_player = state_c.current_player
        lam_curr = state_c.legal_action_mask
        opp_player = 1 - current_player
        time_player_idx = p_step_idx_c[batch_indexing, current_player]

        (
            actions,
            log_prob_to_write,
            value_logits,
            policy_logits,
            valids_flag,
            controller_state_sel,
        ) = controller.select_actions(
            controller_state=controller_state_c,
            state=state_c,
            key=k_action,
        )
        controller_state_c = controller_state_sel

        keys = jax.random.split(k_step, num_envs)
        next_state = step_fn(state_c, actions, keys)
        done_next = jnp.logical_or(next_state.terminated, next_state.truncated)
        terminated_next = next_state.terminated

        rb_obs = jax.tree.map(
            lambda buf, val: buf.at[batch_indexing, time_player_idx, current_player].set(val),
            rb_c.obs,
            obs_curr,
        )
        rb_legal_action_mask = rb_c.legal_action_mask.at[
            batch_indexing, time_player_idx, current_player
        ].set(lam_curr)
        rb_value = rb_c.value.at[batch_indexing, time_player_idx, current_player].set(value_logits)
        rb_policy_logits = rb_c.policy_logits.at[
            batch_indexing, time_player_idx, current_player
        ].set(policy_logits)
        rb_actions = rb_c.actions.at[batch_indexing, time_player_idx, current_player].set(actions)
        rb_done = rb_c.done.at[batch_indexing, time_player_idx, current_player].set(done_next)
        rb_terminated = rb_c.terminated.at[batch_indexing, time_player_idx, current_player].set(
            terminated_next
        )
        rb_log_prob = rb_c.log_prob.at[batch_indexing, time_player_idx, current_player].set(
            log_prob_to_write
        )
        rb_valids = rb_c.valids.at[batch_indexing, time_player_idx, current_player].set(valids_flag)

        # Write the current player's INPUT hidden state into the buffer.
        # controller_state_c.hidden_state has shape [2_players, E, *state_shape];
        # swap to [E, 2_players, *state_shape] for per-env current-player gather.
        hidden_per_env = controller_state_c.hidden_state.swapaxes(0, 1)
        hidden_curr = hidden_per_env[batch_indexing, current_player]  # [E, *state_shape]
        rb_hidden_state = rb_c.hidden_state.at[
            batch_indexing, time_player_idx, current_player
        ].set(hidden_curr)

        opp_time_prev_idx = p_step_idx_c[batch_indexing, opp_player] - 1
        opp_time_prev_idx = jax.lax.clamp(0, opp_time_prev_idx, num_steps - 1)

        opp_has_prev_in_ep = jnp.logical_and(
            p_step_idx_c[batch_indexing, opp_player] > 0,
            jnp.logical_not(rb_done[batch_indexing, opp_time_prev_idx, opp_player]),
        )
        opp_write_mask = jnp.logical_and(done_next, opp_has_prev_in_ep)

        rb_done = rb_done.at[batch_indexing, opp_time_prev_idx, opp_player].set(
            jnp.where(
                opp_write_mask, done_next, rb_done[batch_indexing, opp_time_prev_idx, opp_player]
            )
        )
        rb_terminated = rb_terminated.at[batch_indexing, opp_time_prev_idx, opp_player].set(
            jnp.where(
                opp_write_mask,
                terminated_next,
                rb_terminated[batch_indexing, opp_time_prev_idx, opp_player],
            )
        )

        curr_rewards = next_state.rewards[batch_indexing, current_player]
        opp_rewards = next_state.rewards[batch_indexing, opp_player]
        rb_rewards = rb_c.rewards.at[batch_indexing, time_player_idx, current_player].set(
            curr_rewards
        )
        rb_rewards = rb_rewards.at[batch_indexing, opp_time_prev_idx, opp_player].set(
            jnp.where(
                opp_write_mask,
                opp_rewards,
                rb_rewards[batch_indexing, opp_time_prev_idx, opp_player],
            )
        )

        rb_aux_targets_buffer = rb_c
        for hook in aux_target_hooks:
            target_values = hook.collect(state_c, current_player, opp_player)
            rb_aux_targets_buffer = hook.update_buffer(
                rb_aux_targets_buffer,
                batch_indexing,
                time_player_idx,
                current_player,
                target_values,
            )

        controller_state_n = controller.post_step(
            controller_state=controller_state_c,
            next_state=next_state,
            done_next=done_next,
            current_player=current_player,
            key=k_post,
        )

        rb_n = RolloutBuffer(
            obs=rb_obs,
            value=rb_value,
            policy_logits=rb_policy_logits,
            legal_action_mask=rb_legal_action_mask,
            done=rb_done,
            terminated=rb_terminated,
            actions=rb_actions,
            rewards=rb_rewards,
            log_prob=rb_log_prob,
            valids=rb_valids,
            aux_targets=rb_aux_targets_buffer.aux_targets,
            hidden_state=rb_hidden_state,
        )

        p_step_idx_n = p_step_idx_c.at[batch_indexing, current_player].add(1)
        return (next_state, rb_n, p_step_idx_n, key_c, controller_state_n), None

    (
        (
            state,
            rollout_buffer,
            player_step_indices,
            key,
            controller_state,
        ),
        _,
    ) = jax.lax.scan(
        body,
        (state, rollout_buffer, player_step_indices, key, controller_state),
        None,
        length=num_steps,
    )

    idxs = jnp.arange(num_steps)
    done_idxs = jnp.where(rollout_buffer.done, idxs[None, :, None], -1)
    last_done_idx = done_idxs.max(axis=1)
    has_done = last_done_idx >= 0
    time_idx = idxs[None, :, None]
    mask_after_last = (time_idx > last_done_idx[:, None, :]) & has_done[:, None, :]
    valids_trimmed = jnp.where(mask_after_last, False, rollout_buffer.valids)

    rollout_buffer = rollout_buffer.replace(valids=valids_trimmed)

    rollout_buffer, controller_outputs = controller.finalize(
        rollout_buffer=rollout_buffer, controller_state=controller_state
    )
    return rollout_buffer, controller_outputs


@partial(
    jax.jit,
    static_argnames=(
        "init",
        "step_fn",
        "num_envs",
        "num_steps",
        "action_shape",
        "aux_target_hooks",
    ),
)
def selfplay_rollout(
    agent: BaseRolloutActor,
    key: jax.Array,
    init,
    step_fn,
    num_envs: int,
    num_steps: int,
    action_shape: tuple[int, ...],
    aux_target_hooks: tuple[AuxTargetHook, ...] = (),
) -> tuple[RolloutBuffer, jax.Array, jax.Array]:
    """
    Collect a self-play batch using an agent.

    Both players use the same agent for action selection.

    Parameters
    ----------
    agent : BaseRolloutActor
        Agent to use for both players.
    key : jax.Array
        PRNG key.
    init : Callable
        Vectorized environment init.
    step_fn : Callable
        Vectorized environment step with auto reset.
    num_envs : int
        Parallel games.
    num_steps : int
        Max unroll steps.
    action_shape : tuple[int, ...]
        Action shape.

    Returns
    -------
    tuple[RolloutBuffer, jax.Array, jax.Array]
        Collected rollout buffers, number of completed rollouts, and number of valid steps.
    """
    controller = SelfPlayController(agent=agent)
    rollout_buffer, _ = _rollout_core_impl(
        init,
        step_fn,
        num_envs,
        num_steps,
        key,
        action_shape,
        controller=controller,
        aux_target_hooks=aux_target_hooks,
    )
    num_rollouts = rollout_buffer.done.sum()
    num_episodes = rollout_buffer.valids.sum()
    return rollout_buffer, num_rollouts, num_episodes


@partial(
    jax.jit,
    static_argnames=(
        "init",
        "step_fn",
        "num_envs",
        "num_steps",
        "action_shape",
        "main_seat",
        "aux_target_hooks",
    ),
)
def rollout_vs_opponent(
    main_agent: BaseRolloutActor,
    opponent_agent: BaseRolloutActor,
    key: jax.Array,
    init,
    step_fn,
    num_envs: int,
    num_steps: int,
    action_shape: tuple[int, ...],
    main_seat: int = 0,
    aux_target_hooks: tuple[AuxTargetHook, ...] = (),
) -> tuple[RolloutBuffer, jax.Array, jax.Array, jax.Array]:
    """
    Collect a rollout where the main agent plays against a fixed opponent agent.

    Parameters
    ----------
    main_agent : BaseRolloutActor
        Agent being trained (its steps are marked valid).
    opponent_agent : BaseRolloutActor
        Opponent agent (its steps are masked out).
    key : jax.Array
        PRNG key.
    init : Callable
        Vectorized environment init.
    step_fn : Callable
        Vectorized environment step with auto reset.
    num_envs : int
        Parallel games.
    num_steps : int
        Max unroll steps.
    action_shape : tuple[int, ...]
        Action shape.
    main_seat : int
        Seat of the main agent (0 or 1).

    Returns
    -------
    tuple[RolloutBuffer, jax.Array, jax.Array, jax.Array]
        Collected rollout buffers, number of completed rollouts, number of valid steps,
        and sum of rewards for the main agent.
    """
    controller = AgentOpponentRolloutController(
        main_agent=main_agent,
        opponent_agent=opponent_agent,
        main_seat=main_seat,
    )

    rollout_buffer, _ = _rollout_core_impl(
        init,
        step_fn,
        num_envs,
        num_steps,
        key,
        action_shape,
        controller=controller,
        aux_target_hooks=aux_target_hooks,
    )
    done_mask = rollout_buffer.done[:, :, main_seat]
    num_rollouts = done_mask.sum()
    num_episodes = rollout_buffer.valids[:, :, main_seat].sum()
    rewards_main = rollout_buffer.rewards[:, :, main_seat]
    sum_rewards = jnp.where(done_mask, rewards_main, 0.0).sum()
    return rollout_buffer, num_rollouts, num_episodes, sum_rewards
