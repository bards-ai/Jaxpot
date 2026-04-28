import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from jaxpot.agents import BaseRolloutActor


def _compute_eval_stats(
    total_rewards: jax.Array,
    total_terminals: jax.Array,
    total_wins: jax.Array,
    total_losses: jax.Array,
    total_action_counts: jax.Array,
    num_envs: int,
    num_steps: int,
) -> dict:
    """Compute evaluation statistics from accumulated counters."""
    term_sum = total_terminals.sum()
    sum_rewards = total_rewards.sum()
    sum_wins = total_wins.sum()
    sum_losses = total_losses.sum()

    denom = jnp.maximum(term_sum.astype(jnp.float32), 1.0)
    avg_reward = sum_rewards / denom
    win_rate = sum_wins.astype(jnp.float32) / denom
    lose_rate = sum_losses.astype(jnp.float32) / denom
    draw_rate = jnp.maximum(0.0, 1.0 - win_rate - lose_rate)
    done_rate = term_sum.astype(jnp.float32) / jnp.array(num_envs * num_steps, dtype=jnp.float32)
    action_total = jnp.maximum(total_action_counts.sum(), jnp.array(1.0, dtype=jnp.float32))
    action_frac = total_action_counts / action_total

    return {
        "avg_reward": avg_reward,
        "win_rate": win_rate,
        "lose_rate": lose_rate,
        "draw_rate": draw_rate,
        "done_rate": done_rate,
        "num_games": term_sum,
        "action_counts": total_action_counts,
        "action_frac": action_frac,
    }


def _compute_mover_stats(
    terminals: jax.Array,
    wins: jax.Array,
    losses: jax.Array,
    rewards: jax.Array | None = None,
) -> dict:
    """Compute win/lose/draw rates from accumulated counters."""
    term_sum = terminals.sum()
    denom = jnp.maximum(term_sum.astype(jnp.float32), 1.0)
    win_rate = wins.sum().astype(jnp.float32) / denom
    lose_rate = losses.sum().astype(jnp.float32) / denom
    draw_rate = jnp.maximum(0.0, 1.0 - win_rate - lose_rate)
    stats = {
        "num_games": term_sum,
        "win_rate": win_rate,
        "lose_rate": lose_rate,
        "draw_rate": draw_rate,
    }
    if rewards is not None:
        stats["avg_reward"] = rewards.sum() / denom
    return stats


def _evaluate(
    agent: BaseRolloutActor,
    opponent: BaseRolloutActor | None,
    key: jax.Array,
    init,
    step_fn,
    num_envs: int,
    num_steps: int,
    model_seat: int,
    deterministic: bool = False,
) -> dict:
    """Core evaluation loop. When opponent is None, uses random actions."""
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_envs)
    state = init(keys)
    A = state.legal_action_mask.shape[-1]

    hidden_state = agent.init_hidden_state(num_envs)
    first_mover_init = state.current_player

    def body(carry, _):
        (
            state_c,
            key_c,
            hidden_state_c,
            first_mover_c,
            total_rewards_c,
            total_terminals_c,
            total_wins_c,
            total_losses_c,
            action_counts_c,
            fm_rewards_c,
            fm_terminals_c,
            fm_wins_c,
            fm_losses_c,
        ) = carry

        curr = state_c.current_player

        key_c, k_model, k_opp, k_step = jax.random.split(key_c, 4)

        # Model actions (use the carry hidden state, not the initial closure)
        agent_output = agent.sample_actions(state_c, k_model, hidden_state_c)
        new_hidden_state = agent_output.hidden_state

        if deterministic:
            model_actions = jnp.argmax(agent_output.policy_logits, axis=-1)
        else:
            pi = distrax.Categorical(logits=agent_output.policy_logits)
            model_actions = pi.sample(seed=k_model)

        # Opponent actions (random when opponent is None)
        if opponent is not None:
            opp_logits = opponent.sample_actions(state_c, k_opp, hidden_state_c).policy_logits
        else:
            opp_logits = jnp.where(state_c.legal_action_mask, 1.0, -1e9)
        opp_actions = distrax.Categorical(logits=opp_logits).sample(seed=k_opp)

        actions = jnp.asarray(
            jnp.where(curr == model_seat, model_actions, opp_actions), dtype=jnp.int32
        )

        model_acted_mask = (curr == model_seat).astype(jnp.float32)
        counts_inc = jnp.bincount(actions, weights=model_acted_mask, length=A).astype(jnp.float32)

        keys = jax.random.split(k_step, num_envs)
        next_state = step_fn(state_c, actions, keys)
        done_next = jnp.logical_or(next_state.terminated, next_state.truncated)

        rew = next_state.rewards[jnp.arange(num_envs), model_seat]
        term_inc = done_next.astype(jnp.int32)
        win_inc = jnp.logical_and(done_next, rew > 0).astype(jnp.int32)
        loss_inc = jnp.logical_and(done_next, rew < 0).astype(jnp.int32)

        total_rewards_n = total_rewards_c + jnp.where(done_next, rew, 0.0)
        total_terminals_n = total_terminals_c + term_inc
        total_wins_n = total_wins_c + win_inc
        total_losses_n = total_losses_c + loss_inc
        action_counts_n = action_counts_c + counts_inc

        # First-mover stats
        model_was_first = first_mover_c == model_seat
        fm_rewards_n = fm_rewards_c + jnp.where(
            jnp.logical_and(model_was_first, done_next), rew, 0.0
        )
        fm_terminals_n = fm_terminals_c + jnp.where(model_was_first, term_inc, 0)
        fm_wins_n = fm_wins_c + jnp.where(model_was_first, win_inc, 0)
        fm_losses_n = fm_losses_c + jnp.where(model_was_first, loss_inc, 0)

        # After auto_reset, update first_mover for new episodes
        first_mover_n = jnp.where(done_next, next_state.current_player, first_mover_c)

        # Update hidden state only for steps where model acted, reset on done.
        # hidden_state_c is batch-first ``[E, *state_shape]``; broadcast the
        # per-env masks across the opaque trailing dims.
        trailing = (1,) * (hidden_state_c.ndim - 1)
        is_model = (curr == model_seat).reshape(-1, *trailing)
        hidden_state_updated = jnp.where(is_model, new_hidden_state, hidden_state_c)
        done_mask = done_next.reshape(-1, *trailing)
        hidden_state_updated = jnp.where(done_mask, 0.0, hidden_state_updated)

        return (
            next_state,
            key_c,
            hidden_state_updated,
            first_mover_n,
            total_rewards_n,
            total_terminals_n,
            total_wins_n,
            total_losses_n,
            action_counts_n,
            fm_rewards_n,
            fm_terminals_n,
            fm_wins_n,
            fm_losses_n,
        ), None

    stats_init = (
        jnp.zeros((num_envs,), dtype=jnp.float32),  # total_rewards
        jnp.zeros((num_envs,), dtype=jnp.int32),  # total_terminals
        jnp.zeros((num_envs,), dtype=jnp.int32),  # total_wins
        jnp.zeros((num_envs,), dtype=jnp.int32),  # total_losses
        jnp.zeros((A,), dtype=jnp.float32),  # action_counts
        jnp.zeros((num_envs,), dtype=jnp.float32),  # fm_rewards
        jnp.zeros((num_envs,), dtype=jnp.int32),  # fm_terminals
        jnp.zeros((num_envs,), dtype=jnp.int32),  # fm_wins
        jnp.zeros((num_envs,), dtype=jnp.int32),  # fm_losses
    )
    init_carry = (state, key, hidden_state, first_mover_init, *stats_init)

    final_carry, _ = jax.lax.scan(body, init_carry, xs=None, length=num_steps)

    (
        _,
        _,
        _,
        _,
        total_rewards,
        total_terminals,
        total_wins,
        total_losses,
        total_action_counts,
        fm_rewards,
        fm_terminals,
        fm_wins,
        fm_losses,
    ) = final_carry

    result = _compute_eval_stats(
        total_rewards,
        total_terminals,
        total_wins,
        total_losses,
        total_action_counts,
        num_envs,
        num_steps,
    )

    # p0 stats (model moved first)
    result["p0"] = _compute_mover_stats(fm_terminals, fm_wins, fm_losses, fm_rewards)

    # p1 stats (model moved second) — derived from totals
    result["p1"] = _compute_mover_stats(
        total_terminals - fm_terminals,
        total_wins - fm_wins,
        total_losses - fm_losses,
        total_rewards - fm_rewards,
    )

    return result


def evaluate_vs_random(
    model: BaseRolloutActor,
    key: jax.Array,
    init,
    step_fn,
    num_envs: int,
    num_steps: int,
    model_seat: int,
    deterministic: bool = False,
) -> dict:
    return _evaluate(
        model,
        None,
        key,
        init,
        step_fn,
        num_envs,
        num_steps,
        model_seat,
        deterministic,
    )


evaluate_vs_random_jited = nnx.jit(evaluate_vs_random, static_argnums=(2, 3, 4, 5, 6, 7))


def evaluate_vs_opponent(
    model: BaseRolloutActor,
    opponent: BaseRolloutActor,
    key: jax.Array,
    init,
    step_fn,
    num_envs: int,
    num_steps: int,
    model_seat: int = 0,
    deterministic: bool = False,
) -> dict:
    return _evaluate(
        model,
        opponent,
        key,
        init,
        step_fn,
        num_envs,
        num_steps,
        model_seat,
        deterministic,
    )


evaluate_vs_opponent_jited = nnx.jit(evaluate_vs_opponent, static_argnums=(3, 4, 5, 6, 7, 8))
