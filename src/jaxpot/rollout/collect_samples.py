from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from jaxpot.agents import BaseRolloutActor, PolicyActor
from jaxpot.league import LeagueManager
from jaxpot.rollout.advantage_gatherers import AdvantageGatherer
from jaxpot.rollout.aux_target_hooks import AuxTargetHook
from jaxpot.rollout.buffer import (
    RolloutAuxTransform,
    TrainingDataBuffer,
    concatenate_training_data,
    create_training_buffer,
)
from jaxpot.rollout.rollouts import rollout_vs_opponent, selfplay_rollout

if TYPE_CHECKING:
    from jaxpot.league import LeagueManager


def _models_to_policy_actors(models: tuple) -> tuple:
    """Convert a tuple of models to PolicyActor instances."""
    return tuple(PolicyActor(model=m) for m in models)


def collect_selfplay(
    agent: BaseRolloutActor,
    key: jax.Array,
    init,
    step_fn,
    num_envs: int,
    num_steps: int,
    action_shape: tuple[int, ...],
    advantage_gatherer: AdvantageGatherer,
    seq_len: int = 1,
    aux_target_hooks: tuple[AuxTargetHook, ...] = (),
    rollout_transforms: tuple[RolloutAuxTransform, ...] = (),
) -> tuple[TrainingDataBuffer, jax.Array, jax.Array]:
    """
    Collect a self-play training batch using an agent.

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
        Number of parallel games.
    num_steps : int
        Max unroll steps per game.
    action_shape : tuple[int, ...]
        Action shape.
    seq_len: int
        Sequence length for training.
    advantage_gatherer : AdvantageGatherer
        Strategy object for constructing advantages / value masks.
    aux_target_hooks : tuple[AuxTargetHook, ...]
        Auxiliary target hooks.
    rollout_transforms : tuple[RolloutAuxTransform, ...]
        Rollout transforms.

    Returns
    -------
    tuple[TrainingDataBuffer, jax.Array, jax.Array]
        Training batch, number of completed rollouts, and number of completed episodes.
    """
    rollout_buffer, num_rollouts, num_episodes = selfplay_rollout(
        agent,
        key,
        init,
        step_fn,
        num_envs,
        num_steps,
        action_shape,
        aux_target_hooks=aux_target_hooks,
    )
    advantages, v_masks = advantage_gatherer.gather(
        rollout_buffer, seats=(0, 1), num_steps=num_steps
    )
    training_data = create_training_buffer(
        rollout_buffer,
        advantages,
        seats=(0, 1),
        seq_len=seq_len,
        rollout_transforms=rollout_transforms,
        value_loss_masks=v_masks,
    )
    return training_data, num_rollouts, num_episodes


def collect_vs_opponent(
    main_agent: BaseRolloutActor,
    opponent_agent: BaseRolloutActor,
    key: jax.Array,
    init,
    step_fn,
    num_envs: int,
    num_steps: int,
    action_shape: tuple[int, ...],
    advantage_gatherer: AdvantageGatherer,
    main_seat: int = 0,
    seq_len: int = 1,
    aux_target_hooks: tuple[AuxTargetHook, ...] = (),
    rollout_transforms: tuple[RolloutAuxTransform, ...] = (),
) -> tuple[TrainingDataBuffer, jax.Array, jax.Array, jax.Array]:
    """
    Collect a training batch where the main agent plays against a fixed opponent.

    Parameters
    ----------
    main_agent : BaseRolloutActor or PPOAgent
        Agent being trained (its steps are marked valid). PPOAgent is automatically converted.
    opponent_agent : BaseRolloutActor or PPOAgent
        Opponent agent (its steps are masked out). PPOAgent is automatically converted.
    key : jax.Array
        PRNG key.
    init : Callable
        Vectorized environment init.
    step_fn : Callable
        Vectorized environment step with auto reset.
    num_envs : int
        Number of parallel games.
    num_steps : int
        Max unroll steps per game.
    action_shape : tuple[int, ...]
        Action shape.
    main_seat : int
        Seat of the main agent (0 or 1).
    advantage_gatherer : AdvantageGatherer
        Strategy object for constructing advantages / value masks.

    Returns
    -------
    tuple[TrainingDataBuffer, jax.Array, jax.Array, jax.Array]
        Training batch, number of completed rollouts, number of completed episodes and sum of rewards.
    """
    rollout_buffer, num_rollouts, num_episodes, sum_rewards = rollout_vs_opponent(
        main_agent,
        opponent_agent,
        key,
        init,
        step_fn,
        num_envs,
        num_steps,
        action_shape,
        main_seat,
        aux_target_hooks=aux_target_hooks,
    )
    advantages, v_masks = advantage_gatherer.gather(
        rollout_buffer, seats=(main_seat,), num_steps=num_steps
    )
    training_data = create_training_buffer(
        rollout_buffer,
        advantages,
        seats=(main_seat,),
        seq_len=seq_len,
        rollout_transforms=rollout_transforms,
        value_loss_masks=v_masks,
    )
    return training_data, num_rollouts, num_episodes, sum_rewards


def _allocate_envs_per_opponent(w: jax.Array, num_envs: int, base_unit: int = 256) -> jax.Array:
    """
    Allocate environment slots per opponent in `base_unit`-sized chunks.

    Parameters
    ----------
    w : jax.Array
        Sampling weights for each opponent.
    num_envs : int
        Total number of environments; must be divisible by `base_unit`.
    base_unit : int
        Base unit size for allocation.

    Returns
    -------
    jax.Array
        Environment counts per opponent, each a multiple of `base_unit` that sums to
        `num_envs`.
    """
    num_opponents = len(w)
    if num_opponents == 0:
        raise ValueError("No opponents available for league collection.")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if num_envs % base_unit != 0:
        raise ValueError(f"num_envs must be divisible by {base_unit} for batched rollouts.")

    weights = jnp.asarray(w, dtype=jnp.float32)
    weight_sum = float(jnp.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("Opponent weights must sum to a positive value.")

    norm_weights = weights / weight_sum
    total_units = num_envs // base_unit
    raw_units = norm_weights * total_units

    units = jnp.floor(raw_units).astype(jnp.int32)
    remaining_units = total_units - int(jnp.sum(units))
    if remaining_units > 0:
        frac_order = jnp.argsort(raw_units - units)[::-1]
        top_indices = frac_order[:remaining_units]
        units = units.at[top_indices].add(1)

    return units * base_unit


def collect_league(
    main_agent: BaseRolloutActor,
    league: "LeagueManager",
    key: jax.Array,
    init,
    step_fn,
    num_envs: int,
    num_steps: int,
    action_shape: tuple[int, ...],
    advantage_gatherer: AdvantageGatherer,
    base_unit: int = 256,
    main_seat: int = 0,
    seq_len: int = 1,
    aux_target_hooks: tuple[AuxTargetHook, ...] = (),
    rollout_transforms: tuple[RolloutAuxTransform, ...] = (),
) -> tuple[TrainingDataBuffer, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Collect a training batch where the main agent plays against a league of opponents.

    This version runs a separate rollout_vs_opponent per opponent to avoid
    computing unused actions for non-selected opponents each step.
    """
    opponent_models, weights = league.get_league_models_and_weights()
    opponent_agents = _models_to_policy_actors(opponent_models)
    num_opponents = len(opponent_agents)
    if num_opponents == 0:
        raise ValueError("No opponents available in league.")

    env_splits = _allocate_envs_per_opponent(weights, num_envs, base_unit)
    key, *subkeys = jax.random.split(key, num_opponents + 1)

    training_batches = []
    term_counts_list = []
    sum_rewards_list = []
    num_rollouts_total = jnp.array(0, dtype=jnp.int32)
    num_episodes_total = jnp.array(0, dtype=jnp.int32)

    for idx, (opp_agent, env_count, k_sub) in enumerate(
        zip(opponent_agents, env_splits, subkeys, strict=True)
    ):
        if int(env_count) == 0:
            term_counts_list.append(jnp.array(0, dtype=jnp.int32))
            sum_rewards_list.append(jnp.array(0.0, dtype=jnp.float32))
            continue

        batch, num_rollouts, num_episodes, rewards_sum = collect_vs_opponent(
            main_agent=main_agent,
            opponent_agent=opp_agent,
            key=k_sub,
            init=init,
            step_fn=step_fn,
            num_envs=int(env_count),
            num_steps=num_steps,
            action_shape=action_shape,
            main_seat=main_seat,
            seq_len=seq_len,
            aux_target_hooks=aux_target_hooks,
            rollout_transforms=rollout_transforms,
            advantage_gatherer=advantage_gatherer,
        )
        training_batches.append(batch)
        num_rollouts_total = num_rollouts_total + num_rollouts
        num_episodes_total = num_episodes_total + num_episodes
        term_counts_list.append(num_rollouts)
        sum_rewards_list.append(rewards_sum)

    league_term = jnp.stack(term_counts_list)
    league_rewards = jnp.stack(sum_rewards_list)
    training_data = concatenate_training_data(training_batches)
    return training_data, num_rollouts_total, num_episodes_total, league_term, league_rewards


def collect_archive_league(
    main_agent: BaseRolloutActor,
    league: "LeagueManager",
    key: jax.Array,
    init,
    step_fn,
    num_envs: int,
    num_steps: int,
    action_shape: tuple[int, ...],
    advantage_gatherer: AdvantageGatherer,
    main_seat: int = 0,
    seq_len: int = 1,
    base_unit: int = 256,
    aux_target_hooks: tuple[AuxTargetHook, ...] = (),
    rollout_transforms: tuple[RolloutAuxTransform, ...] = (),
) -> tuple[TrainingDataBuffer, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Collect a training batch where the main agent plays against archived league opponents.

    Parameters
    ----------
    main_agent : BaseRolloutActor or PPOAgent
        Agent being trained. PPOAgent is automatically converted.
    league : LeagueManager
        League manager containing archived opponent models and weights.
    key : jax.Array
        PRNG key.
    init : Callable
        Vectorized environment init.
    step_fn : Callable
        Vectorized environment step with auto reset.
    num_envs : int
        Number of parallel games.
    num_steps : int
        Max unroll steps per game.
    advantage_gatherer : AdvantageGatherer
        Strategy object for constructing advantages / value masks.
    action_shape : tuple[int, ...]
        Action shape.
    main_seat : int
        Seat of the main agent (0 or 1).
    seq_len : int
        Sequence length for training.
    aux_target_hooks : tuple[AuxTargetHook, ...]
        Auxiliary target hooks.
    rollout_transforms : tuple[RolloutAuxTransform, ...]
        Rollout transforms.

    Returns
    -------
    tuple[TrainingDataBuffer, jax.Array, jax.Array, jax.Array, jax.Array]
        Training batch, number of completed rollouts, number of completed episodes and
        statistics for the league - termination and sum of rewards per opponent.
    """
    opponent_models, weights = league.get_archive_models_and_weights()
    opponent_agents = _models_to_policy_actors(opponent_models)
    num_opponents = len(opponent_agents)
    if num_opponents == 0:
        raise ValueError("No opponents available in league.")

    env_splits = _allocate_envs_per_opponent(weights, num_envs, base_unit)
    key, *subkeys = jax.random.split(key, num_opponents + 1)

    training_batches = []
    term_counts_list = []
    sum_rewards_list = []
    num_rollouts_total = jnp.array(0, dtype=jnp.int32)
    num_episodes_total = jnp.array(0, dtype=jnp.int32)

    for idx, (opp_agent, env_count, k_sub) in enumerate(
        zip(opponent_agents, env_splits, subkeys, strict=True)
    ):
        if int(env_count) == 0:
            term_counts_list.append(jnp.array(0, dtype=jnp.int32))
            sum_rewards_list.append(jnp.array(0.0, dtype=jnp.float32))
            continue

        batch, num_rollouts, num_episodes, rewards_sum = collect_vs_opponent(
            main_agent=main_agent,
            opponent_agent=opp_agent,
            key=k_sub,
            init=init,
            step_fn=step_fn,
            num_envs=int(env_count),
            num_steps=num_steps,
            action_shape=action_shape,
            main_seat=main_seat,
            seq_len=seq_len,
            aux_target_hooks=aux_target_hooks,
            rollout_transforms=rollout_transforms,
            advantage_gatherer=advantage_gatherer,
        )
        training_batches.append(batch)
        num_rollouts_total = num_rollouts_total + num_rollouts
        num_episodes_total = num_episodes_total + num_episodes
        term_counts_list.append(num_rollouts)
        sum_rewards_list.append(rewards_sum)

    archive_term = jnp.stack(term_counts_list)
    archive_rewards = jnp.stack(sum_rewards_list)
    training_data = concatenate_training_data(training_batches)
    return training_data, num_rollouts_total, num_episodes_total, archive_term, archive_rewards
