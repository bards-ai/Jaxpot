import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
from typing import Callable

import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from jax.sharding import Mesh
from loguru import logger
from omegaconf import DictConfig
from pgx.experimental import auto_reset

from jaxpot.agents import BaseTrainingAgent, RandomActor
from jaxpot.env.history_wrapper import wrap_vectorized_env_with_history
from jaxpot.evaluator.base import BaseEvaluator
from jaxpot.evaluator.utils import log_evaluation_output
from jaxpot.loggers.logger import Logger
from jaxpot.rl.trainer import Trainer
from jaxpot.rollout.advantage_gatherers import AdvantageGatherer
from jaxpot.rollout.aux_target_hooks import AuxTargetHook
from jaxpot.rollout.buffer import concatenate_training_data
from jaxpot.rollout.collect_samples import (
    collect_archive_league,
    collect_league,
    collect_selfplay,
    collect_vs_opponent,
)
from jaxpot.utils.checkpoints import (
    BestCheckpointManager,
    BestCheckpointSelection,
    Checkpoint,
    CheckpointManager,
    configure_best_checkpoint_selection,
    create_lr_schedule,
)
from jaxpot.utils.logging import JsonLinesLogger, TrainingProgress, setup_loguru_logging
from jaxpot.utils.sharding import create_device_mesh, get_replicated_sharding
from jaxpot.utils.target_network import TargetNetwork
from jaxpot.utils.timer import Timer
from jaxpot.utils.utils import dump_debug_file


def build_aux_target_hooks(trainer: Trainer) -> tuple[AuxTargetHook, ...]:
    """Build rollout auxiliary target hooks based on env and trainer configuration."""
    hooks_by_target: dict[str, AuxTargetHook] = {}
    for aux_loss in trainer.auxiliary_losses:
        hook = aux_loss.make_target_hook()
        existing = hooks_by_target.get(hook.target_field)
        if existing is not None and type(existing) is not type(hook):
            raise ValueError(
                f"Conflicting hooks for target '{hook.target_field}': "
                f"{type(existing).__name__} vs {type(hook).__name__}"
            )
        hooks_by_target[hook.target_field] = hook
    return tuple(hooks_by_target.values())


def make_env_fns(cfg):
    env = instantiate(cfg.env)
    init = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    no_auto_reset_step_fn = jax.jit(jax.vmap(env.step))

    history_len = int(getattr(cfg, "env_history_len", 1))
    if history_len > 1:
        logger.info(f"Wrapping env observations with history_len={history_len}")
        init, step_fn = wrap_vectorized_env_with_history(
            init,
            step_fn,
            history_len=history_len,
        )
        _, no_auto_reset_step_fn = wrap_vectorized_env_with_history(
            init,
            no_auto_reset_step_fn,
            history_len=history_len,
        )
    return (
        env,
        init,
        step_fn,
        no_auto_reset_step_fn,
    )


def train(
    cfg: DictConfig,
    env,
    init: Callable,
    step_fn: Callable,
    no_auto_reset_step_fn: Callable,
    experiment_tracker: Logger,
    checkpoint: Checkpoint,
    checkpoint_manager: CheckpointManager,
    best_checkpoint_manager: BestCheckpointManager,
    mesh: Mesh,
) -> None:
    timer = Timer()
    start_iter = checkpoint.iteration
    key = checkpoint.key
    league = checkpoint.league
    num_rollouts = checkpoint.num_rollouts
    num_episodes = checkpoint.num_episodes
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    replicated_sharding = get_replicated_sharding(mesh)
    checkpoint.model = jax.device_put(checkpoint.model, replicated_sharding)
    checkpoint.optimizer = jax.device_put(checkpoint.optimizer, replicated_sharding)
    key = jax.device_put(key, replicated_sharding)
    num_actions = env.num_actions if isinstance(env.num_actions, tuple) else (env.num_actions,)

    trainer = instantiate(
        cfg.trainer,
        optimizer=checkpoint.optimizer,
        mesh=mesh,
        start_iteration=start_iter,
        _convert_="object",
    )
    aux_target_hooks = build_aux_target_hooks(trainer)
    rollout_transforms = trainer.get_rollout_transforms()

    advantage_gatherer: AdvantageGatherer = instantiate(cfg.advantage_gatherer, _convert_="object")

    agent: BaseTrainingAgent = instantiate(
        cfg.train_agent,
        model=checkpoint.model,
        trainer=trainer,
        _convert_="object",
    )
    logger.info(f"Initialized training agent: {type(agent).__name__}")

    target_network = None
    target_policy_actor = None
    if cfg.use_target_selfplay:
        target_network = TargetNetwork(agent.model, tau=cfg.target_tau)
        target_policy_actor = agent.rollout_actor.setup(
            step_fn=step_fn, no_auto_reset_step_fn=no_auto_reset_step_fn
        )

    random_agent = RandomActor()

    # Derive seq_len from trainer (defaults to 1 for non-recurrent)
    seq_len = int(trainer.seq_len)
    if seq_len > 1 and cfg.num_steps % seq_len != 0:
        raise ValueError(f"num_steps ({cfg.num_steps}) must be divisible by seq_len ({seq_len})")
    if agent.model.is_recurrent:
        logger.debug(f"Recurrent mode: hidden_shape={agent.model.hidden_shape}, seq_len={seq_len}")

    should_dump_debug = cfg.dump_debug_every is not None and cfg.dump_debug_every > 0

    it = start_iter
    lr_schedule = create_lr_schedule(cfg)

    # Initialize evaluators from config
    evaluators: list[BaseEvaluator] = []
    for eval_cfg in cfg.eval:
        logger.debug(f"Instantiating evaluator: {eval_cfg}")
        evaluator = instantiate(
            eval_cfg,
            init=init,
            step_fn=step_fn,
            no_auto_reset_step_fn=no_auto_reset_step_fn,
            agent=agent,
            league=league,
        )
        evaluators.append(evaluator)

    evaluator_eval_every_by_name = {e.name: int(e.eval_every) for e in evaluators}
    best_checkpoint_selection: BestCheckpointSelection = configure_best_checkpoint_selection(
        cfg, evaluator_eval_every_by_name
    )

    progress = TrainingProgress(cfg.total_iters, experiment_name=cfg.experiment_name)

    json_logger: JsonLinesLogger | None = None
    if cfg.log_to_file is not None:
        json_logger = JsonLinesLogger(output_dir / str(cfg.log_to_file))

    progress.start()
    logger.info("Starting training...")
    try:
        for it in range(start_iter, cfg.total_iters):
            timer.start("total")
            # Ensure different RNG each iteration
            key = jax.random.fold_in(key, it)

            agent.eval()
            is_random_warmup = it < int(cfg.random_warmup_iters)
            log_payload = {}
            random_batch, league_batch, archive_batch = None, None, None
            league_num_rollouts, league_sum_rewards = None, None
            archive_num_rollouts, archive_sum_rewards = None, None

            # Split keys upfront for all collectors
            key, k_self, k_rand, k_opp, k_arch = jax.random.split(key, 5)

            rollout_actor = agent.rollout_actor.setup(
                step_fn=step_fn, no_auto_reset_step_fn=no_auto_reset_step_fn
            )

            with timer("collect_selfplay"), mesh:
                if cfg.use_target_selfplay:
                    if target_policy_actor is None:
                        raise ValueError("Target policy actor is required for target self-play.")
                    selfplay_batch, selfplay_rollouts, selfplay_episodes, _ = collect_vs_opponent(
                        main_agent=rollout_actor,
                        opponent_agent=target_policy_actor,
                        key=k_self,
                        init=init,
                        step_fn=step_fn,
                        num_envs=cfg.selfplay_num_envs,
                        num_steps=cfg.num_steps,
                        action_shape=num_actions,
                        main_seat=0,
                        seq_len=seq_len,
                        aux_target_hooks=aux_target_hooks,
                        rollout_transforms=rollout_transforms,
                        advantage_gatherer=advantage_gatherer,
                    )
                else:
                    selfplay_batch, selfplay_rollouts, selfplay_episodes = collect_selfplay(
                        rollout_actor,
                        k_self,
                        init,
                        step_fn,
                        num_envs=cfg.selfplay_num_envs,
                        num_steps=cfg.num_steps,
                        action_shape=num_actions,
                        seq_len=seq_len,
                        aux_target_hooks=aux_target_hooks,
                        rollout_transforms=rollout_transforms,
                        advantage_gatherer=advantage_gatherer,
                    )

            if is_random_warmup and int(cfg.random_num_envs) > 0:
                with timer("collect_random"), mesh:
                    random_batch, random_rollouts, random_episodes, random_sum_rewards = (
                        collect_vs_opponent(
                            main_agent=rollout_actor,
                            opponent_agent=random_agent,
                            key=k_rand,
                            init=init,
                            step_fn=step_fn,
                            num_envs=cfg.random_num_envs,
                            num_steps=cfg.num_steps,
                            action_shape=num_actions,
                            seq_len=seq_len,
                            aux_target_hooks=aux_target_hooks,
                            rollout_transforms=rollout_transforms,
                            advantage_gatherer=advantage_gatherer,
                        )
                    )

            if not is_random_warmup and league.size() > 0:
                with timer("collect_league"), mesh:
                    (
                        league_batch,
                        league_rollouts,
                        league_episodes,
                        league_num_rollouts,
                        league_sum_rewards,
                    ) = collect_league(
                        rollout_actor,
                        league,
                        k_opp,
                        init,
                        step_fn,
                        num_envs=cfg.league_num_envs,
                        num_steps=cfg.num_steps,
                        action_shape=num_actions,
                        base_unit=cfg.base_unit,
                        seq_len=seq_len,
                        aux_target_hooks=aux_target_hooks,
                        rollout_transforms=rollout_transforms,
                        advantage_gatherer=advantage_gatherer,
                    )

            if not is_random_warmup and league.has_active_archive():
                with timer("collect_archive_league"), mesh:
                    (
                        archive_batch,
                        archive_rollouts,
                        archive_episodes,
                        archive_num_rollouts,
                        archive_sum_rewards,
                    ) = collect_archive_league(
                        rollout_actor,
                        league,
                        k_arch,
                        init,
                        step_fn,
                        num_envs=cfg.archive_num_envs,
                        num_steps=cfg.num_steps,
                        action_shape=num_actions,
                        base_unit=cfg.base_unit,
                        seq_len=seq_len,
                        aux_target_hooks=aux_target_hooks,
                        rollout_transforms=rollout_transforms,
                        advantage_gatherer=advantage_gatherer,
                    )

            with timer("concatenate_batches"):
                training_data = concatenate_training_data(
                    [selfplay_batch, random_batch, league_batch, archive_batch]
                )
                num_samples = int(training_data.adv.shape[0]) * seq_len
                num_valid_samples = jnp.sum(training_data.valids)
                valid_percentage = num_valid_samples / float(num_samples)

            agent.train()
            with timer("training"):
                metrics = agent.update(training_data=training_data)
            if cfg.use_target_selfplay and target_network is not None:
                update_freq = int(cfg.target_update_freq)
                if update_freq > 0 and it % update_freq == 0:
                    target_network.soft_update(agent.model)

            # Single sync point: accumulate all counts and sync once
            total_rollouts = selfplay_rollouts
            total_episodes = selfplay_episodes
            if random_batch is not None:
                total_rollouts = total_rollouts + random_rollouts
                total_episodes = total_episodes + random_episodes
                log_payload.update(
                    {"train_vs_random/reward": (random_sum_rewards / random_rollouts).item()}
                )
            if league_batch is not None:
                total_rollouts = total_rollouts + league_rollouts
                total_episodes = total_episodes + league_episodes
            if archive_batch is not None:
                total_rollouts = total_rollouts + archive_rollouts
                total_episodes = total_episodes + archive_episodes

            sync_payload = {
                "num_valid_samples": num_valid_samples,
                "valid_percentage": valid_percentage,
                "total_rollouts": total_rollouts,
                "total_episodes": total_episodes,
            }
            with timer("transfer_metrics"):
                host_payload = jax.device_get(sync_payload)

            num_valid_samples_host = int(host_payload["num_valid_samples"])
            valid_percentage_host = float(host_payload["valid_percentage"])
            total_rollouts_host = int(host_payload["total_rollouts"])
            total_episodes_host = int(host_payload["total_episodes"])
            num_rollouts += total_rollouts_host
            num_episodes += total_episodes_host

            log_payload.update(
                {
                    "iteration/num_valid_samples": num_valid_samples_host,
                    "iteration/valid_percentage": valid_percentage_host,
                    "iteration/rollouts": total_rollouts_host,
                    "iteration/episodes": total_episodes_host,
                }
            )

            league.update_scores_from_collection(
                league_num_rollouts=league_num_rollouts,
                league_sum_rewards=league_sum_rewards,
                archive_num_rollouts=archive_num_rollouts,
                archive_sum_rewards=archive_sum_rewards,
            )

            if any(e.should_eval(it) for e in evaluators):
                agent.eval()
                for eval_idx, evaluator in enumerate(evaluators):
                    if evaluator.should_eval(it):
                        eval_key = jax.random.fold_in(key, eval_idx)
                        with timer(evaluator.name):
                            eval_output = evaluator.eval(eval_key)
                        log_evaluation_output(eval_output, log_payload, experiment_tracker, it)

            log_payload.update(
                {
                    "iter": it,
                    **metrics,
                    "num_rollouts": num_rollouts,
                    "num_episodes": num_episodes,
                    "iteration/learning_rate": float(
                        lr_schedule(agent.trainer.optimizer.opt_state.gradient_step[...])
                    ),
                    "iteration/num_samples": num_samples,
                    "iteration/league_archive_size": len(league.archive),
                    "iteration/active_archive_size": league.num_active_archive(),
                }
            )
            if should_dump_debug and it % cfg.dump_debug_every == 0:
                with timer("dump_debug_file"):
                    dump_debug_file(
                        training_data, it, output_dir=str(output_dir / "debug_training_data")
                    )

            if cfg.league_log_every > 0 and it % cfg.league_log_every == 0:
                with timer("log_league_standings"):
                    experiment_tracker.log_table(
                        "league/standings",
                        league.entries_to_pandas(),
                        step=it,
                    )
                    experiment_tracker.log_table(
                        "league/archive",
                        league.archive_to_pandas(),
                        step=it,
                    )

            # Periodic checkpointing
            ckpt_path: str | None = None
            if cfg.save_every > 0 and it % cfg.save_every == 0:
                ckpt_path = checkpoint_manager.save(
                    agent.model,
                    agent.trainer.optimizer,
                    key,
                    league,
                    it,
                    num_rollouts,
                    num_episodes,
                    experiment_tracker.run_id,
                )
                logger.info(f"Saved checkpoint: {ckpt_path}")
                checkpoint_manager.prune_old_checkpoints(int(cfg.keep_last_k))

                if (
                    best_checkpoint_selection.enabled
                    and best_checkpoint_selection.metric is not None
                ):
                    metric_obj = log_payload.get(best_checkpoint_selection.metric)
                    if metric_obj is None:
                        logger.debug(
                            "Skipping best checkpoint save: metric "
                            f"'{best_checkpoint_selection.metric}' missing at iter={it}."
                        )
                    else:
                        metric_value = float(metric_obj)
                        score = (
                            metric_value
                            if best_checkpoint_selection.mode == "max"
                            else -metric_value
                        )
                        best_checkpoint_manager.maybe_save(ckpt_path, it, score)

            # Milestone checkpoints (never pruned)
            milestone_every = int(cfg.milestone_every)
            if milestone_every > 0 and it % milestone_every == 0 and it > 0:
                checkpoint_manager.save_milestone(
                    agent.model,
                    agent.trainer.optimizer,
                    key,
                    league,
                    it,
                    num_rollouts,
                    num_episodes,
                    experiment_tracker.run_id,
                )

            # Periodically freeze main agent into the league for diversity.
            # After warmup: fill league quickly, then slow down additions.
            # league_fill_every: interval while league has open slots
            # league_add_every: interval once league is full (prune + replace)
            league_full = league.size() >= int(cfg.league_max_size)
            add_every = (
                int(cfg.league_add_every)
                if league_full
                else int(cfg.get("league_fill_every", cfg.league_add_every))
            )
            if (
                add_every > 0
                and not is_random_warmup
                and (league.size() == 0 or it % add_every == 0)
            ):
                league.add_from_model(agent.model, name=f"agent_{it}")

            timer.stop("total")
            timings_stats = timer.get_stats()

            t_total = timings_stats.get("total", {}).get("mean", 0)
            if t_total > 0:
                log_payload["iteration/sps"] = int(num_samples / t_total)

            log_payload.update(
                {f"timings/{name}": data["mean"] for name, data in timings_stats.items()}
            )

            progress.update(it, metrics, log_payload, timer, league=league)
            experiment_tracker.log(log_payload, it)

            if json_logger is not None:
                json_logger.log(it, log_payload, metrics, timer)

        progress.stop()
        if json_logger is not None:
            json_logger.close()
    except KeyboardInterrupt:
        progress.stop()
        if json_logger is not None:
            json_logger.close()
        logger.warning("Training interrupted. Saving emergency checkpoint...")
        ckpt_path = checkpoint_manager.save(
            agent.model,
            agent.trainer.optimizer,
            key,
            league,
            it,
            num_rollouts,
            num_episodes,
            experiment_tracker.run_id,
        )
        logger.info(f"Saved emergency checkpoint: {ckpt_path}")
        raise


@hydra.main(version_base=None, config_path="config", config_name="train_selfplay.yaml")
def main(cfg: DictConfig):
    setup_loguru_logging(cfg)
    mesh = create_device_mesh()

    env, init, step_fn, no_auto_reset_step_fn = make_env_fns(cfg)
    if cfg.resume_from is not None:
        checkpoint_path = str(cfg.resume_from)
    else:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        checkpoint_path = str(output_dir / "checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_manager = CheckpointManager(checkpoint_path)
    best_checkpoint_manager = BestCheckpointManager(
        path=Path(checkpoint_path).parent / "best_checkpoints",
        keep_top_k=int(cfg.best_checkpoint_top_k),
    )
    checkpoint = checkpoint_manager.resume_or_start(cfg, env)
    tracker: Logger = instantiate(cfg.logger, run_id=checkpoint.run_id)
    tracker.log_config(cfg)

    try:
        train(
            cfg,
            env,
            init,
            step_fn,
            no_auto_reset_step_fn,
            tracker,
            checkpoint,
            checkpoint_manager,
            best_checkpoint_manager,
            mesh,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted")
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Training failed: {e}")
    finally:
        logger.info("Closing logger...")
        tracker.close()


if __name__ == "__main__":
    out = main()
