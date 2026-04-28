"""
Evaluate a checkpointed model as PPO policy vs PGX baseline.

This script is eval-only: it does not run the training loop.
It intentionally uses PPOAgent (PolicyActor rollout) so actions come from
the model policy directly (deterministic argmax), not MCTS sampling/search.

Usage:
    python eval_pgx_policy.py resume_from=/path/to/checkpoints
    python eval_pgx_policy.py resume_from=/path/to/checkpoints num_eval_envs=512 num_steps=384
    python eval_pgx_policy.py resume_from=/path/to/000350 eval.num_envs=256 eval.num_steps=256
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import jax
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from train_selfplay import make_env_fns
from jaxpot.agents import AlphaZeroAgent, PPOAgent, PolicyActor
from jaxpot.evaluate import evaluate_vs_opponent_jited
from jaxpot.evaluator.utils import bayesian_elo, calculate_elo
from jaxpot.models.pgx_baseline import PGXBaselineModel
from jaxpot.utils.checkpoints import CheckpointManager

_AGENT_MODE = "alphazero"


def _pop_agent_mode_flag(argv: list[str]) -> str:
    mode = "alphazero"
    keep: list[str] = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--agent-mode="):
            mode = arg.split("=", 1)[1].strip().lower()
        elif arg == "--agent-mode":
            if i + 1 >= len(argv):
                raise SystemExit("--agent-mode requires a value: ppo or alphazero")
            mode = argv[i + 1].strip().lower()
            i += 1
        else:
            keep.append(arg)
        i += 1
    if mode not in {"ppo", "alphazero"}:
        raise SystemExit(f"Invalid --agent-mode={mode}. Use 'ppo' or 'alphazero'.")
    argv[:] = keep
    return mode


_AGENT_MODE = _pop_agent_mode_flag(sys.argv)


def _resolve_checkpoint_dir(resume_from: str) -> Path:
    path = Path(resume_from).expanduser().resolve()
    if CheckpointManager.is_checkpoint(path):
        return path
    latest = CheckpointManager.find_latest_checkpoint_dir(path)
    if latest is None:
        raise FileNotFoundError(
            f"No checkpoint found under resume_from={path}. "
            "Pass a checkpoint step directory (e.g. .../000350) or a parent checkpoints dir."
        )
    return Path(latest).resolve()


def _load_model(cfg: DictConfig, env):
    if not cfg.get("resume_from"):
        raise SystemExit("Missing required override: resume_from=/path/to/checkpoints_or_step_dir")

    ckpt_dir = _resolve_checkpoint_dir(str(cfg.resume_from))
    checkpoint_manager = CheckpointManager(ckpt_dir)
    checkpoint = checkpoint_manager.resume(ckpt_dir, cfg, env)
    logger.info(f"Loaded checkpoint iteration={checkpoint.iteration}")
    return checkpoint.model, checkpoint.optimizer, int(checkpoint.iteration), f"factorio_jax_ckpt:{ckpt_dir}"


def _resolve_eval_settings(cfg: DictConfig) -> tuple[str, int, int]:
    baseline_model_id = "go_9x9_v0"
    num_envs = int(cfg.get("num_eval_envs", 256))
    num_steps = int(cfg.get("num_steps", 256))

    eval_cfg = cfg.get("eval")
    if eval_cfg:
        for evaluator_cfg in eval_cfg:
            target = str(evaluator_cfg.get("_target_", ""))
            if target.endswith("PGXBaselineEvaluator"):
                baseline_model_id = str(evaluator_cfg.get("baseline_model_id", baseline_model_id))
                num_envs = int(evaluator_cfg.get("num_envs", num_envs))
                num_steps = int(evaluator_cfg.get("num_steps", num_steps))
                break
    return baseline_model_id, num_envs, num_steps


@hydra.main(version_base=None, config_path="../config", config_name="experiment/go_9x9/go_9_alphazero_config")
def main(cfg: DictConfig) -> None:
    env, init, step_fn, no_auto_reset_step_fn = make_env_fns(cfg)
    model, optimizer, start_iteration, model_source = _load_model(cfg, env)
    logger.info(f"Using model source: {model_source} (agent_mode={_AGENT_MODE})")

    repo_root = Path(__file__).resolve().parent.parent
    if _AGENT_MODE == "ppo":
        trainer_cfg = OmegaConf.load(repo_root / "config" / "trainer" / "ppo.yaml")
        trainer = instantiate(
            trainer_cfg,
            optimizer=optimizer,
            start_iteration=start_iteration,
            _convert_="object",
        )
        agent = PPOAgent(model=model, trainer=trainer)
        agent.eval()
        rollout_actor = agent.rollout_actor.setup(step_fn=step_fn)
    else:
        trainer_cfg = OmegaConf.load(repo_root / "config" / "trainer" / "alphazero.yaml")
        trainer = instantiate(
            trainer_cfg,
            optimizer=optimizer,
            start_iteration=start_iteration,
            _convert_="object",
        )
        agent = AlphaZeroAgent(model=model, trainer=trainer, mcts_config=cfg.train_agent.mcts_config)
        agent.eval()
        rollout_actor = agent.rollout_actor.setup(
            step_fn=step_fn, no_auto_reset_step_fn=no_auto_reset_step_fn
        )

    baseline_model_id, num_envs, num_steps = _resolve_eval_settings(cfg)
    opponent = PolicyActor(model=PGXBaselineModel(baseline_model_id, is_eval=True))

    key = jax.random.key(int(cfg.seed))
    key, k0, k1 = jax.random.split(key, 3)

    results_0 = evaluate_vs_opponent_jited(
        rollout_actor,
        opponent,
        k0,
        init,
        step_fn,
        num_envs,
        num_steps,
        model_seat=0,
        deterministic=True,
    )
    results_1 = evaluate_vs_opponent_jited(
        rollout_actor,
        opponent,
        k1,
        init,
        step_fn,
        num_envs,
        num_steps,
        model_seat=1,
        deterministic=True,
    )

    avg_reward = (results_0["avg_reward"] + results_1["avg_reward"]) / 2.0
    win_rate = (results_0["win_rate"] + results_1["win_rate"]) / 2.0
    lose_rate = (results_0["lose_rate"] + results_1["lose_rate"]) / 2.0
    draw_rate = (results_0["draw_rate"] + results_1["draw_rate"]) / 2.0
    elo = calculate_elo(win_rate, draw_rate)

    n0 = max(int(results_0["num_games"]), 0)
    n1 = max(int(results_1["num_games"]), 0)
    wins_p0 = n0 * float(results_0["win_rate"])
    losses_p0 = n0 * float(results_0["lose_rate"])
    draws_p0 = n0 * float(results_0["draw_rate"])
    wins_p1 = n1 * float(results_1["win_rate"])
    losses_p1 = n1 * float(results_1["lose_rate"])
    draws_p1 = n1 * float(results_1["draw_rate"])
    bayes_elo, bayes_elo_std = bayesian_elo(
        wins_p0,
        losses_p0,
        draws_p0,
        wins_p1,
        losses_p1,
        draws_p1,
    )

    logger.info("Deterministic PPO policy vs PGX baseline")
    logger.info(f"baseline_model_id={baseline_model_id}, num_envs={num_envs}, num_steps={num_steps}")
    logger.info(
        f"Seat0: reward={float(results_0['avg_reward']):+.4f} "
        f"W/L/D={float(results_0['win_rate']):.4f}/{float(results_0['lose_rate']):.4f}/{float(results_0['draw_rate']):.4f} "
        f"terminals={int(results_0['num_games'])}"
    )
    logger.info(
        f"Seat1: reward={float(results_1['avg_reward']):+.4f} "
        f"W/L/D={float(results_1['win_rate']):.4f}/{float(results_1['lose_rate']):.4f}/{float(results_1['draw_rate']):.4f} "
        f"terminals={int(results_1['num_games'])}"
    )
    logger.info(
        f"Average: reward={float(avg_reward):+.4f} "
        f"W/L/D={float(win_rate):.4f}/{float(lose_rate):.4f}/{float(draw_rate):.4f} "
        f"elo={float(elo):+.2f} bayesian_elo={float(bayes_elo):+.2f}±{float(bayes_elo_std):.2f}"
    )


if __name__ == "__main__":
    main()
