"""
Evaluation script for PGX games with animated SVG visualization.

Supports two modes:
- Agent vs Random: Provide only checkpoint (default)
- Agent vs Agent: Provide both checkpoint and checkpoint2

Note: PGX randomizes which player moves first each game. Results are reported
with a breakdown by whether Agent 1 moved first or second.

Usage:
    # Agent 1 vs Random
    python eval_pgx.py experiment=chess/default checkpoint=/path/to/checkpoint
    python eval_pgx.py experiment=connect4/default checkpoint=/path num_games=10

    # Agent 1 vs Agent 2
    python eval_pgx.py experiment=chess/default checkpoint=/path1 checkpoint2=/path2
    python eval_pgx.py experiment=connect4/default checkpoint=/path1 checkpoint2=/path2 deterministic=true deterministic2=false
"""

from __future__ import annotations

from pathlib import Path

import distrax
import hydra
import jax
import jax.numpy as jnp
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig

from flax import nnx

from jaxpot.env.visualizer import save_gif, save_svg_animation
from jaxpot.models.base import PolicyValueModel
from jaxpot.utils.checkpoints import CheckpointManager


def _resolve_checkpoint_dir(path_str: str) -> Path:
    """Resolve checkpoint path from string, finding latest if directory provided."""
    p = Path(path_str).expanduser()
    p_resolved = p.resolve()

    if CheckpointManager.is_checkpoint(p_resolved):
        return p_resolved

    latest = CheckpointManager.find_latest_checkpoint_dir(p_resolved)
    if latest is None:
        raise ValueError(f"No checkpoint found in {p_resolved}")
    return Path(latest)


def _make_inference_fn(model: PolicyValueModel) -> tuple:
    """Create a JIT-compiled inference function for a model.

    Returns ``(infer_fn, model_state, init_hidden_h, init_hidden_c)``.
    The inference function accepts and returns hidden state so that recurrent
    models (e.g. ResNet-LSTM) accumulate memory across game steps.
    For non-recurrent models the hidden state is tiny zeros that flow through
    unchanged — no branching needed.
    """
    graphdef, state = nnx.split(model)
    init_hidden_h, init_hidden_c = model.init_state(1)  # [L, 1, H]

    @jax.jit
    def _infer(state, obs, legal_action_mask, key, deterministic, hidden_h, hidden_c):
        mdl = nnx.merge(graphdef, state)
        model_output = mdl(obs[None, ...], hidden_h=hidden_h, hidden_c=hidden_c)
        logits = model_output.policy_logits[0]
        masked_logits = jnp.where(legal_action_mask, logits, -1e9)
        action = jnp.where(
            deterministic,
            jnp.argmax(masked_logits),
            distrax.Categorical(logits=masked_logits).sample(seed=key),
        )
        return action, model_output.hidden_h, model_output.hidden_c

    return _infer, state, init_hidden_h, init_hidden_c


@jax.jit
def _select_random_action(legal_action_mask: jax.Array, key: jax.Array) -> jax.Array:
    logits = jnp.where(legal_action_mask, 0.0, -1e9)
    return distrax.Categorical(logits=logits).sample(seed=key)


def play_single_game(
    env_init: callable,
    env_step: callable,
    model_infer_fn: callable,
    model_state: nnx.State,
    key: jax.Array,
    max_steps: int,
    deterministic: bool = False,
    opponent_infer_fn: callable | None = None,
    opponent_state: nnx.State | None = None,
    opponent_deterministic: bool = False,
    model_seat: int = 0,
    model_init_hidden_h: jax.Array | None = None,
    model_init_hidden_c: jax.Array | None = None,
    opponent_init_hidden_h: jax.Array | None = None,
    opponent_init_hidden_c: jax.Array | None = None,
) -> tuple[list, dict, int, int, list[tuple[int, int]]]:
    """
    Play a single game, capturing all states for visualization.

    Uses pre-compiled JIT inference functions for speed.
    Hidden state is threaded through steps for recurrent models.

    Returns (states, metrics, model_seat, first_player, game_actions) where
    game_actions is a list of (current_player, action) tuples.
    """
    key, init_key = jax.random.split(key, 2)

    if model_seat not in (0, 1):
        raise ValueError(f"model_seat must be 0 or 1, got {model_seat}")

    state = env_init(init_key)
    states = [state]
    game_actions: list[tuple[int, int]] = []

    det_flag = jnp.bool_(deterministic)
    opp_det_flag = jnp.bool_(opponent_deterministic)

    first_player = int(state.current_player)
    model_moved_first = first_player == model_seat

    # Fresh hidden state per game
    model_hidden_h = model_init_hidden_h
    model_hidden_c = model_init_hidden_c
    opp_hidden_h = opponent_init_hidden_h
    opp_hidden_c = opponent_init_hidden_c

    for step_idx in range(max_steps):
        if state.terminated or state.truncated:
            break

        key, action_key = jax.random.split(key)
        current_player = int(state.current_player)

        if current_player == model_seat:
            action, model_hidden_h, model_hidden_c = model_infer_fn(
                model_state, state.observation, state.legal_action_mask,
                action_key, det_flag, model_hidden_h, model_hidden_c,
            )
        elif opponent_infer_fn is not None:
            action, opp_hidden_h, opp_hidden_c = opponent_infer_fn(
                opponent_state, state.observation, state.legal_action_mask,
                action_key, opp_det_flag, opp_hidden_h, opp_hidden_c,
            )
        else:
            action = _select_random_action(state.legal_action_mask, action_key)

        game_actions.append((current_player, int(action)))

        key, step_key = jax.random.split(key)
        state = env_step(state, action, step_key)
        states.append(state)

    reward = float(state.rewards[model_seat]) if state.terminated else 0.0
    metrics = {
        "num_steps": len(states) - 1,
        "terminated": bool(state.terminated),
        "truncated": bool(state.truncated),
        "reward": reward,
        "win": reward > 0,
        "loss": reward < 0,
        "draw": reward == 0 and bool(state.terminated),
        "model_moved_first": model_moved_first,
    }

    return states, metrics, model_seat, first_player, game_actions


@hydra.main(version_base=None, config_path="config", config_name="eval_pgx.yaml")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    if cfg.get("checkpoint") in (None, "???", ""):
        raise SystemExit("Provide checkpoint=/path/to/checkpoint")

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    svg_dir = output_dir / "animations"
    svg_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PGX Model Evaluation with Visualization")
    logger.info("=" * 60)

    logger.info(f"Creating environment: {cfg.env._target_}")
    env = instantiate(cfg.env)
    logger.info(f"Environment: {env.id}")

    logger.info(f"Loading Agent 1 checkpoint from: {cfg.checkpoint}")
    ckpt_dir = _resolve_checkpoint_dir(str(cfg.checkpoint))
    ckpt_mgr = CheckpointManager(ckpt_dir)
    checkpoint = ckpt_mgr.resume(ckpt_dir, cfg, env)
    model = checkpoint.model
    model.eval()
    model_infer_fn, model_state, model_init_h, model_init_c = _make_inference_fn(model)
    logger.info(f"Loaded Agent 1 checkpoint at iteration {checkpoint.iteration}")

    opponent_infer_fn = None
    opponent_state = None
    opp_init_h = None
    opp_init_c = None
    has_opponent_model = cfg.get("checkpoint2") not in (None, "???", "")
    if has_opponent_model:
        logger.info(f"Loading Agent 2 checkpoint from: {cfg.checkpoint2}")
        ckpt_dir2 = _resolve_checkpoint_dir(str(cfg.checkpoint2))
        ckpt_mgr2 = CheckpointManager(ckpt_dir2)
        checkpoint2 = ckpt_mgr2.resume(ckpt_dir2, cfg, env)
        opponent_model = checkpoint2.model
        opponent_model.eval()
        opponent_infer_fn, opponent_state, opp_init_h, opp_init_c = _make_inference_fn(opponent_model)
        logger.info(f"Loaded Agent 2 checkpoint at iteration {checkpoint2.iteration}")
    else:
        logger.info("No Agent 2 checkpoint provided - using random opponent")

    env_init = jax.jit(env.init)
    env_step = jax.jit(env.step)

    key = jax.random.key(int(cfg.seed))

    total_wins = 0
    total_losses = 0
    total_draws = 0
    all_rewards = []
    all_num_steps = []

    first_mover_stats = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
    second_mover_stats = {"wins": 0, "losses": 0, "draws": 0, "games": 0}

    opponent_label = "Agent 2" if has_opponent_model else "Random"
    logger.info(f"Playing {cfg.num_games} games: Agent 1 vs {opponent_label}")
    logger.info("-" * 60)

    key, seat_key = jax.random.split(key)
    model_seat = int(jax.random.bernoulli(seat_key))

    for game_idx in range(int(cfg.num_games)):
        key, game_key = jax.random.split(key)

        states, metrics, model_seat, first_player, game_actions = play_single_game(
            env_init=env_init,
            env_step=env_step,
            model_infer_fn=model_infer_fn,
            model_state=model_state,
            key=game_key,
            max_steps=int(cfg.max_steps_per_game),
            deterministic=bool(cfg.deterministic),
            opponent_infer_fn=opponent_infer_fn,
            opponent_state=opponent_state,
            opponent_deterministic=bool(cfg.get("deterministic2", cfg.deterministic)),
            model_seat=model_seat,
            model_init_hidden_h=model_init_h,
            model_init_hidden_c=model_init_c,
            opponent_init_hidden_h=opp_init_h,
            opponent_init_hidden_c=opp_init_c,
        )

        all_rewards.append(metrics["reward"])
        all_num_steps.append(metrics["num_steps"])

        if metrics["win"]:
            total_wins += 1
            result_str = "WIN"
        elif metrics["loss"]:
            total_losses += 1
            result_str = "LOSS"
        else:
            total_draws += 1
            result_str = "DRAW"

        if metrics["model_moved_first"]:
            stats = first_mover_stats
            mover_str = "1st"
        else:
            stats = second_mover_stats
            mover_str = "2nd"

        stats["games"] += 1
        if metrics["win"]:
            stats["wins"] += 1
        elif metrics["loss"]:
            stats["losses"] += 1
        else:
            stats["draws"] += 1

        # Duplicate the final state to create a pause effect
        if states and (states[-1].terminated or states[-1].truncated):
            final_state = states[-1]
            for _ in range(int(cfg.num_final_frames)):
                states.append(final_state)

        second_player = 1 - first_player
        if model_seat == first_player:
            player_labels = {
                first_player: "1st: Agent 1",
                second_player: f"2nd: {opponent_label}",
            }
        else:
            player_labels = {
                first_player: f"1st: {opponent_label}",
                second_player: "2nd: Agent 1",
            }

        svg_path = svg_dir / f"game_{game_idx:03d}_{result_str.lower()}.svg"
        gif_path = svg_dir / f"game_{game_idx:03d}_{result_str.lower()}.gif"
        viz_kwargs = dict(
            states=states,
            color_theme=cfg.color_theme,
            scale=float(cfg.scale),
            frame_duration_seconds=float(cfg.frame_duration_ms) / 1000.0,
            player_labels=player_labels,
        )
        save_svg_animation(filename=svg_path, **viz_kwargs)
        save_gif(filename=gif_path, **viz_kwargs)

        # Write text game record for Quoridor
        if env.id == "quoridor" and game_actions:
            from jaxpot.env.quoridor.notation import format_game_record

            if model_seat == 0:
                p0_lbl, p1_lbl = "Agent 1", opponent_label
            else:
                p0_lbl, p1_lbl = opponent_label, "Agent 1"
            if metrics["win"]:
                result_code = "1-0" if model_seat == 0 else "0-1"
            elif metrics["loss"]:
                result_code = "0-1" if model_seat == 0 else "1-0"
            else:
                result_code = "1/2-1/2"
            txt_path = svg_dir / f"game_{game_idx:03d}_{result_str.lower()}.txt"
            txt_path.write_text(
                format_game_record(game_actions, p0_lbl, p1_lbl, result_code, first_player)
            )

        logger.info(
            f"Game {game_idx + 1:2d}/{cfg.num_games}: "
            f"{result_str:5s} | {mover_str} mover | steps={metrics['num_steps']:3d} | "
            f"reward={metrics['reward']:+.2f} | saved to {svg_path.name}"
        )

    logger.info("-" * 60)
    logger.info(f"Evaluation Summary (Agent 1 vs {opponent_label}):")
    logger.info(f"  Total games:    {cfg.num_games}")
    logger.info(f"  Agent 1 wins:   {total_wins:3d} ({total_wins / cfg.num_games:6.1%})")
    logger.info(f"  Agent 1 losses: {total_losses:3d} ({total_losses / cfg.num_games:6.1%})")
    logger.info(f"  Draws:          {total_draws:3d} ({total_draws / cfg.num_games:6.1%})")
    logger.info(f"  Avg reward:     {sum(all_rewards) / len(all_rewards):+.4f}")
    logger.info(f"  Avg steps:      {sum(all_num_steps) / len(all_num_steps):.1f}")

    logger.info("")
    logger.info("Agent 1 breakdown by move order:")

    def _format_stats(stats: dict, label: str) -> None:
        if stats["games"] == 0:
            logger.info(f"  {label}: (no games)")
            return
        n = stats["games"]
        wins, losses, draws = stats["wins"], stats["losses"], stats["draws"]
        logger.info(
            f"  {label}: {n:3d} games | "
            f"W {wins:3d} ({wins / n:5.1%}) | "
            f"L {losses:3d} ({losses / n:5.1%}) | "
            f"D {draws:3d} ({draws / n:5.1%})"
        )

    _format_stats(first_mover_stats, "As 1st mover")
    _format_stats(second_mover_stats, "As 2nd mover")

    logger.info("-" * 60)
    logger.info(f"Animations saved to: {svg_dir}")
    logger.info("Open SVG files in browser to view animated gameplay")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
