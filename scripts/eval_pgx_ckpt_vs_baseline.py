"""
Evaluate a PGX AlphaZero .ckpt checkpoint vs PGX baseline.

This script uses the same action path as Jaxpot policy evaluation:
PGX checkpoint wrapper -> PPOAgent.rollout_actor (PolicyActor) ->
evaluate_vs_opponent_jited.

Usage:
  .venv/bin/python eval_pgx_ckpt_vs_baseline.py \
    --ckpt /home/mprzymus/projects/pgx/examples/alphazero/checkpoints/go_9x9_20260326004029/000400.ckpt \
    --num-envs 256 --num-steps 256
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import haiku as hk
import jax
import pgx
from flax import nnx
from pgx.experimental import auto_reset  # type: ignore
from pgx._src.baseline import _create_az_model_v0  # type: ignore

from jaxpot.agents import PPOAgent, PolicyActor, AlphaZeroAgent
from jaxpot.alphazero.mcts import MCTSConfig
from jaxpot.evaluate import evaluate_vs_opponent_jited
from jaxpot.evaluator.utils import bayesian_elo, calculate_elo
from jaxpot.models.pgx_baseline import PGXBaselineModel
from jaxpot.models.utils import BaseModel, ModelOutput

class _PGXConfigShim:
    """Compatibility shim for PGX checkpoints pickled with __main__.Config."""


class _PGXCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name == "Config":
            return _PGXConfigShim
        return super().find_class(module, name)


def _load_pgx_checkpoint(ckpt_path: Path):
    with ckpt_path.open("rb") as f:
        payload = _PGXCompatUnpickler(f).load()
    return payload


class PGXCheckpointModel(BaseModel):
    """PGX .ckpt model wrapped as an nnx-compatible BaseModel."""

    def __init__(self, payload: dict, action_dim: int):
        super().__init__(obs_shape=(), action_dim=action_dim)
        cfg = payload.get("config")
        model_params, model_state = payload["model"]
        self.model_params = nnx.data(model_params)
        self.model_state = nnx.data(model_state)
        self.num_channels = nnx.data(int(getattr(cfg, "num_channels", 128)))
        self.num_layers = nnx.data(int(getattr(cfg, "num_layers", 6)))
        self.resnet_v2 = nnx.data(bool(getattr(cfg, "resnet_v2", True)))

        def forward_fn(x):
            net = _create_az_model_v0(
                num_actions=action_dim,
                num_channels=self.num_channels,
                num_layers=self.num_layers,
                resnet_v2=self.resnet_v2,
            )
            return net(x, is_training=False, test_local_stats=False)

        self.forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    def __call__(self, x) -> ModelOutput:
        (logits, value), _ = self.forward.apply(self.model_params, self.model_state, x)
        # Keep value shape [B, 1] for AlphaZero/MCTS compatibility.
        if value.ndim == 1:
            value = value[:, None]
        return ModelOutput(value=value, policy_logits=logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to PGX .ckpt file")
    parser.add_argument("--env-id", default="go_9x9")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--agent-mode", choices=("ppo", "alphazero"), default="ppo")
    parser.add_argument("--num-simulations", type=int, default=32)
    parser.add_argument("--max-num-considered-actions", type=int, default=16)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.03)
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--discount-gamma", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if ckpt_path.suffix != ".ckpt":
        raise ValueError(f"Expected .ckpt file, got: {ckpt_path}")

    env = pgx.make(args.env_id)
    init = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    no_auto_reset_step_fn = jax.jit(jax.vmap(env.step))

    payload = _load_pgx_checkpoint(ckpt_path)
    model = PGXCheckpointModel(payload, int(env.num_actions))
    if args.agent_mode == "ppo":
        rollout_actor = PPOAgent(model=model, trainer=None).rollout_actor.setup(step_fn=step_fn)
    else:
        az_agent = AlphaZeroAgent(
            model=model,
            trainer=None,
            mcts_config=MCTSConfig(
                num_simulations=args.num_simulations,
                max_num_considered_actions=args.max_num_considered_actions,
                gamma=args.discount_gamma,
            ),
        )
        rollout_actor = az_agent.rollout_actor.setup(
            step_fn=step_fn, no_auto_reset_step_fn=no_auto_reset_step_fn
        )
    opponent_agent = PolicyActor(model=PGXBaselineModel(f"{args.env_id}_v0", is_eval=True))

    key = jax.random.PRNGKey(args.seed)
    key, k0, k1 = jax.random.split(key, 3)
    r0 = evaluate_vs_opponent_jited(
        rollout_actor,
        opponent_agent,
        k0,
        init,
        step_fn,
        args.num_envs,
        args.num_steps,
        model_seat=0,
        deterministic=args.deterministic,
    )
    r1 = evaluate_vs_opponent_jited(
        rollout_actor,
        opponent_agent,
        k1,
        init,
        step_fn,
        args.num_envs,
        args.num_steps,
        model_seat=1,
        deterministic=args.deterministic,
    )

    avg_reward = (r0["avg_reward"] + r1["avg_reward"]) / 2.0
    win_rate = (r0["win_rate"] + r1["win_rate"]) / 2.0
    lose_rate = (r0["lose_rate"] + r1["lose_rate"]) / 2.0
    draw_rate = (r0["draw_rate"] + r1["draw_rate"]) / 2.0
    elo = calculate_elo(win_rate, draw_rate)

    n0 = max(int(r0["num_games"]), 0)
    n1 = max(int(r1["num_games"]), 0)
    bayes_elo, bayes_elo_std = bayesian_elo(
        n0 * float(r0["win_rate"]),
        n0 * float(r0["lose_rate"]),
        n0 * float(r0["draw_rate"]),
        n1 * float(r1["win_rate"]),
        n1 * float(r1["lose_rate"]),
        n1 * float(r1["draw_rate"]),
    )

    print(f"Checkpoint: {ckpt_path}")
    print(
        f"Env: {args.env_id}, num_envs={args.num_envs}, num_steps={args.num_steps}, "
        f"agent_mode={args.agent_mode}, deterministic={args.deterministic}"
    )
    print(
        "Seat0: "
        f"reward={float(r0['avg_reward']):+.4f} "
        f"W/L/D={float(r0['win_rate']):.4f}/{float(r0['lose_rate']):.4f}/{float(r0['draw_rate']):.4f} "
        f"terminals={r0['num_games']}"
    )
    print(
        "Seat1: "
        f"reward={float(r1['avg_reward']):+.4f} "
        f"W/L/D={float(r1['win_rate']):.4f}/{float(r1['lose_rate']):.4f}/{float(r1['draw_rate']):.4f} "
        f"terminals={r1['num_games']}"
    )
    print(
        "Average: "
        f"reward={float(avg_reward):+.4f} "
        f"W/L/D={float(win_rate):.4f}/{float(lose_rate):.4f}/{float(draw_rate):.4f} "
        f"elo={float(elo):+.2f} bayesian_elo={float(bayes_elo):+.2f}±{float(bayes_elo_std):.2f}"
    )


if __name__ == "__main__":
    main()
