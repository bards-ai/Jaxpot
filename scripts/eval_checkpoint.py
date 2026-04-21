#!/usr/bin/env python3
"""Evaluate a Connect4 checkpoint against a random opponent.

Usage:
    python scripts/eval_checkpoint.py /path/to/checkpoint_dir

The checkpoint_dir should contain metadata.json and a state/ subdirectory
(orbax checkpoint format). For example:
    python scripts/eval_checkpoint.py outputs/connect4/checkpoints/000100

The script loads the Connect4ResNet model from the checkpoint, then runs
evaluate_vs_random with both deterministic and stochastic policies, printing
detailed results for each seat and mode.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from pgx.connect_four import ConnectFour
from pgx.experimental import auto_reset

# Ensure the project source is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jaxpot.models.connect4_resnet import Connect4ResNet

from jaxpot.evaluator.evaluate import evaluate_vs_random_jited


def load_model_from_checkpoint(
    ckpt_dir: str | Path,
    *,
    action_dim: int = 7,
    num_channels: int = 128,
    num_blocks: int = 10,
    policy_head_channels: int = 32,
    value_head_channels: int = 32,
    value_hidden_dim: int = 256,
) -> Connect4ResNet:
    """Load a Connect4ResNet model from an orbax checkpoint directory.

    Parameters
    ----------
    ckpt_dir
        Path to the checkpoint directory (must contain state/ and metadata.json).
    action_dim, num_channels, num_blocks, policy_head_channels,
    value_head_channels, value_hidden_dim
        Model hyper-parameters (must match the architecture used during training).

    Returns
    -------
    Connect4ResNet
        The restored model ready for inference.
    """
    ckpt_dir = Path(ckpt_dir).resolve()
    state_path = ckpt_dir / "state"

    if not state_path.exists():
        raise FileNotFoundError(f"Orbax state directory not found at {state_path}")

    # Print checkpoint metadata if available
    metadata_path = ckpt_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"Checkpoint metadata: {json.dumps(metadata, indent=2)}")
    else:
        print("Warning: No metadata.json found in checkpoint directory.")

    # Create an abstract model (no actual arrays allocated) to get the graph structure
    abstract_model = nnx.eval_shape(
        lambda: Connect4ResNet(
            rngs=nnx.Rngs(0),
            action_dim=action_dim,
            num_channels=num_channels,
            num_blocks=num_blocks,
            policy_head_channels=policy_head_channels,
            value_head_channels=value_head_channels,
            value_hidden_dim=value_hidden_dim,
        )
    )
    graphdef, abstract_state = nnx.split(abstract_model)

    # Restore weights from the orbax checkpoint (match training's AsyncCheckpointer)
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    payload = checkpointer.restore(
        state_path,
        args=ocp.args.StandardRestore(fallback_sharding=fallback_sharding),
    )

    # Replace the abstract state with the loaded weights and merge
    nnx.replace_by_pure_dict(abstract_state, payload["model"])
    model = nnx.merge(graphdef, abstract_state)

    return model


def _print_mover_stats(results: dict, label: str) -> None:
    """Print first/second mover breakdown."""
    mover = results.get(label)
    if mover is None:
        return
    win_rate = float(mover["win_rate"])
    lose_rate = float(mover["lose_rate"])
    draw_rate = float(mover["draw_rate"])
    term_count = int(mover["num_games"])
    print(f"      Win  : {win_rate:.4f}  ({win_rate * 100:.1f}%)")
    print(f"      Lose : {lose_rate:.4f}  ({lose_rate * 100:.1f}%)")
    print(f"      Draw : {draw_rate:.4f}  ({draw_rate * 100:.1f}%)")
    print(f"      Games: {term_count}")


def run_evaluation(
    model: Connect4ResNet,
    *,
    num_envs: int = 2048,
    num_steps: int = 128,
    seed: int = 42,
) -> None:
    """Run evaluate_vs_random with first/second mover breakdown."""

    env = ConnectFour()
    init = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(auto_reset(env.step, env.init)))

    key = jax.random.key(seed)

    modes = [
        ("Stochastic (sampling)", False),
        ("Deterministic (greedy)", True),
    ]

    print("=" * 70)
    print(f"Connect4 Evaluation vs Random Opponent")
    print(f"  num_envs  = {num_envs}")
    print(f"  num_steps = {num_steps}")
    print(f"  seed      = {seed}")
    print("=" * 70)

    for mode_name, deterministic in modes:
        print(f"\n{'─' * 70}")
        print(f"  Mode: {mode_name}")
        print(f"{'─' * 70}")

        key, eval_key = jax.random.split(key)

        t0 = time.perf_counter()
        results = evaluate_vs_random_jited(
            model,
            eval_key,
            init,
            step_fn,
            num_envs=num_envs,
            num_steps=num_steps,
            model_seat=0,
            deterministic=deterministic,
        )
        jax.block_until_ready(results)
        elapsed = time.perf_counter() - t0

        avg_reward = float(results["avg_reward"])
        win_rate = float(results["win_rate"])
        lose_rate = float(results["lose_rate"])
        draw_rate = float(results["draw_rate"])
        done_rate = float(results["done_rate"])
        term_count = int(results["num_games"])
        action_frac = results["action_frac"]

        print(f"\n  Overall:")
        print(f"    Avg reward  : {avg_reward:+.4f}")
        print(f"    Win rate    : {win_rate:.4f}  ({win_rate * 100:.1f}%)")
        print(f"    Lose rate   : {lose_rate:.4f}  ({lose_rate * 100:.1f}%)")
        print(f"    Draw rate   : {draw_rate:.4f}  ({draw_rate * 100:.1f}%)")
        print(f"    Done rate   : {done_rate:.4f}")
        print(f"    Games done  : {term_count}")
        print(f"    Eval time   : {elapsed:.2f}s")

        action_pcts = [f"{float(action_frac[i]) * 100:.1f}%" for i in range(len(action_frac))]
        print(f"    Action dist : [{', '.join(action_pcts)}]")

        print(f"\n    P0 (model moves first):")
        _print_mover_stats(results, "p0")
        print(f"    P1 (model moves second):")
        _print_mover_stats(results, "p1")

    print(f"\n{'=' * 70}")
    print("Evaluation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Connect4 checkpoint against a random opponent."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint directory (containing state/ and metadata.json).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2048,
        help="Number of parallel environments (default: 2048).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="Number of steps per environment (default: 128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    # Model architecture overrides (must match training config)
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--policy-head-channels", type=int, default=32)
    parser.add_argument("--value-head-channels", type=int, default=32)
    parser.add_argument("--value-hidden-dim", type=int, default=256)

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint_path).resolve()
    if not ckpt_path.exists():
        print(f"Error: Checkpoint path does not exist: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint from: {ckpt_path}")
    model = load_model_from_checkpoint(
        ckpt_path,
        num_channels=args.num_channels,
        num_blocks=args.num_blocks,
        policy_head_channels=args.policy_head_channels,
        value_head_channels=args.value_head_channels,
        value_hidden_dim=args.value_hidden_dim,
    )
    model.eval()
    print("Model loaded successfully.\n")

    run_evaluation(
        model,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
