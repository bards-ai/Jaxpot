import json
import pickle
import shutil
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx

from jaxpot.league import LeagueManager
from jaxpot.models.base import PolicyValueModel
from jaxpot.rollout.buffer import TrainingDataBuffer


def ensure_dir(path: str | Path) -> None:
    """
    Create directory if it does not exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def format_ckpt_name(iteration: int) -> str:
    """Return directory name like 000025 for a given iteration."""
    return f"{int(iteration):06d}"


def save_checkpoint(
    model: PolicyValueModel,
    iteration: int,
    run_dir: str,
    optimizer: nnx.Optimizer | None = None,
    key: jax.Array | None = None,
    league: LeagueManager | None = None,
    run_id: str | None = None,
    checkpointer: ocp.StandardCheckpointer | None = None,
    num_games_done: int = 0,
) -> str:
    """
    Save a training checkpoint containing the full model and iteration.

    Parameters
    ----------
    model : PolicyValueModel
        Model instance to serialize.
    iteration : int
        Current training iteration.
    run_dir : str
        Directory to store checkpoint files.
    optimizer : nnx.Optimizer | None
        Optimizer state to save.
    key : jax.Array | None
        RNG key to save.
    league : LeagueManager | None
        League state to save.
    run_id : str | None
        Run ID for resuming.
    checkpointer : ocp.StandardCheckpointer | None
        Optional checkpointer instance.

    Returns
    -------
    str
        Path to the written checkpoint file.
    """
    # Ensure absolute checkpoint destination
    run_dir_abs = Path(run_dir).resolve()
    ensure_dir(run_dir_abs)
    step_dir = run_dir_abs / format_ckpt_name(iteration)
    if step_dir.exists():
        try:
            if not any(step_dir.iterdir()):
                step_dir.rmdir()
            else:
                return str(step_dir)
        except Exception:
            traceback.print_exc()

    ensure_dir(step_dir)

    # Save metadata (non-JAX types) separately as JSON
    metadata: dict[str, object] = {
        "iter": int(iteration),
        "num_games": num_games_done,
        "has_league": league is not None,
    }
    if run_id is not None:
        metadata["run_id"] = run_id

    metadata_path = step_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save league state separately using pickle (contains JAX arrays)
    if league is not None:
        league_path = step_dir / "league.pkl"
        with open(league_path, "wb") as f:
            pickle.dump(league.to_dict(), f)

    # Extract pure dict states for model and optimizer (JAX-compatible only)
    model_state_pure = nnx.to_pure_dict(nnx.state(model))
    payload: dict[str, object] = {"model": model_state_pure}
    if optimizer is not None:
        opt_state_pure = nnx.to_pure_dict(nnx.state(optimizer))
        payload["optimizer"] = opt_state_pure
    if key is not None:
        payload["key"] = key

    # Use provided checkpointer if available; otherwise create a local one
    local_checkpointer = False
    if checkpointer is None:
        checkpointer = ocp.StandardCheckpointer()
        local_checkpointer = True

    checkpointer.save(step_dir / "state", payload)

    # If we created a local checkpointer for this save, block until finished
    # to avoid asynchronous errors being lost when the object is deleted.
    if local_checkpointer:
        checkpointer.wait_until_finished()
    return str(step_dir)


def find_latest_checkpoint(run_dir: str) -> str | None:
    """
    Find the lexicographically-last .ckpt in a directory.

    Parameters
    ----------
    run_dir : str
        Directory to search.

    Returns
    -------
    str | None
        Path to latest checkpoint or None if not found.
    """
    p = Path(run_dir).resolve()
    if not p.exists():
        return None
    # Look for numeric-named subdirectories
    dirs = [d for d in p.iterdir() if d.is_dir() and d.name.isdigit()]
    if not dirs:
        return None
    candidates = sorted(dirs, key=lambda d: d.name)
    return str(candidates[-1].resolve())


def prune_old_checkpoints(run_dir: str, keep_last_k: int) -> None:
    """
    Remove older checkpoint subdirectories, keeping only the latest K by name.

    Parameters
    ----------
    run_dir : str
        Directory containing numeric-named checkpoint subdirectories.
    keep_last_k : int
        Number of most recent checkpoints to keep. If <= 0, do nothing.
    """
    try:
        if int(keep_last_k) <= 0:
            return
        p = Path(run_dir).resolve()
        if not p.exists():
            return
        step_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.isdigit()]
        if len(step_dirs) <= keep_last_k:
            return
        step_dirs_sorted = sorted(step_dirs, key=lambda d: d.name)
        to_delete = step_dirs_sorted[: max(0, len(step_dirs_sorted) - int(keep_last_k))]
        for d in to_delete:
            try:
                shutil.rmtree(d)
            except Exception:
                traceback.print_exc()
    except Exception:
        traceback.print_exc()


def dump_debug_file(
    training_data_batch: TrainingDataBuffer, iteration: int, output_dir: str
) -> None:
    valid_mask = training_data_batch.valids.astype(bool)
    valid_indices = jnp.where(valid_mask)[0]
    num_valid = int(valid_indices.shape[0])
    num_samples_to_save = min(50, num_valid)

    samples_list = []
    for idx_in_valid in range(num_samples_to_save):
        i = int(valid_indices[idx_in_valid])

        # TODO handle this somehow better, maybe wrapp obs object with pprint
        obs_data = None
        if hasattr(training_data_batch.obs, "pprint"):
            obs_data = training_data_batch.obs.pprint(i).split("\n")
        else:
            obs_i = training_data_batch.obs[i]
            obs_data = {
                "shape": list(obs_i.shape),
                "dtype": str(obs_i.dtype),
                "min": float(jnp.min(obs_i)),
                "max": float(jnp.max(obs_i)),
                "mean": float(jnp.mean(obs_i)),
            }

        sample = {
            "sample_idx": idx_in_valid,
            "original_batch_idx": i,
            "obs": obs_data,
            "old_val": training_data_batch.value[i].tolist(),
            "returns": training_data_batch.returns[i].tolist(),
            "advantages": float(training_data_batch.adv[i]),
            "actions": int(training_data_batch.actions[i]),
            "log_prob": float(training_data_batch.log_prob[i]),
            "legal_action_mask": training_data_batch.legal_action_mask[i].tolist(),
            **training_data_batch.get_auxiliary_targets_for_sample(i),
        }
        samples_list.append(sample)

    samples_data = {
        "num_valid_samples_total": num_valid,
        "num_samples_saved": num_samples_to_save,
        "samples": samples_list,
    }

    save_path = Path(output_dir) / f"training_samples_first_{iteration}.json"
    ensure_dir(save_path.parent)
    with open(save_path, "w") as f:
        json.dump(samples_data, f, indent=2, ensure_ascii=False)
