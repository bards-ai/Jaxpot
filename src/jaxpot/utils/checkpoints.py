import json
import pickle
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import jax
import optax
import orbax.checkpoint as ocp
from flax import nnx
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from pgx import Env

from jaxpot.league import LeagueManager
from jaxpot.models.base import PolicyValueModel


def create_lr_schedule(cfg: DictConfig) -> optax.Schedule:
    """Create a learning rate schedule based on config.

    Supports:
    - "constant": Fixed learning rate
    - "linear_decay": Linear decay from lr to lr_end
    - "cosine": Cosine annealing from lr to lr_end

    All schedules support warmup via lr_warmup_iters.
    """
    base_lr = float(cfg.lr)
    lr_end = float(getattr(cfg, "lr_end", base_lr * 0.01))
    warmup_iters = int(getattr(cfg, "lr_warmup_iters", 0))
    total_iters = int(cfg.total_iters)
    schedule_type = str(getattr(cfg, "lr_schedule", "constant"))

    # Estimate steps per iteration based on expected data collection
    # This is approximate - actual steps depend on batch size and data collected
    num_epochs = int(getattr(cfg.trainer, "num_epochs", 2))
    batch_size = int(getattr(cfg.trainer, "batch_size", 1024))

    # Rough estimate of samples per iteration
    selfplay_envs = int(getattr(cfg, "selfplay_num_envs", 128))
    num_steps = int(getattr(cfg, "num_steps", 128))
    samples_per_iter = selfplay_envs * num_steps * 2  # *2 for both players in selfplay
    steps_per_iter = max(1, (samples_per_iter * num_epochs) // batch_size)

    total_steps = total_iters * steps_per_iter
    warmup_steps = min(
        warmup_iters * steps_per_iter, total_steps // 2
    )  # Cap warmup at 50% of total
    decay_steps = max(1, total_steps - warmup_steps)  # Ensure at least 1 decay step

    if schedule_type == "constant":
        base_schedule = optax.constant_schedule(base_lr)
    elif schedule_type == "linear_decay":
        base_schedule = optax.linear_schedule(
            init_value=base_lr,
            end_value=lr_end,
            transition_steps=decay_steps,
        )
    elif schedule_type == "cosine":
        base_schedule = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=decay_steps,
            alpha=lr_end / base_lr if base_lr > 0 else 0.0,
        )
    else:
        logger.warning(f"Unknown lr_schedule '{schedule_type}', using constant")
        base_schedule = optax.constant_schedule(base_lr)

    if warmup_steps > 0:
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=base_lr,
                    transition_steps=warmup_steps,
                ),
                base_schedule,
            ],
            boundaries=[warmup_steps],
        )

    return base_schedule


@dataclass
class BestCheckpointEntry:
    iteration: int
    score: float
    checkpoint_dir: str


@dataclass
class Checkpoint:
    model: PolicyValueModel
    optimizer: nnx.Optimizer
    key: jax.Array
    league: LeagueManager
    iteration: int
    num_rollouts: int
    num_episodes: int
    run_id: str | None


class CheckpointManager:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

    @staticmethod
    def run_dir_name(path: str | Path, prefix: str = "nlhe") -> str:
        path = Path(path)
        ts = time.strftime("%Y%m%d%H%M%S")
        return str(path / f"{prefix}_{ts}")

    @staticmethod
    def format_ckpt_name(iteration: int) -> str:
        """Return directory name like 000025 for a given iteration."""
        return f"{int(iteration):06d}"

    @staticmethod
    def is_checkpoint(path: str | Path) -> bool:
        path = Path(path)
        return (path / "metadata.json").exists() or (path / "state").exists()

    @staticmethod
    def find_latest_checkpoint_dir(path: str | Path) -> str | None:
        """
        Find the lexicographically-last checkpoint dir in a path.

        Parameters
        ----------
        path : str | Path
            Path to search.

        Returns
        -------
        str | None
            Path to latest checkpoint dir or None if not found.
        """
        p = Path(path).resolve()
        if not p.exists():
            return None
        # Look for numeric-named subdirectories
        dirs = [d for d in p.iterdir() if d.is_dir() and d.name.isdigit()]
        if not dirs:
            return None
        candidates = sorted(dirs, key=lambda d: d.name)
        return str(candidates[-1].resolve())

    def resume_or_start(self, cfg: DictConfig, env: Env) -> Checkpoint:
        if self.is_checkpoint(self.path):
            checkpoint = self.resume(self.path, cfg, env)
            self.path = self.path.parent.resolve()
            return checkpoint

        latest_checkpoint_dir = self.find_latest_checkpoint_dir(self.path)
        if latest_checkpoint_dir is not None:
            return self.resume(latest_checkpoint_dir, cfg, env)
        else:
            return self.start(cfg, env)

    def start(self, cfg: DictConfig, env: Env) -> Checkpoint:
        key = jax.random.key(cfg.seed)
        model = instantiate(cfg.model, _partial_=True)(
            rngs=nnx.Rngs(key),
            obs_shape=env.observation_shape,
            action_dim=env.num_actions,
        )

        # Create learning rate schedule
        lr_schedule = create_lr_schedule(cfg)
        base_tx = optax.chain(optax.adamw(lr_schedule, b1=0.9, b2=0.999))
        tx = optax.MultiSteps(base_tx, every_k_schedule=cfg.grad_accum_steps)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        league = LeagueManager(
            cfg=cfg,
            max_size=cfg.league_max_size,
            temp=cfg.league_temp,
            alpha=cfg.league_score_alpha,
            archive_max_size=cfg.archive_max_size,
        )

        return Checkpoint(
            model=model,
            optimizer=optimizer,
            key=key,
            league=league,
            iteration=0,
            num_rollouts=0,
            num_episodes=0,
            run_id=None,
        )

    def resume(self, path: str | Path, cfg: DictConfig, env: Env) -> Checkpoint:
        ckpt_dir = Path(path).resolve()
        metadata_path = ckpt_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        with open(metadata_path) as f:
            metadata = json.load(f)
            iteration = int(metadata.get("iter", 0)) + 1
            num_rollouts = int(metadata.get("num_rollouts", 0))
            num_episodes = int(metadata.get("num_episodes", 0))
            run_id = metadata.get("run_id", None)

        league_path = ckpt_dir / "league.pkl"
        with open(league_path, "rb") as f:
            league_dict = pickle.load(f)

        league = LeagueManager(
            cfg=cfg,
            max_size=cfg.league_max_size,
            temp=cfg.league_temp,
            alpha=cfg.league_score_alpha,
            archive_max_size=cfg.archive_max_size,
        )
        league.from_dict(league_dict, env)

        obs_shape = env.observation_shape
        raw_actions = env.num_actions
        if isinstance(raw_actions, tuple):
            action_dim = tuple(int(d) for d in raw_actions)
        else:
            action_dim = (int(raw_actions),)

        abstract_model = nnx.eval_shape(
            lambda: instantiate(cfg.model, _partial_=True)(
                rngs=nnx.Rngs(0),
                obs_shape=obs_shape,
                action_dim=action_dim,
            )
        )
        graphdef, abstract_state = nnx.split(abstract_model)

        # Create abstract optimizer with the same structure (using LR schedule)
        lr_schedule = create_lr_schedule(cfg)
        base_tx = optax.chain(optax.adamw(lr_schedule, b1=0.9, b2=0.999))
        tx = optax.MultiSteps(base_tx, every_k_schedule=cfg.grad_accum_steps)

        abstract_opt = nnx.eval_shape(lambda: nnx.Optimizer(abstract_model, tx, wrt=nnx.Param))
        opt_graphdef, abstract_opt_state = nnx.split(abstract_opt)

        # Load state from orbax checkpoint (fallback_sharding handles cross-device restore)
        state_path = ckpt_dir / "state"
        fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        payload = self.checkpointer.restore(
            state_path,
            args=ocp.args.StandardRestore(fallback_sharding=fallback_sharding),
        )

        # Restore model
        nnx.replace_by_pure_dict(abstract_state, payload["model"])
        model = nnx.merge(graphdef, abstract_state)

        nnx.replace_by_pure_dict(abstract_opt_state, payload["optimizer"])
        optimizer = nnx.merge(opt_graphdef, abstract_opt_state)
        key = payload["key"]

        return Checkpoint(
            model=model,
            optimizer=optimizer,
            key=key,
            league=league,
            iteration=iteration,
            num_rollouts=num_rollouts,
            num_episodes=num_episodes,
            run_id=run_id,
        )

    def _save_to(
        self,
        checkpoint_path: Path,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        key: jax.Array,
        league: LeagueManager,
        iteration: int,
        num_rollouts: int,
        num_episodes: int,
        run_id: str,
    ) -> str:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        metadata = {
            "iter": int(iteration),
            "num_rollouts": num_rollouts,
            "num_episodes": num_episodes,
            "has_league": league is not None,
            "run_id": run_id,
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        with open(checkpoint_path / "league.pkl", "wb") as f:
            pickle.dump(league.to_dict(), f)

        model_state_pure = nnx.to_pure_dict(nnx.state(model))
        opt_state_pure = nnx.to_pure_dict(nnx.state(optimizer))
        payload = {
            "model": model_state_pure,
            "optimizer": opt_state_pure,
            "key": key,
        }
        self.checkpointer.save(
            checkpoint_path / "state",
            args=ocp.args.StandardSave(payload),
        )
        self.checkpointer.wait_until_finished()

        return str(checkpoint_path)

    def save(
        self, model, optimizer, key, league, iteration, num_rollouts, num_episodes, run_id
    ) -> str:
        path = self.path / self.format_ckpt_name(iteration)
        return self._save_to(
            path, model, optimizer, key, league, iteration, num_rollouts, num_episodes, run_id
        )

    def save_milestone(
        self, model, optimizer, key, league, iteration, num_rollouts, num_episodes, run_id
    ) -> str:
        """Save a checkpoint to the milestones directory (never pruned)."""
        milestones_dir = self.path / "milestones"
        milestones_dir.mkdir(parents=True, exist_ok=True)
        path = milestones_dir / self.format_ckpt_name(iteration)
        result = self._save_to(
            path, model, optimizer, key, league, iteration, num_rollouts, num_episodes, run_id
        )
        logger.info(f"Saved milestone checkpoint: {path}")
        return result

    def prune_old_checkpoints(self, keep_last_k: int) -> None:
        try:
            if int(keep_last_k) <= 0:
                return
            p = Path(self.path).resolve()
            if not p.exists():
                return
            step_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.isdigit()]
            if len(step_dirs) <= keep_last_k:
                return
            step_dirs_sorted = sorted(step_dirs, key=lambda d: d.name)
            to_delete = step_dirs_sorted[: max(0, len(step_dirs_sorted) - int(keep_last_k))]
            for d in to_delete:
                shutil.rmtree(d)
        except Exception:
            traceback.print_exc()


class BestCheckpointManager:
    """
    Manages top-k best checkpoints according to evaluation scores.

    Checkpoints are stored in a separate directory and tracked via a JSON manifest.
    Only the top k checkpoints by score are retained.
    """

    MANIFEST_FILE = "best_checkpoints.json"

    def __init__(self, path: str | Path, keep_top_k: int = 5):
        """
        Parameters
        ----------
        path : str | Path
            Directory where best checkpoints will be stored.
        keep_top_k : int
            Number of top checkpoints to retain.
        """
        self.path = Path(path).resolve()
        self.keep_top_k = keep_top_k
        self.path.mkdir(parents=True, exist_ok=True)
        self._entries: list[BestCheckpointEntry] = self._load_manifest()

    def _manifest_path(self) -> Path:
        return self.path / self.MANIFEST_FILE

    def _load_manifest(self) -> list[BestCheckpointEntry]:
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return []
        try:
            with open(manifest_path) as f:
                data = json.load(f)
            return [
                BestCheckpointEntry(
                    iteration=e["iteration"],
                    score=e["score"],
                    checkpoint_dir=e["checkpoint_dir"],
                )
                for e in data
            ]
        except Exception:
            traceback.print_exc()
            return []

    def _save_manifest(self) -> None:
        manifest_path = self._manifest_path()
        data = [
            {
                "iteration": e.iteration,
                "score": e.score,
                "checkpoint_dir": e.checkpoint_dir,
            }
            for e in self._entries
        ]
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def _is_better_than_worst(self, score: float) -> bool:
        if len(self._entries) < self.keep_top_k:
            return True
        worst_score = min(e.score for e in self._entries)
        return score > worst_score

    def maybe_save(
        self,
        source_checkpoint_path: str | Path,
        iteration: int,
        score: float,
    ) -> str | None:
        """
        Save checkpoint if score qualifies for top-k.

        Parameters
        ----------
        source_checkpoint_path : str | Path
            Path to the source checkpoint directory to copy.
        iteration : int
            Training iteration number.
        score : float
            Evaluation score.

        Returns
        -------
        str | None
            Path to saved best checkpoint, or None if not saved.
        """
        if not self._is_better_than_worst(score):
            logger.debug(
                f"Score {score:.4f} at iter {iteration} not in top {self.keep_top_k}, skipping"
            )
            return None

        source_path = Path(source_checkpoint_path).resolve()
        if not source_path.exists():
            logger.warning(f"Source checkpoint {source_path} does not exist")
            return None

        dest_name = f"{iteration:06d}_score_{score:.4f}"
        dest_path = self.path / dest_name

        try:
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)

            new_entry = BestCheckpointEntry(
                iteration=iteration,
                score=score,
                checkpoint_dir=str(dest_path),
            )
            self._entries.append(new_entry)
            self._prune_worst()
            self._save_manifest()

            logger.info(
                f"Saved best checkpoint: iter={iteration}, score={score:.4f}, path={dest_path}"
            )
            return str(dest_path)

        except Exception:
            traceback.print_exc()
            return None

    def _prune_worst(self) -> None:
        if len(self._entries) <= self.keep_top_k:
            return

        self._entries.sort(key=lambda e: e.score, reverse=True)
        to_remove = self._entries[self.keep_top_k :]
        self._entries = self._entries[: self.keep_top_k]

        for entry in to_remove:
            try:
                ckpt_path = Path(entry.checkpoint_dir)
                if ckpt_path.exists():
                    shutil.rmtree(ckpt_path)
                    logger.debug(
                        f"Removed checkpoint: iter={entry.iteration}, score={entry.score:.4f}"
                    )
            except Exception:
                traceback.print_exc()

    def get_best(self) -> BestCheckpointEntry | None:
        """Return the entry with the highest score."""
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.score)

    def get_all_entries(self) -> list[BestCheckpointEntry]:
        """Return all tracked entries sorted by score (descending)."""
        return sorted(self._entries, key=lambda e: e.score, reverse=True)
