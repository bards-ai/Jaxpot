import csv
import io

import jax
import numpy as np
import pandas as pd
from flax import nnx
from hydra.utils import instantiate
from jax import numpy as jnp

from jaxpot.models.base import PolicyValueModel


class LeagueEntry:
    def __init__(
        self,
        model: PolicyValueModel,
        score_ema: float = 0.0,  # Score of main model against entry model
        rollouts: int = 0,
        name: str | None = None,
    ):
        self.model = model
        self.score_ema = float(score_ema)
        self.rollouts = int(rollouts)
        self.name = name


class LeagueArchiveEntry(LeagueEntry):
    def __init__(
        self,
        model: PolicyValueModel,
        score_ema: float = 0.0,  # Score of main model against archive entry model
        rollouts: int = 0,
        name: str | None = None,
        active: bool = False,
    ):
        LeagueEntry.__init__(self, model, score_ema, rollouts, name)
        self.active = active

    @staticmethod
    def from_entry(entry: LeagueEntry, active: bool = False) -> "LeagueArchiveEntry":
        return LeagueArchiveEntry(entry.model, entry.score_ema, entry.rollouts, entry.name, active)


class LeagueManager:
    def __init__(
        self,
        cfg,
        max_size: int,
        temp: float,
        alpha: float,
        archive_max_size: int = 64,
        forgotten_result_margin: float = 0.001,
    ):
        self.cfg = cfg
        self.max_size = int(max_size)
        self.temp = float(max(temp, 1e-6))
        self.alpha = float(alpha)
        self.entries: list[LeagueEntry] = []
        # Archive of older frozen agents kept for periodic evaluation/training
        self.archive: list[LeagueArchiveEntry] = []
        self.archive_max_size = int(max(0, archive_max_size))
        self.forgotten_result_margin = forgotten_result_margin

    def size(self) -> int:
        return len(self.entries)

    def clone_frozen_model(self, model: PolicyValueModel) -> PolicyValueModel:
        """
        Create an evaluation-mode copy of the given model with identical parameters.

        Returns
        -------
        PolicyValueModel
            Frozen model copy (eval mode).
        """
        graphdef, state = nnx.split(model)
        cloned = nnx.merge(graphdef, state)
        cloned.eval()
        return cloned

    def _restore_model_from_entry(
        self, entry_data: dict, graphdef: nnx.GraphDef, abstract_state: nnx.State
    ) -> PolicyValueModel:
        """Restore a model from serialized entry data using a template graphdef."""
        state_pure = entry_data["state"]
        nnx.replace_by_pure_dict(abstract_state, state_pure)
        model = nnx.merge(graphdef, abstract_state)
        model.eval()
        return model

    def add_from_model(
        self,
        model: PolicyValueModel,
        initial_score: float = -0.05,
        name: str | None = None,
    ) -> None:
        frozen = self.clone_frozen_model(model)
        self.entries.append(LeagueEntry(frozen, initial_score, name=name))
        self.prune()

    def prune(self) -> None:
        self.entries.sort(key=lambda e: e.score_ema)
        keep = self.entries[: self.max_size]
        to_archive = self.entries[self.max_size :]
        self.entries = keep

        for e in to_archive:
            archive_entry = LeagueArchiveEntry.from_entry(e)
            self.archive.append(archive_entry)
        self._archive_prune()

    def _archive_prune(self) -> None:
        if self.archive_max_size <= 0:
            self.archive = []
            return
        if len(self.archive) <= self.archive_max_size:
            return

        self.archive.sort(key=lambda e: e.score_ema)
        excess = len(self.archive) - self.archive_max_size
        # First, remove inactive entries (highest scoring = worst models first)
        inactive_indices = [i for i, e in enumerate(self.archive) if not e.active]
        removed = 0
        for idx in inactive_indices[::-1]:
            if removed >= excess:
                break
            del self.archive[idx]
            removed += 1

        # If still need to remove more, drop highest-scoring archived agents
        remaining_excess = excess - removed
        if remaining_excess > 0:
            del self.archive[:remaining_excess]

    def update_score(self, idx: int, avg_reward: float, rollouts: int) -> None:
        entry = self.entries[idx]
        # Track bb/hand-like performance by EMA of per-episode average reward
        a = self.alpha
        entry.score_ema = (1.0 - a) * entry.score_ema + a * float(avg_reward)
        entry.rollouts += int(rollouts)

    def update_archive_score(self, idx: int, avg_reward: float, rollouts: int) -> None:
        entry = self.archive[idx]
        a = self.alpha
        entry.score_ema = (1.0 - a) * entry.score_ema + a * float(avg_reward)
        entry.rollouts += int(rollouts)
        entry.active = entry.score_ema < self.forgotten_result_margin

    def update_scores_from_collection(
        self,
        league_num_rollouts: jax.Array | None = None,
        league_sum_rewards: jax.Array | None = None,
        archive_num_rollouts: jax.Array | None = None,
        archive_sum_rewards: jax.Array | None = None,
    ) -> None:
        """
        Update league and archive scores from collection results.

        Parameters
        ----------
        league_num_rollouts : jax.Array | None
            Array of episode counts per league entry.
        league_sum_rewards : jax.Array | None
            Array of sum rewards per league entry.
        archive_num_rollouts : jax.Array | None
            Array of episode counts per active archive entry.
        archive_sum_rewards : jax.Array | None
            Array of sum rewards per active archive entry.
        """
        if league_num_rollouts is not None and league_sum_rewards is not None:
            league_term_np = np.asarray(league_num_rollouts)
            league_sum_np = np.asarray(league_sum_rewards)
            for idx in range(len(league_term_np)):
                n_ep = int(league_term_np[idx])
                if n_ep <= 0:
                    continue
                self.update_score(idx, float(league_sum_np[idx]) / n_ep, n_ep)

        if archive_num_rollouts is not None and archive_sum_rewards is not None:
            archive_term_np = np.asarray(archive_num_rollouts)
            archive_sum_np = np.asarray(archive_sum_rewards)
            active_indices = [i for i, entry in enumerate(self.archive) if entry.active]
            for active_pos, archive_idx in enumerate(active_indices):
                n_ep = int(archive_term_np[active_pos])
                if n_ep <= 0:
                    continue
                self.update_archive_score(
                    archive_idx, float(archive_sum_np[active_pos]) / n_ep, n_ep
                )

    def get_league_models_and_weights(self) -> tuple[tuple[nnx.Module, ...], jax.Array]:
        modules = tuple([entry.model for entry in self.entries])
        scores = jnp.array([float(entry.score_ema) for entry in self.entries], dtype=jnp.float32)
        scaled = -scores / jnp.clip(self.temp, 1e-6, None)
        weights = jax.nn.softmax(scaled)

        return modules, weights

    def get_archive_models_and_weights(self) -> tuple[tuple[nnx.Module, ...], jax.Array]:
        models = []
        scores = []
        for entry in self.archive:
            if not entry.active:
                continue

            models.append(entry.model)
            scores.append(float(entry.score_ema))

        if not scores:
            return tuple(models), jnp.array([], dtype=jnp.float32)

        scores_jnp = jnp.array(scores, dtype=jnp.float32)
        scaled = -scores_jnp / jnp.clip(self.temp, 1e-6, None)
        weights = jax.nn.softmax(scaled)

        return tuple(models), weights

    def has_active_archive(self) -> bool:
        return any(entry.active for entry in self.archive)

    def num_active_archive(self) -> int:
        return len([entry for entry in self.archive if entry.active])

    def to_dict(self) -> dict:
        """
        Serialize league state to dictionary.

        Returns
        -------
        dict
            Serialized league state.
        """
        entries_data = []
        for entry in self.entries:
            state = nnx.state(entry.model)
            state_pure = nnx.to_pure_dict(state)
            entry_data = {
                "state": state_pure,
                "score_ema": float(entry.score_ema),
                "rollouts": int(entry.rollouts),
                "name": entry.name,
            }
            entries_data.append(entry_data)

        archive_data = []
        for entry in self.archive:
            state = nnx.state(entry.model)
            state_pure = nnx.to_pure_dict(state)
            archive_data.append(
                {
                    "state": state_pure,
                    "score_ema": float(entry.score_ema),
                    "rollouts": int(entry.rollouts),
                    "name": entry.name,
                    "active": entry.active,
                }
            )

        return {
            "max_size": self.max_size,
            "temp": self.temp,
            "alpha": self.alpha,
            "entries": entries_data,
            "archive": archive_data,
            "archive_max_size": self.archive_max_size,
        }

    def from_dict(self, data: dict, env) -> None:
        """
        Restore league state from dictionary.

        Parameters
        ----------
        data : dict
            Serialized league state from to_dict().
        """
        obs_shape = env.observation_shape
        raw_actions = env.num_actions
        if isinstance(raw_actions, tuple):
            action_dim = raw_actions
        else:
            action_dim = (raw_actions,)
        model_template = nnx.eval_shape(
            lambda: instantiate(self.cfg.model, _partial_=True)(
                rngs=nnx.Rngs(0),
                obs_shape=env.observation_shape,
                action_dim=env.num_actions,
            )
        )
        graphdef, _ = nnx.split(model_template)

        self.max_size = int(data["max_size"])
        self.temp = float(data["temp"])
        self.alpha = float(data["alpha"])
        self.entries = []
        self.archive = []
        self.archive_max_size = int(
            data.get("archive_max_size", getattr(self, "archive_max_size", 64))
        )

        for entry_data in data["entries"]:
            _, fresh_state = nnx.split(model_template)
            model = self._restore_model_from_entry(entry_data, graphdef, fresh_state)
            entry = LeagueEntry(
                model,
                score_ema=float(entry_data["score_ema"]),
                name=entry_data["name"],
                rollouts=int(entry_data["rollouts"]),
            )
            self.entries.append(entry)

        for entry_data in data.get("archive", []):
            _, fresh_state = nnx.split(model_template)
            model = self._restore_model_from_entry(entry_data, graphdef, fresh_state)
            entry = LeagueArchiveEntry(
                model,
                score_ema=float(entry_data["score_ema"]),
                name=entry_data["name"],
                active=entry_data["active"],
                rollouts=int(entry_data["rollouts"]),
            )
            self.archive.append(entry)
        self._archive_prune()

    def entries_to_csv(self) -> str:
        """
        Serialize active league entries to a CSV string.

        Returns
        -------
        str
            CSV-encoded standings sorted by decreasing score EMA.
        """
        buffer = io.StringIO()
        fieldnames = ("name", "score_ema", "rollouts")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        sorted_entries = sorted(self.entries, key=lambda e: float(e.score_ema))
        for entry in sorted_entries:
            writer.writerow(
                {
                    "name": entry.name or "agent",
                    "score_ema": float(entry.score_ema),
                    "rollouts": int(entry.rollouts),
                }
            )

        return buffer.getvalue()

    def entries_to_pandas(self) -> pd.DataFrame:
        entries_data = []
        for entry in self.entries:
            entries_data.append(
                {
                    "name": entry.name or "agent",
                    "score_ema": float(entry.score_ema),
                    "rollouts": int(entry.rollouts),
                }
            )
        return pd.DataFrame(entries_data)

    def archive_to_csv(self) -> str:
        """
        Serialize archive to a CSV string.

        Returns
        -------
        str
            CSV-encoded archive sorted by decreasing score EMA.
        """
        buffer = io.StringIO()
        fieldnames = ("name", "score_ema", "rollouts", "active")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        sorted_archive = sorted(self.archive, key=lambda e: float(e.score_ema))
        for entry in sorted_archive:
            writer.writerow(
                {
                    "name": entry.name or "agent",
                    "score_ema": float(entry.score_ema),
                    "rollouts": int(entry.rollouts),
                    "active": entry.active,
                }
            )
        return buffer.getvalue()

    def archive_to_pandas(self) -> pd.DataFrame:
        archive_data = []
        for entry in self.archive:
            archive_data.append(
                {
                    "name": entry.name or "agent",
                    "score_ema": float(entry.score_ema),
                    "rollouts": int(entry.rollouts),
                    "active": entry.active,
                }
            )
        return pd.DataFrame(archive_data)
