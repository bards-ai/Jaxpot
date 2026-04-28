import pandas as pd
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from jaxpot.loggers.logger import Logger


class WandbLogger(Logger):
    def __init__(
        self,
        project_name: str,
        run_id: str | None = None,
        run_name: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        **kwargs,
    ):
        super().__init__(run_id)
        self.project_name = project_name
        self.run_name = run_name
        self.entity = entity

        self.run = wandb.init(
            project=self.project_name,
            id=self._run_id,
            name=self.run_name,
            group=group,
            entity=entity,
            tags=tags,
            resume="allow",
            settings=wandb.Settings(console="off"),
            **kwargs,
        )

    def log_config(self, cfg: DictConfig) -> None:
        plain_cfg = OmegaConf.to_container(cfg, resolve=True)
        self.run.config.update(plain_cfg, allow_val_change=True)

    def log(self, data: dict, step: int) -> None:
        wandb.log(data, step=step)

    def log_histogram(self, name: str, counts_array, bins, step: int) -> None:
        # W&B has a limit of 512 bins for histograms
        if len(counts_array) > 512:
            logger.warning(
                f"Action space {len(counts_array)} too large for W&B histogram, skipping. Increase bin size to avoid this warning."
            )
            return
        else:
            histogram = wandb.Histogram(np_histogram=(counts_array, bins))
            self.log({name: histogram}, step)

    def log_table(self, name: str, data: pd.DataFrame, step: int) -> None:
        # W&B does not support tracking tables over time, so table logging is disabled.
        pass

    @property
    def run_id(self) -> str:
        return self.run.id

    def close(self) -> None:
        wandb.finish()
