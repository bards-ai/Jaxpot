import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from jaxpot.loggers.logger import Logger


class MultiLogger(Logger):
    """
    Composite logger that delegates to multiple loggers.

    When instantiated with an empty list, acts as a no-op logger.

    Parameters
    ----------
    logger_configs
        Mapping of child-logger name to Hydra-style DictConfig. Each child is
        instantiated with a shared ``run_id``. The keys are used only for
        reporting/override ergonomics; iteration order is insertion order.
    run_id
        Shared run identifier propagated to all child loggers.
    """

    def __init__(
        self,
        run_id: str | None = None,
        logger_configs: dict[str, DictConfig] | None = None,
    ):
        super().__init__(run_id)
        self._loggers: list[Logger] = []
        for cfg in (logger_configs or {}).values():
            self._loggers.append(instantiate(cfg, run_id=self._run_id))

    def log_config(self, cfg: DictConfig) -> None:
        for logger in self._loggers:
            logger.log_config(cfg)

    def log(self, data: dict, step: int) -> None:
        for logger in self._loggers:
            logger.log(data, step)

    def log_histogram(self, name: str, counts_array, bins, step: int) -> None:
        for logger in self._loggers:
            logger.log_histogram(name, counts_array, bins, step)

    def log_table(self, name: str, data: pd.DataFrame, step: int) -> None:
        for logger in self._loggers:
            logger.log_table(name, data, step)

    @property
    def run_id(self) -> str:
        assert self._run_id is not None
        return self._run_id

    def close(self) -> None:
        for logger in self._loggers:
            logger.close()
