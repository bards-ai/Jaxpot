from abc import ABC, abstractmethod

import pandas as pd
from omegaconf import DictConfig


class Logger(ABC):
    def __init__(self, run_id: str | None = None):
        self._run_id = run_id

    @abstractmethod
    def log_config(self, cfg: DictConfig) -> None:
        pass

    @abstractmethod
    def log(self, data: dict, step: int) -> None:
        pass

    @abstractmethod
    def log_histogram(self, name: str, counts_array, bins, step: int) -> None:
        pass

    @abstractmethod
    def log_table(self, name: str, data: pd.DataFrame, step: int) -> None:
        pass

    @property
    @abstractmethod
    def run_id(self) -> str:
        pass

    @abstractmethod
    def close(self):
        pass
