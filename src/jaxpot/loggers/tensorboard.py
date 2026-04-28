from io import StringIO

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from jaxpot.loggers.logger import Logger


class TensorBoardLogger(Logger):
    def __init__(
        self,
        run_id: str | None = None,
        run_name: str | None = None,
        log_dir: str = "runs",
    ):
        super().__init__(run_id)
        self._log_dir = log_dir
        self._writer = SummaryWriter(logdir=self._log_dir, comment=run_name)

    def log_config(self, cfg: DictConfig) -> None:
        plain_cfg = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(plain_cfg, dict)
        self._writer.add_hparams(
            hparam_dict=self._flatten_config(plain_cfg),
            metric_dict={},
            name=".",
        )
        self._writer.flush()

    def log(self, data: dict, step: int) -> None:
        for key, value in data.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, step)
        self._writer.flush()

    def log_histogram(self, name: str, counts_array, bins, step: int) -> None:
        self._writer.add_histogram_raw(
            tag=name,
            min=float(bins[0]),
            max=float(bins[-1]),
            num=int(counts_array.sum()),
            sum=float((counts_array * (bins[:-1] + bins[1:]) / 2).sum()),
            sum_squares=float((counts_array * ((bins[:-1] + bins[1:]) / 2) ** 2).sum()),
            bucket_limits=bins[1:].tolist(),
            bucket_counts=counts_array.tolist(),
            global_step=step,
        )
        self._writer.flush()

    def log_table(self, name: str, data: pd.DataFrame, step: int) -> None:
        buffer = StringIO()
        data.to_csv(buffer, index=False)
        text = buffer.getvalue()
        formatted_text = f"```csv\n{text}\n```"
        self._writer.add_text(str(name), formatted_text, step)
        self._writer.flush()

    @property
    def run_id(self) -> str:
        assert self._run_id is not None
        return self._run_id

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()

    @staticmethod
    def _flatten_config(cfg: dict, parent_key: str = "", sep: str = "/") -> dict:
        items: list[tuple[str, str | int | float | bool]] = []
        for k, v in cfg.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(TensorBoardLogger._flatten_config(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple, set)):
                items.append((new_key, str(v)))
            elif v is None:
                items.append((new_key, "None"))
            elif isinstance(v, (str, int, float, bool)):
                items.append((new_key, v))
            else:
                items.append((new_key, str(v)))
        return dict(items)
