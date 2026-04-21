from pathlib import Path
from typing import TextIO


class FileLogger:
    def __init__(self, log_file: str | Path | None = None):
        self.log_file = log_file
        self.file: TextIO | None = None

    def setup(self) -> Path | None:
        """
        Initialize the log file and write a header.

        Returns
        -------
        Path
            Absolute path of the created log file.
        """
        if self.log_file is None:
            return None

        p = Path(self.log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.file = p.open("w")
        return p

    def write(self, message: str) -> None:
        if self.file and not self.file.closed:
            self.file.write(message + "\n")
            self.file.flush()

    def close(self) -> None:
        if self.file and not self.file.closed:
            self.file.close()
