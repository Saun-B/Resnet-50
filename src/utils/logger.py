from __future__ import annotations

import csv
import logging
from pathlib import Path
from datetime import datetime

def setup_logger(log_dir: str | Path, name: str = "train", log_filename: str = "train.log") -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logger initialized. Writing to: {log_dir / log_filename}")
    return logger

class CSVLogger:
    def __init__(self, csv_path: str | Path, fieldnames: list[str] | None = None):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        self._writer = None
        self._file = None

    def open(self):
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames or [])
        if self.fieldnames:
            self._writer.writeheader()

    def log(self, row: dict):
        if self._writer is None:
            self.fieldnames = list(row.keys())
            self.open()
        assert self._writer is not None
        self._writer.writerow(row)
        self._file.flush()
    def close(self):
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None