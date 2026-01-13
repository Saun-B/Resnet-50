from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image

class ListFileImageDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        list_file: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.list_file = Path(list_file)
        self.transform = transform

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"dataset_root not found: {self.dataset_root}")
        if not self.list_file.exists():
            raise FileNotFoundError(f"list_file not found: {self.list_file}")

        self.samples: List[Tuple[str, int]] = []
        with self.list_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    raise ValueError(f"Bad line at {self.list_file}:{line_no}: {line}")
                rel_path = parts[0]
                label = int(parts[1])
                self.samples.append((rel_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in list_file: {self.list_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, label = self.samples[idx]
        img_path = self.dataset_root / rel_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label