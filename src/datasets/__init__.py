from __future__ import annotations

from pathlib import Path
from torch.utils.data import DataLoader

from .imagenet_style import ListFileImageDataset
from .transforms import build_train_transforms, build_val_transforms

def build_dataset(
    dataset_root: str | Path,
    split: str,
    splits_dir: str | Path = "data/splits",
    img_size: int = 224,
):
    splits_dir = Path(splits_dir)
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")

    list_file = splits_dir / ("train_list.txt" if split == "train" else "val_list.txt")
    transform = build_train_transforms(img_size) if split == "train" else build_val_transforms(img_size)

    return ListFileImageDataset(
        dataset_root=dataset_root,
        list_file=list_file,
        transform=transform,
    )

def build_dataloader(
    dataset,
    split: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    shuffle = (split == "train")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == "train"),
    )