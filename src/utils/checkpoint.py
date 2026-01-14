from __future__ import annotations
from pathlib import Path
from typing import Any

def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    map_location: str = "cpu",
):
    import torch
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    return ckpt