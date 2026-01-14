from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn

def build_optimizer(
    model: nn.Module,
    name: str = "sgd",
    lr: float = 0.1,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
):
    name = name.lower()

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=False,
        )

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown optimizer name: {name}")