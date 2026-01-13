from __future__ import annotations

import math
import torch

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 0, last_epoch: int = -1):
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(0, warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps and self.warmup_steps > 0:
                lr = base_lr * float(step) / float(self.warmup_steps)
            else:
                progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
                lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            lrs.append(lr)
        return lrs

def build_scheduler(
    optimizer,
    name: str = "step",

    step_size: int = 10,
    gamma: float = 0.1,

    t_max: int = 20,

    total_steps: int | None = None,
    warmup_steps: int = 0,
):
    name = name.lower()

    if name in ("none", "no", "null"):
        return None

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    if name in ("warmup_cosine", "warmupcosine"):
        if total_steps is None:
            raise ValueError("total_steps is required for warmup_cosine scheduler")
        return WarmupCosineScheduler(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    raise ValueError(f"Unknown scheduler name: {name}")