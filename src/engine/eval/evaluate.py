from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct1 = 0
    correct5 = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        logits = model(images)
        loss = criterion(logits, targets)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        maxk = min(5, logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1))

        correct1 += correct[:, :1].sum().item()
        correct5 += correct[:, :maxk].sum().item()

    avg_loss = total_loss / max(1, total_samples)
    top1 = 100.0 * correct1 / max(1, total_samples)
    top5 = 100.0 * correct5 / max(1, total_samples)

    return {"loss": avg_loss, "top1": top1, "top5": top5}