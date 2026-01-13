from __future__ import annotations

import torch

@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)):
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (B,C). Got {logits.shape}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D (B,). Got {targets.shape}")

    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()

    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    out = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        out[f"top{k}"] = 100.0 * correct_k / batch_size
    return out