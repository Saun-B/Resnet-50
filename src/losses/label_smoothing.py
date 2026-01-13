from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.1):
        super().__init__()
        if not (0.0 <= eps < 1.0):
            raise ValueError("eps must be in [0, 1)")
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs, target, reduction="mean")
        smooth = -log_probs.mean(dim=-1).mean()
        return (1.0 - self.eps) * nll + self.eps * smooth