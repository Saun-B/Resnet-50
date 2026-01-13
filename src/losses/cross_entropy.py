from __future__ import annotations

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    """
    Wrapper cho nn.CrossEntropyLoss.
    Input: logits (B, C), target (B,)
    Output: scalar loss
    """

    def __init__(self, weight=None, ignore_index: int = -100):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target)