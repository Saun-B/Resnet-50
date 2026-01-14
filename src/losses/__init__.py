from __future__ import annotations
from .cross_entropy import CrossEntropyLoss
from .label_smoothing import LabelSmoothingCrossEntropy

def build_criterion(name: str = "ce", label_smoothing_eps: float = 0.1):
    name = name.lower()
    if name in ("ce", "cross_entropy", "crossentropy"):
        return CrossEntropyLoss()
    if name in ("lsce", "label_smoothing", "labelsmoothing"):
        return LabelSmoothingCrossEntropy(eps=label_smoothing_eps)
    raise ValueError(f"Unknown loss name: {name}")