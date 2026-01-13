from __future__ import annotations

from typing import Literal
from .resnet50 import resnet50

def build_model(name: Literal["resnet50"], num_classes: int):
    if name == "resnet50":
        return resnet50(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}")