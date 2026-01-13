from __future__ import annotations

import torch
import torch.nn as nn

from .conv_helpers import conv1x1, conv3x3

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        out_channels = planes * self.expansion

        self.conv1 = conv1x1(in_channels, planes, stride=1)
        self.bn1 = norm_layer(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = norm_layer(planes)

        self.conv3 = conv1x1(planes, out_channels, stride=1)
        self.bn3 = norm_layer(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out