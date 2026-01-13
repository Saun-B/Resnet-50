from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import Bottleneck, ResNetStem, conv1x1

class ResNet(nn.Module):
    def __init__(
        self,
        block: type[Bottleneck],
        layers: list[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.norm_layer = norm_layer

        self.stem = ResNetStem(in_channels=3, out_channels=64)

        self.in_channels = 64

        self.layer1 = self._make_layer(block, planes=64,  blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights(zero_init_residual)

    def _init_weights(self, zero_init_residual: bool):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0.0)

    def _make_layer(self, block: type[Bottleneck], planes: int, blocks: int, stride: int):
        norm_layer = self.norm_layer
        downsample = None

        out_channels = planes * block.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels, stride=stride),
                norm_layer(out_channels),
            )

        layers = []
        layers.append(
            block(
                in_channels=self.in_channels,
                planes=planes,
                stride=stride,
                downsample=downsample,
                norm_layer=norm_layer,
            )
        )
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=self.in_channels,
                    planes=planes,
                    stride=1,
                    downsample=None,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet50(num_classes: int = 1000, zero_init_residual: bool = False) -> ResNet:
    return ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        zero_init_residual=zero_init_residual,
    )