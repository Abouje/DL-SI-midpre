from typing import List

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 8, channels: List[int] = None) -> None:
        super().__init__()
        if channels is None:
            channels = [64, 128, 256, 512]

        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        c_in = in_channels
        for c_out in channels:
            self.down_blocks.append(DoubleConv(c_in, c_out))
            self.pools.append(nn.MaxPool2d(2))
            c_in = c_out

        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2)

        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        c_in = channels[-1] * 2
        for c_out in reversed(channels):
            self.up_transpose.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2))
            self.up_blocks.append(DoubleConv(c_out * 2, c_out))
            c_in = c_out

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down, pool in zip(self.down_blocks, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up_t, up_b, skip in zip(self.up_transpose, self.up_blocks, reversed(skips)):
            x = up_t(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = up_b(x)

        return self.head(x)
