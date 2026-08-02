"""Shared CNN architecture for grayscale and RGB corner regression.

The only difference between the two variants is in_channels (1 vs 3).
Everything else — architecture, loss, schedule, augmentation strategy —
is kept identical so the comparison is fair.

Input:  (B, C, 216, 384)  where C = 1 (gray) or C = 3 (RGB)
Output: (B, 8)  normalized corner coordinates in [0, 1]

Spatial sizes after each ConvBlock (stride-2 first conv, stride-1 second):
    Input:  216 × 384
    Block 1: 108 × 192   32 ch
    Block 2:  54 ×  96   64 ch
    Block 3:  27 ×  48  128 ch
    Block 4:  14 ×  24  192 ch
    Block 5:   7 ×  12  256 ch

The spatial grid is kept at 6×10 after the head so position information
is preserved — not collapsed to 1×1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DocumentCNN(nn.Module):
    def __init__(self, in_channels: int = 1, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32,  stride=2),  # (B, 32,  108, 192)
            ConvBlock(32,          64,  stride=2),  # (B, 64,   54,  96)
            ConvBlock(64,          128, stride=2),  # (B, 128,  27,  48)
            ConvBlock(128,         192, stride=2),  # (B, 192,  14,  24)
            ConvBlock(192,         256, stride=2),  # (B, 256,   7,  12)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(256, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Keep 6×10 grid so the model retains spatial/location information.
        self.pool = nn.AdaptiveAvgPool2d((6, 10))   # (B, 64, 6, 10)
        flat_dim = 64 * 6 * 10                      # 3840
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.reduce(x)
        x = self.pool(x)
        return self.head(x)
