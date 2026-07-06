#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
轻量灼烧恢复网络。

输入:
    x: [B, 5, H, W]，历史 5 帧灰度红外图像。

输出:
    P_logits: [B, 1, H, W]，灼烧概率 logits，使用时再 sigmoid。
    C: [B, 1, H, W]，有效偏置纠正图，范围 0~1。

推理用法:
    future_restored = future_frame - C
    或:
    future_restored = future_frame - (P > threshold) * C
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：depthwise 3x3 + pointwise 1x1。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LiteMultiScaleBlock(nn.Module):
    """轻量多尺度块：3x3 分支 + 膨胀 3x3 分支 + 1x1 融合。"""

    def __init__(self, channels: int):
        super().__init__()
        mid_channels = max(channels // 2, 8)
        self.local = nn.Sequential(
            DepthwiseSeparableConv(channels, mid_channels),
            DepthwiseSeparableConv(mid_channels, mid_channels),
        )
        self.context = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([self.local(x), self.context(x)], dim=1)
        return x + self.fuse(y)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = DepthwiseSeparableConv(in_channels, out_channels, stride=2)
        self.ms = LiteMultiScaleBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ms(self.down(x))


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class BurnRecoveryNet(nn.Module):
    """
    1x1 时序融合 + 两级轻量多尺度编码器 + 轻量解码器 + 双输出头。
    """

    def __init__(self, in_frames: int = 5, base_channels: int = 32):
        super().__init__()
        self.in_frames = int(in_frames)
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.temporal_fuse = nn.Sequential(
            nn.Conv2d(in_frames, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(c1, c1),
        )

        self.enc1 = EncoderStage(c1, c2)
        self.enc2 = EncoderStage(c2, c3)

        self.bottleneck = nn.Sequential(
            LiteMultiScaleBlock(c3),
            DepthwiseSeparableConv(c3, c3),
        )

        self.dec1 = DecoderStage(c3, c2, c2)
        self.dec2 = DecoderStage(c2, c1, c1)

        self.prob_head = nn.Sequential(
            DepthwiseSeparableConv(c1, c1),
            nn.Conv2d(c1, 1, kernel_size=1),
        )
        self.correction_head = nn.Sequential(
            DepthwiseSeparableConv(c1, c1),
            nn.Conv2d(c1, 1, kernel_size=1),
        )
        self._init_sparse_output_biases()

    def _init_sparse_output_biases(self):
        nn.init.constant_(self.prob_head[-1].bias, -4.0)
        nn.init.constant_(self.correction_head[-1].bias, -4.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_frames:
            raise ValueError(f"Expected {self.in_frames} input frames, got {x.shape[1]}")

        s1 = self.temporal_fuse(x)
        s2 = self.enc1(s1)
        z = self.enc2(s2)
        z = self.bottleneck(z)

        y = self.dec1(z, s2)
        y = self.dec2(y, s1)

        prob_logits = self.prob_head(y)
        correction = torch.sigmoid(self.correction_head(y))
        return prob_logits, correction


def restore_future_frame(
    future_frame: torch.Tensor,
    prob: torch.Tensor,
    correction: torch.Tensor,
    threshold: float | None = 0.5,
    prob_is_logits: bool = False,
) -> torch.Tensor:
    """
    使用网络输出恢复未来帧。

    future_frame: [B, 1, H, W] 或 [B, H, W]
    prob: [B, 1, H, W]
    correction: [B, 1, H, W]
    threshold:
        None 表示直接 future - C；
        数值表示使用安全门控 future - (P > threshold) * C。
    """
    squeeze_back = False
    if future_frame.ndim == 3:
        future_frame = future_frame.unsqueeze(1)
        squeeze_back = True

    if prob_is_logits:
        prob = torch.sigmoid(prob)

    if threshold is None:
        restored = future_frame - correction
    else:
        gate = (prob > threshold).to(correction.dtype)
        restored = future_frame - gate * correction

    restored = torch.clamp(restored, 0.0, 1.0)
    return restored.squeeze(1) if squeeze_back else restored


if __name__ == "__main__":
    model = BurnRecoveryNet(in_frames=5, base_channels=32)
    x = torch.randn(2, 5, 256, 320)
    logits, c = model(x)
    p = torch.sigmoid(logits)
    print("P logits:", tuple(logits.shape), logits.min().item(), logits.max().item())
    print("P:", tuple(p.shape), p.min().item(), p.max().item())
    print("C:", tuple(c.shape), c.min().item(), c.max().item())
