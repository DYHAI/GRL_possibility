"""Shared U-Net and normalization utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def normalize_channels(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean[:, None, None]) / std[:, None, None]


def denormalize_channels(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std[:, None, None] + mean[:, None, None]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 4, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def radial_energy(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = field - field.mean()
    spec = np.abs(np.fft.rfft2(f)) ** 2
    nx, ny = f.shape
    kx = np.fft.fftfreq(nx)
    ky = np.fft.rfftfreq(ny)
    kx_g, ky_g = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(kx_g**2 + ky_g**2)
    kbins = np.linspace(0, K.max(), 36)
    power = np.zeros(len(kbins) - 1)
    for i in range(len(kbins) - 1):
        mask = (K >= kbins[i]) & (K < kbins[i + 1])
        power[i] = spec[mask].mean() if mask.any() else 0.0
    k_centers = 0.5 * (kbins[:-1] + kbins[1:])
    return k_centers, power
