import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import abc


__all__ = ["ECBlock", "DCBlock", "Encoder", "Decoder"]


class ECBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        padding = math.ceil((kernel_size - stride) / 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = activation

    def forward(self, x):
        x = self.block(x)
        if self.activation is not None:
            return self.activation(x)
        return x


class DCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=stride),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = activation

    def forward(self, x):
        x = self.block(x)
        if self.activation is not None:
            return self.activation(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.blocks = nn.Sequential(
            ECBlock(in_channels, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ECBlock(64, 128, stride=2),
            ECBlock(128, 256, stride=2),
            ECBlock(256, 512, stride=2),
        )
        self.logits = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, out_dim * 2),
        )

    def reparameterize(self, mean, logvar, L=1):
        mean = mean.repeat(L, 1, 1).squeeze()
        logvar = logvar.repeat(L, 1, 1).squeeze()
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z

    def forward(self, x, L=1):
        x = self.blocks(x)
        x = self.logits(x)
        mean, logvar = torch.split(x, x.shape[-1] // 2, -1)
        z = self.reparameterize(mean, logvar, L).view(x.shape[0] * L, -1)
        return z, mean, logvar


class Decoder(nn.Module):
    def __init__(self, in_dim: int, out_channels: int, msize: int = 7):
        super().__init__()
        self.msize = msize
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim * msize * msize, bias=False),
            nn.BatchNorm1d(in_dim * msize * msize),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            DCBlock(in_dim, 256, stride=2),
            DCBlock(256, 128, stride=2),
            DCBlock(128, 64, stride=2),
            DCBlock(64, out_channels, stride=2, activation=None),
        )

    def forward(self, x):
        x = self.head(x)
        x = x.view(x.shape[0], -1, self.msize, self.msize)
        x = self.blocks(x)
        return x
