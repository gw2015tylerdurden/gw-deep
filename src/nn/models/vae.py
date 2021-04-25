import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc

from .basic import *


__all__ = ["VAE"]


class VAE(BaseModule):
    def __init__(self, in_channels: int = 3, z_dim: int = 512, msize: int = 7):
        super().__init__()
        self.encoder = Encoder(in_channels, z_dim)
        self.decoder = Decoder(z_dim, in_channels, msize)
        self.weight_init()

    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encoder(x)
        x_rec = self.decoder(z)
        bce = self.bce(x_rec, x)
        kl_gauss = self.kl_gauss(mean, logvar)
        return bce, kl_gauss

    def params(self, x: torch.Tensor):
        assert not self.training
        z, mean, logvar = self.encoder(x)
        x_rec = self.decoder(z)
        return mean, x_rec

    def bce(self, x_rec: torch.Tensor, x: torch.Tensor):
        # Minibatch size, Channels, Height, Width
        M, C, H, W = x.shape
        bce = F.binary_cross_entropy_with_logits(x_rec, x, reduction="sum")
        return bce / C / W / H

    def bernoulli(self, x_rec: torch.Tensor, x: torch.Tensor):
        '''
        instable caclulation. loss becomes inf
        '''
        M, C, H, W = x.shape
        sig_x_rec = x_rec.sigmoid()
        correct_bern = -torch.sum(x * torch.log(sig_x_rec)) / C / H / W
        incorrect_bern = -torch.sum((1 - x) * torch.log(1 - sig_x_rec)) / C / H / W
        return correct_bern, incorrect_bern

   def mse(self, x_rec: torch.Tensor, x: torch.Tensor):
        # Minibatch size, Channels, Height, Width
        M, C, H, W = x.shape
        # divided only M, not C,H,W. MSE value should be a distance in R^{C*H*W}
        return torch.sum((x_rec - x) ** 2) / M

    def kl_gauss(self, mean: torch.Tensor, logvar: torch.Tensor):
        # Minibatch size, J: dimension of latent space
        M, J = mean.shape
        kl = -0.5 * torch.sum(1 + logvar - torch.pow(mean, 2) - logvar.exp())
        # divided only M, not J. D_kl should be a divergence in R^{J}
        return kl / J
