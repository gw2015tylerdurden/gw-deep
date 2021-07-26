import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc
import numpy as np

from .basic import *


__all__ = ["VAE"]


class VAE(BaseModule):
    def __init__(self, in_channels: int = 3, z_dim: int = 512, msize: int = 7, is_output_gaussian=False):
        super().__init__()
        self.encoder = Encoder(in_channels, z_dim)
        if is_output_gaussian:
            self.decoder = GaussianDecoder(z_dim, in_channels, msize)
        else:
            self.decoder = Decoder(z_dim, in_channels, msize)
        self.weight_init()
        self.is_output_gaussian = is_output_gaussian

    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encoder(x)
        if self.is_output_gaussian:
            mean_output, logvar_output = self.decoder(z)
            logp_xz = self.gaussian(x, mean_output, logvar_output)
        else:
            x_rec = self.decoder(z)
            logp_xz = self.bce(x_rec, x)
            #logp_xz = self.mse(x_rec, x)

        kl_gauss = self.kl_gauss(mean, logvar)
        return logp_xz, kl_gauss

    def params(self, x: torch.Tensor):
        assert not self.training
        z, mean, logvar = self.encoder(x)
        if self.is_output_gaussian:
            mean_output, logvar_output = self.decoder(z)
            return mean, mean_output, logvar_output
        else:
            x_rec = self.decoder(z)
            return mean, x_rec

    def bce(self, x_rec: torch.Tensor, x: torch.Tensor):
        # Minibatch size, Channels, Height, Width
        M, C, H, W = x.shape
        bce = F.binary_cross_entropy_with_logits(x_rec, x, reduction="sum")
        #return bce / C / W / H
        return bce / M

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
        return 0.5 * torch.sum((x_rec - x) ** 2) / M

    def gaussian(self, x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor):
        r"""
        x : input image
        mean, logvar : decoder output from latent variables, same shape as x
        logvar = ln sigma**2
        """
        # Minibatch size, Channels, Height, Width
        M, C, H, W = x.shape
        # negative log likelihood of x under gaussian
        gauss = torch.sum(
            (0.5 * np.log(2.0 * np.pi))
            + (0.5 * logvar)
            + ( 0.5 * ((x - mean) ** 2) / (torch.exp(logvar)))
        )
        #gauss = torch.sum(torch.log(sigma)) + torch.sum(((x-mean)/sigma)**2)

        return gauss / M

    def kl_gauss(self, mean: torch.Tensor, logvar: torch.Tensor):
        # Minibatch size, J: dimension of latent space
        M, J = mean.shape
        kl = -0.5 * torch.sum(1 + logvar - torch.pow(mean, 2) - logvar.exp())
        # divided only M, not J. D_kl should be a divergence in R^{J}
        return kl / M
