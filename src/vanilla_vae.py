import torch
import torch.nn as nn
from . import config, cache_pad
import numpy as np


class ConvEncoder(nn.Module):
    """
    Multi Layer Convolutional Variational Encoder
    """
    def __init__(self, channels, kernel, ratios, lin_size, use_cached_padding):
        super().__init__()

        self.channels = channels
        self.kernel = kernel
        self.ratios = ratios

        self.convs = []
        for i in range(len(self.ratios)):
            self.convs += [
                cache_pad(self.kernel // 2, self.channels[i],
                          use_cached_padding),
                nn.Conv1d(self.channels[i],
                          self.channels[i + 1],
                          self.kernel,
                          padding=0,
                          stride=self.ratios[i]),
                nn.ReLU(),
                nn.BatchNorm1d(self.channels[i + 1])
            ]

        self.convs = nn.Sequential(*self.convs)

        self.lins = []
        for i in range(len(lin_size) - 1):
            self.lins.append(nn.Linear(lin_size[i], lin_size[i + 1]))
            if i != len(lin_size) - 2:
                self.lins.append(nn.ReLU())
        self.lins = nn.Sequential(*self.lins)

    def forward(self, x):
        x = self.convs(x)
        x = x.permute(0, 2, 1)
        x = self.lins(x)
        x = x.permute(0, 2, 1)
        return x


class ConvDecoder(nn.Module):
    """
    Multi Layer Convolutional Variational Decoder
    """
    def __init__(self, channels, ratios, lin_size, kernel, use_cached_padding):

        self.channels = channels
        self.ratios = ratios
        self.kernel = kernel

        super().__init__()
        self.channels = list(self.channels)
        self.channels[0] *= 2

        self.lin_size = list(lin_size)
        self.lin_size[-1] //= 2

        self.lins = []

        for i in range(len(self.lin_size) - 1)[::-1]:
            self.lins.append(nn.Linear(self.lin_size[i + 1], self.lin_size[i]))
            self.lins.append(nn.ReLU())
        self.lins = nn.Sequential(*self.lins)

        self.convs = []

        for i in range(len(self.ratios))[::-1]:
            if self.ratios[i] == 1:
                self.convs += [
                    cache_pad(self.kernel // 2, self.channels[i + 1],
                              use_cached_padding),
                    nn.Conv1d(self.channels[i + 1], self.channels[i],
                              self.kernel)
                ]

            else:
                self.convs += [
                    nn.ConvTranspose1d(self.channels[i + 1],
                                       self.channels[i],
                                       2 * self.ratios[i],
                                       stride=self.ratios[i],
                                       padding=self.ratios[i] // 2),
                ]
            if i:
                self.convs += [nn.ReLU(), nn.BatchNorm1d(self.channels[i])]

        self.convs = nn.Sequential(*self.convs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.lins(x)
        x = x.permute(0, 2, 1)
        x = self.convs(x)
        return x

    def allow_spreading(self):
        for elm in self.convs:
            if elm.__class__.__name__ == "ConvTranspose1d":
                elm.padding = (0, )


class TopVAE(nn.Module):
    """
    Top Variational Auto Encoder
    """
    def __init__(self, channels, kernel, ratios, lin_size, use_cached_padding):
        super().__init__()
        self.encoder = ConvEncoder(channels, kernel, ratios, lin_size,
                                   use_cached_padding)
        self.decoder = ConvDecoder(channels, ratios, lin_size, kernel,
                                   use_cached_padding)

        self.channels = channels
        self.lin_size = lin_size

        skipped = 0
        for p in self.parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                skipped += 1

    def encode(self, x):
        out = self.encoder(x)
        mean, logvar = torch.split(out, self.lin_size[-1] // 2, 1)
        z = torch.randn_like(mean) * torch.exp(logvar) + mean
        return z, mean, logvar

    def decode(self, z):
        rec = self.decoder(z)
        mean, logvar = torch.split(rec, self.channels[0], 1)
        mean = torch.sigmoid(mean)
        logvar = torch.clamp(logvar, min=-10, max=0)
        y = torch.randn_like(mean) * torch.exp(logvar) + mean
        return y, mean, logvar

    def forward(self, x):
        z, mean_z, logvar_z = self.encode(x)
        y, mean_y, logvar_y = self.decode(z)
        return y, mean_y, logvar_y, mean_z, logvar_z

    def loss(self, x):
        y, mean_y, logvar_y, mean_z, logvar_z = self.forward(x)

        loss_rec = logvar_y + (x - mean_y)**2 * torch.exp(-logvar_y)

        loss_reg = mean_z**2 + torch.exp(logvar_z) - logvar_z - 1

        loss_rec = torch.mean(loss_rec)
        loss_reg = torch.mean(loss_reg)

        return y, mean_y, logvar_y, mean_z, logvar_z, loss_rec, loss_reg
