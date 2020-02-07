import torch
import torch.nn as nn
from . import config


class ConvEncoder(nn.Module):
    """
    Multi Layer Convolutional Variational Encoder
    """
    def __init__(self, channels, kernel, ratios):
        super().__init__()

        self.channels = channels
        self.kernel = kernel
        self.ratios = ratios

        self.convs = nn.ModuleList([
            nn.Conv1d(self.channels[i],
                      self.channels[i+1],
                      self.kernel,
                      padding=self.kernel//2,
                      stride=self.ratios[i])\
            for i in range(len(self.ratios))
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.channels[i])\
            for i in range(1,len(self.ratios))
        ])

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs) - 1:
                x = self.bns[i](torch.relu(x))
        return x


class ConvDecoder(nn.Module):
    """
    Multi Layer Convolutional Variational Decoder
    """
    def __init__(self, channels, ratios, kernel):

        self.channels = channels
        self.ratios = ratios
        self.kernel = kernel

        super().__init__()
        channels = list(self.channels)
        channels[-1] //= 2
        channels[0] *= 2
        self.convs = nn.ModuleList([])
        for i in range(len(self.ratios))[::-1]:
            if self.ratios[i] != 1:
                self.convs.append(
                    nn.ConvTranspose1d(channels[i + 1],
                                       channels[i],
                                       2 * self.ratios[i],
                                       padding=self.ratios[i] // 2,
                                       stride=self.ratios[i]))
            else:
                self.convs.append(
                    nn.Conv1d(channels[i + 1],
                              channels[i],
                              self.kernel,
                              padding=self.kernel // 2))

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(channels[i])\
            for i in range(1,len(self.ratios))[::-1]
        ])

        self.ar_freq = nn.GRU(1, 512, 1, batch_first=True)
        self.lin_post_ar = nn.Linear(512, 1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs) - 1:
                x = self.bns[i](torch.relu(x))
        # X.shape B x 128 x T
        bsize = x.shape[0]
        x = x.permute(0, 2, 1).reshape(-1, self.channels[0], 1)
        x = self.ar_freq(x)[0]
        x = self.lin_post_ar(x)
        x = x.reshape(bsize, -1, self.channels[0]).permute(0, 2, 1)

        return x


class TopVAE(nn.Module):
    """
    Top Variational Auto Encoder
    """
    def __init__(self, channels, kernel, ratios):
        super().__init__()
        self.encoder = ConvEncoder(channels, kernel, ratios)
        self.decoder = ConvDecoder(channels, ratios, kernel)

        self.channels = channels

        skipped = 0
        for p in self.parameters():
            try:
                nn.init.xavier_normal_(p)
            except:
                skipped += 1
        print(f"Skipped {skipped} parameters during initialisation")

    def encode(self, x):
        mean, logvar = torch.split(self.encoder(x), self.channels[-1] // 2, 1)
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
