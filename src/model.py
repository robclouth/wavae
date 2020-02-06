import torch
import torch.nn as nn
from . import Generator, Discriminator, MelEncoder, TopVAE, config


class Vanilla(nn.Module):
    def __init__(self):
        super().__init__()
        self.melencoder = MelEncoder()
        self.topvae = TopVAE()

    def forward(self, x):
        with torch.no_grad():
            S = self.melencoder(x)
        y, mean_y, logvar_y, mean_z, logvar_z = self.topvae(S)
        return y, mean_y, logvar_y, mean_z, logvar_z


class melGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MelEncoder()
        self.decoder = Generator(input_size=config.INPUT_SIZE,
                                 ngf=config.NGF,
                                 n_residual_layers=config.N_RES_G,
                                 ratios=config.RATIOS)

    def forward(self, x):
        mel = self.encoder(x)
        y = self.decoder(mel)
        return y


def get_model():
    if config.TYPE == "melgan":
        return melGAN()
    elif config.TYPE == "vanilla":
        return Vanilla()
    else:
        raise Exception(f"Model type {config.TYPE} not understood")
