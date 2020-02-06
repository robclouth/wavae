import torch
import torch.nn as nn
from . import Generator, Discriminator, MelEncoder, TopVAE, config


class Vanilla(nn.Module):
    def __init__(self, hop, ratios, input_size, channels, kernel):
        super().__init__()
        self.melencoder = MelEncoder(hop=hop, input_size=input_size)
        self.topvae = TopVAE(channels=channels, kernel=kernel, ratios=ratios)

    def forward(self, x):
        with torch.no_grad():
            S = self.melencoder(x)
        y, mean_y, logvar_y, mean_z, logvar_z = self.topvae(S)
        return y, mean_y, logvar_y, mean_z, logvar_z


class melGAN(nn.Module):
    def __init__(self, hop, ratios, input_size, ngf, n_res_g):
        super().__init__()
        self.encoder = MelEncoder(hop=hop, input_size=input_size)
        self.decoder = Generator(input_size=input_size,
                                 ngf=ngf,
                                 n_residual_layers=n_res_g,
                                 ratios=ratios)

    def forward(self, x, mel_encoded=False):
        if mel_encoded:
            mel = x
        else:
            mel = self.encoder(x)

        y = self.decoder(mel)
        return y


def get_model(config=config):
    if config.TYPE == "melgan":
        return melGAN(hop=config.HOP_LENGTH,
                      ratios=config.RATIOS,
                      input_size=config.INPUT_SIZE,
                      ngf=config.NGF,
                      n_res_g=config.N_RES_G)

    elif config.TYPE == "vanilla":
        return Vanilla(hop=config.HOP_LENGTH,
                       ratios=config.RATIOS,
                       input_size=config.INPUT_SIZE,
                       channels=config.CHANNELS,
                       kernel=config.KERNEL)
    else:
        raise Exception(f"Model type {config.TYPE} not understood")
