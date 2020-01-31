from . import Generator, Discriminator, Encoder, MelEncoder, config
import torch.nn as nn


class vqvaeGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Generator()

    def forward(self, x):
        zq, diff, idx = self.encoder(x)
        y = self.decoder(zq)
        return y, zq, diff, idx


class melGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MelEncoder()
        self.decoder = Generator()

    def forward(self, x):
        mel = self.encoder(x)
        y = self.decoder(mel)
        return y


def get_model():
    if config.TYPE == "autoencoder":
        return vqvaeGAN()
    elif config.TYPE == "melgan":
        return melGAN()
    else:
        raise Exception(f"Model type {config.TYPE} not understood")
