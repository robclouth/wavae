from . import Generator, Discriminator, Encoder
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