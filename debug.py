import torch
from src.vanilla_vae import ConvDecoder

cd = ConvDecoder([128, 96, 64, 32, 16], [1, 2, 1, 2], 9)
x = torch.randn(1, 8, 8)

print(cd(x).shape)