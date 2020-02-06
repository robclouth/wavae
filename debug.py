from src.vanilla_vae import TopVAE
import torch

x = torch.randn(1, 128, 512)

model = TopVAE()
print(model)
print(model(x)[0].shape)
