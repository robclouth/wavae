import torch
from src import get_model

vae = get_model()
print(vae.__class__.__name__)
print(vae)
x = torch.randn(1, 2**15)
vae(x)