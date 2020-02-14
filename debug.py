#%%
from src import CachedConvTranspose1d
import torch

conv = CachedConvTranspose1d(1, 1, 4, 2, cache=True)
x = torch.randn(1, 1, 16)

print(conv(x).shape)

conv = CachedConvTranspose1d(1, 1, 4, 2, cache=False)
x = torch.randn(1, 1, 16)

print(conv(x).shape)