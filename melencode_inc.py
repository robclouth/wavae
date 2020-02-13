#%%
from src import cache_pad_transpose
import torch
torch.set_grad_enabled(False)
import torch.nn as nn
import matplotlib.pyplot as plt

# %%
conv = nn.ConvTranspose1d(1, 1, 4, stride=2, padding=1)
x = torch.randn(1, 1, 16)

y_full = conv(x)

x = torch.split(x, 8, -1)
conv.padding = (0, )
conv_pad = nn.Sequential(conv, cache_pad_transpose(1, 1, True))
y = []
for elm in x:
    y.append(conv_pad(elm))

y_split = torch.cat(y, -1)

plt.plot(y_full.squeeze())
plt.plot(y_split.squeeze())

# %%

# %%
