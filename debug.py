#%%
import torch
torch.set_grad_enabled(False)
import torch.nn as nn
from src import cache_pad
import matplotlib.pyplot as plt

r = 16


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([
            cache_pad(4, 1, True),
            nn.Conv1d(1, 1, 9, 2),
            cache_pad(2, 1, True),
            nn.Conv1d(1, 1, 5, 2),
            cache_pad(1, 1, True),
            nn.ConvTranspose1d(1, 1, 2 * r, r, padding=r // 2 + r),
        ])

    def forward(self, x):
        for elm in self.model:
            x = elm(x)
            # print(x.shape)
        return x

    def reset(self):
        nb = 0
        for elm in self.model:
            if "Cached" in elm.__class__.__name__:
                elm.reset()
                nb += 1
        print(f"Resetted {nb} caching modules")


# %%
model = ConvNet()
x = torch.randn(1, 1, 128)

y_full = model(x)
model.reset()

y_split = torch.cat([model(elm) for elm in torch.split(x, 16, -1)], -1)

print(y_full.shape)
print(y_split.shape)

plt.plot(y_full.squeeze())
plt.plot(y_split.squeeze())
plt.xlim([200, 400])

# %%

x = torch.randn(1, 1, 16)
conv = nn.ConvTranspose1d(1, 1, 4, 2, padding=1)
cp = cache_pad(1, 1, False)

plt.plot(conv(x).squeeze())

conv.padding = (3, )

y = conv(cp(x))
plt.plot(y.squeeze())

# %%
