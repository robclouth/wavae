#%%
import torch
torch.set_grad_enabled(False)
import matplotlib.pyplot as plt

model_full = torch.jit.load("runs/dry/trace_model_feedforward.ts")
model_incremental = torch.jit.load("runs/dry/trace_model.ts")

# %%

x = torch.randn(1, 2**16)

y_full = model_full(x)

x = torch.split(x, 8192, -1)

y_split = torch.cat([model_incremental(elm) for elm in x], -1)

# %%

plt.plot(y_full.squeeze())
plt.plot(y_split.squeeze())
plt.xlim([8100, 8200])

# %%
