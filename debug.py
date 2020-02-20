#%%
import torch
torch.set_grad_enabled(False)
import matplotlib.pyplot as plt

model = torch.jit.load("runs/untitled/trace_model.ts")
x = torch.randn(1, 2**15)

z = model.encode(x)
print(z.shape)

# %%
plt.imshow(abs(z.squeeze()))
plt.colorbar()

# %%
