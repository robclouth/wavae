#%%
import torch
torch.set_grad_enabled(False)
import matplotlib.pyplot as plt
import librosa as li

model = torch.jit.load("trace_model.ts")
x = torch.from_numpy(li.load("../sample.wav", 16000)[0]).float().unsqueeze(0)

z = model.encode(x)
print(z.shape)

# %%
plt.imshow(abs(z.squeeze()), aspect="auto")
plt.colorbar()

# %%
plt.plot(z.squeeze()[0, :].T)

# %%

y = model(x)