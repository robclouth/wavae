#%%
from src import config, get_model, log_loudness
import torch

config.override(EXTRACT_LOUDNESS=True)
model = get_model()

# %%
x = torch.randn(1, 8192)
model(x, torch.from_numpy(log_loudness(x.numpy(), 512)).float())

# %%
print(log_loudness(x.numpy(), 512).shape)

# %%
