#%%
import torch
torch.set_grad_enabled(False)
model = torch.jit.load("trace_model.ts").cuda()
x = torch.randn(1, 1024).cuda()

#%%
z = model.encode(x)
y = model.decode(z)

# %%
import librosa as li
from tqdm import tqdm
x, sr = li.load("../sample.wav", 16000)
x = torch.from_numpy(x).float().cuda()

if len(x) % 1024:
    x = x[:-(len(x) % 1024)]

x = torch.split(x.reshape(1, -1), 1024, -1)

y = torch.cat([model(elm) for elm in tqdm(x)], -1)
# %%
import sounddevice as sd

sd.play(y.cpu().numpy().reshape(-1), sr)

# %%
