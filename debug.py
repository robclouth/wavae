#%%
import torch
torch.set_grad_enabled(False)
import librosa as li
import sounddevice as sd
from tqdm import tqdm
import matplotlib.pyplot as plt
#%%
crop = lambda x, n: x[:-(len(x) % n)] if len(x) % n else x

model = torch.jit.load("trace_model.ts").cuda()
# %%
x, sr = li.load("realtime/build/audio_16.wav", 16000)
x = x[:10 * sr]
x = torch.split(torch.from_numpy(crop(x, 4096)).float(), 4096, -1)
z = torch.cat([model.encode(elm.cuda()) for elm in tqdm(x)],
              -1).squeeze().cpu()

# %%
import soundfile as sf
sf.write("billie.wav", y, sr)