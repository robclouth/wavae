#%% IMPORT
import torch
torch.set_grad_enabled(False)
from src import config
config.override(NAME="alexander_cached")
from make_wrapper import Wrapper
import librosa as li
import matplotlib.pyplot as plt
import sounddevice as sd
from tqdm import tqdm

x, sr = li.load("drum.wav", 16000)
if len(x) % 4096:
    x = x[:-(len(x) % 4096)]
x = torch.from_numpy(x).float().unsqueeze(0)

#%% LOAD AND COMPUTE FULL WAVEFORM
config.override(USE_CACHED_PADDING=False)
wrapper_full = Wrapper()
wrapper_full.eval()

mel_full = wrapper_full.trace_melencoder(x)
z_full = torch.split(wrapper_full.trace_encoder(mel_full), 16, 1)[0]
melrec_full = torch.sigmoid(
    torch.split(wrapper_full.trace_decoder(z_full), 128, 1)[0])
waveform_full = wrapper_full.trace_melgan(melrec_full).squeeze()

# %% COMPUTE SPLIT WAVEFORM
config.override(USE_CACHED_PADDING=True)
wrapper_split = Wrapper()
wrapper_split.eval()

mel_split = []
z_split = []
melrec_split = []
waveform_split = []

for elm in tqdm(torch.split(x, 4096, 1)):
    mel_split.append(wrapper_split.trace_melencoder(elm))
    z_split.append(
        torch.split(wrapper_split.trace_encoder(mel_split[-1]), 16, 1)[0])
    melrec_split.append(
        torch.sigmoid(
            torch.split(wrapper_split.trace_decoder(z_split[-1]), 128, 1)[0]))
    waveform_split.append(
        wrapper_split.trace_melgan(melrec_split[-1]).squeeze())

mel_split = torch.cat(mel_split, -1)
z_split = torch.cat(z_split, -1)
melrec_split = torch.cat(melrec_split, -1)
waveform_split = torch.cat(waveform_split, -1)
# %% DISPLAY RESULTS

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(mel_full.squeeze(), aspect="auto", origin="lower")
plt.title("mel")
plt.subplot(122)
plt.imshow(mel_split.squeeze(), aspect="auto", origin="lower")
plt.title("mel")
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(z_full.squeeze(), aspect="auto", origin="lower")
plt.title("z")
plt.subplot(122)
plt.imshow(z_split.squeeze(), aspect="auto", origin="lower")
plt.title("z")
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(melrec_full.squeeze(), aspect="auto", origin="lower")
plt.title("melrec")
plt.subplot(122)
plt.imshow(melrec_split.squeeze(), aspect="auto", origin="lower")
plt.title("melrec")
plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(waveform_full.squeeze())
plt.title("waveform")
plt.subplot(122)
plt.plot(waveform_split.squeeze())
plt.title("waveform")
plt.show()

# %%
sd.play(waveform_full.squeeze(), sr)
sd.wait()
sd.play(waveform_split.squeeze(), sr)

# %%
from src import CachedConvTranspose1d

cct1 = CachedConvTranspose1d(1, 1, 4, 2, 1, True)
#%%
x = torch.arange(8).float().reshape(1, 1, -1)
y = torch.arange(8, 16).float().reshape(1, 1, -1)

cct1.pad.reset()
full = cct1(torch.cat([x, y], -1))
cct1.pad.reset()
split = torch.cat([cct1(x), cct1(y)], -1)

plt.plot(full.squeeze())
plt.plot(split.squeeze())

# %%
