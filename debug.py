import sounddevice as sd
import torch
torch.set_grad_enabled(False)
import librosa as li
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

model = torch.jit.load("runs/alexander/trace_model.ts").cuda()

x, sr = li.load("wav/zelda_cropped.wav", 16000)
x = x[:2 * sr]

if len(x) % 512:
    x = x[:-(len(x) % 512)]

x = x.reshape(-1, 512)
y = []

for elm in tqdm(x):
    elm = torch.from_numpy(elm).float().cuda().reshape(1, -1)
    y.append(model(elm).cpu().squeeze().numpy())

sample_N = y[0].shape[0]
hop = (sample_N - 512) // 2 + 512
N = sample_N + (len(y) - 1) * hop

out = np.zeros(N)

for i, elm in enumerate(y):
    out[i * hop:i * hop + sample_N] += elm

sd.play(out, sr)
sd.wait()