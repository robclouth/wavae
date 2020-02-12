import torch
torch.set_grad_enabled(False)
import librosa as li
import numpy as np
import matplotlib.pyplot as plt

model = torch.jit.load("runs/alexander/trace_model.ts")
x, sr = li.load("wav/zelda_cropped.wav", 16000)

x_ = x.reshape(-1, 512)
mel = []

for elm in x_:
    elm = torch.from_numpy(elm).float()
    mel.append(model.melencode(elm.reshape(1, -1)).numpy().squeeze().T)

mel = np.asarray(mel)
mel = mel.reshape(-1, 128).T

mel_li = li.feature.melspectrogram(x)

plt.subplot(121)
plt.imshow(mel, aspect="auto", origin="lower")
plt.subplot(122)
plt.imshow(np.log10(mel_li + 1e-5), aspect="auto", origin="lower")
plt.show()
