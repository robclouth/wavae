import torch
torch.set_grad_enabled(False)
import librosa as li
import matplotlib.pyplot as plt

melencoder = torch.jit.load("../runs/alexander/melencoder_trace.ts")
decoder = torch.jit.load("../runs/alexander/decoder_trace.ts")

x, fs = li.load("audio.wav", 16000)
x = x[:3 * fs]
x = torch.from_numpy(x).float()

x = x[:-(x.shape[-1] % 512)]
x = x.reshape(-1, 512)

out = []

for elm in x:
    out.append(melencoder(elm))

out = torch.cat(out, -1)

plt.imshow(out.squeeze(), aspect="auto")
plt.show()