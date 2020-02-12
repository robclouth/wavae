import sounddevice as sd
import torch
torch.set_grad_enabled(False)
import librosa as li
from time import time
from tqdm import tqdm

encoder = torch.jit.load("runs/alexander/encoder_trace.ts").cuda()
decoder = torch.jit.load("runs/alexander/decoder_trace.ts").cuda()

x, sr = li.load("wav/zelda_cropped.wav", 16000)
x = x.reshape(-1, 512)

y = []

for elm in tqdm(x):
    elm = torch.from_numpy(elm).float().reshape(1, -1).cuda()
    z = encoder(elm)
    y.append(decoder(z))