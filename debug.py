import torch
torch.set_grad_enabled(False)
from time import time
from tqdm import tqdm
model = torch.jit.load("runs/dry/trace_model.ts")
from src import config

x = torch.randn(16000 * 10)

if len(x) % config.BUFFER_SIZE:
    x = x[:-(len(x) % config.BUFFER_SIZE)]

x = x.reshape(-1, config.BUFFER_SIZE)

print("Testing with size", config.BUFFER_SIZE)

for elm in tqdm(x):
    model(elm.reshape(1, -1))
