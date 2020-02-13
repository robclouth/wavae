import torch
torch.set_grad_enabled(False)
from time import time
from tqdm import tqdm
from src import config

model = torch.jit.load("runs/dry/trace_model.ts")

x = torch.randn(100, config.BUFFER_SIZE)

x = x.reshape(-1, config.BUFFER_SIZE)

print("Testing with size", config.BUFFER_SIZE)

for elm in tqdm(x):
    model(elm.reshape(1, -1))
