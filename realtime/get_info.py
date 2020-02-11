import torch

melencoder = torch.jit.load("melencoder_trace.ts")

x = torch.randn(1, 4096)

print(melencoder(x).shape)