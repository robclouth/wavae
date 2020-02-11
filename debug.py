from src import cache_pad
import torch
torch.set_grad_enabled(False)
import torch.nn as nn


class TestScript(nn.Module):
    def __init__(self):
        super().__init__()
        self.pc = cache_pad(2, 1, True)

    def forward(self, x):
        return self.pc(x)


x = torch.arange(16).float().reshape(1, 1, -1)
ts = TestScript()

ts(x)

traced = torch.jit.trace(ts, x)

x = torch.split(x, 4, -1)

for elm in x:
    # print("regular", ts(elm))
    print("traced", traced(elm))
    print("\n")
