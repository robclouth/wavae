import torch
import torch.nn as nn

SCRIPT = True


def cache_pad(*args):
    if SCRIPT:
        return torch.jit.script(CachedPadding(*args))
    else:
        return CachedPadding(*args)


class CachedPadding(nn.Module):
    def __init__(self, padding, channels, cache=False):
        super().__init__()

        left_pad = torch.zeros(1, channels, 2 * padding)
        self.register_buffer("left_pad", left_pad)

        self.padding = padding
        self.cache = cache

    def forward(self, x):
        if self.cache:
            current = torch.cat([self.left_pad, x], -1)
            self.left_pad = x[..., -self.padding:]
        else:
            current = nn.functional.pad(x, (self.padding, self.padding))
        return current

    def __repr__(self):
        return f"CachedPadding(padding={self.padding}, cache={self.cache})"