import torch
import torch.nn as nn
from . import config

SCRIPT = True


def cache_pad(*args):
    if SCRIPT:
        return torch.jit.script(CachedPadding(*args))
    else:
        return CachedPadding(*args)


def cache_pad_transpose(*args):
    if SCRIPT:
        return torch.jit.script(CachedPaddingTranspose(*args))
    else:
        return CachedPaddingTranspose(*args)


class CachedPadding(nn.Module):
    def __init__(self, padding, channels, cache=False):
        super().__init__()

        left_pad = torch.zeros(1, channels, 2 * padding)
        self.register_buffer("left_pad", left_pad)

        self.padding = padding
        self.cache = cache

    def forward(self, x):
        if self.cache:
            padded_x = torch.cat([self.left_pad, x], -1)
            self.left_pad = padded_x[..., -2 * self.padding:]
        else:
            padded_x = nn.functional.pad(x, (self.padding, self.padding))
        return padded_x

    def reset(self):
        self.left_pad.zero_()

    def __repr__(self):
        return f"CachedPadding(padding={self.padding}, cache={self.cache})"


class CachedPaddingTranspose(nn.Module):
    def __init__(self, padding, channels, cache=False):
        super().__init__()

        right_pad = torch.zeros(1, channels, 2 * padding)
        self.register_buffer("right_pad", right_pad)

        self.padding = 2 * padding
        self.cache = cache

    def forward(self, x):
        if self.cache:
            x[..., :self.padding] += self.right_pad
            self.right_pad = x[..., -self.padding:]
            current = x[..., :-self.padding]
        else:
            current = x[..., self.padding // 2:-self.padding // 2]
        return current

    def reset(self):
        self.right_pad.zero_()

    def __repr__(self):
        return f"CachedPaddingTranspose(padding={self.padding}, cache={self.cache})"
