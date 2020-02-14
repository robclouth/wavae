import torch
import torch.nn as nn
from . import config

SCRIPT = False


def cache_pad(*args):
    if SCRIPT:
        return torch.jit.script(CachedPadding(*args))
    else:
        return CachedPadding(*args)


class CachedPadding(nn.Module):
    """
    Cached padding (buffer based inference)

    Replace nn.Conv1d(C,x,x,padding=P) with

    nn.Sequential(
        CachedPadding(P, C, True),
        nn.Conv1d(C,x,x,padding=0)
    )

    And replace nn.ConvTranspose1d(C,x,2 * r,stride=r, padding=r//2) with

    nn.Sequential(
        CachedPadding(1, C, True),
        nn.ConvTranspose1d(C,x,2 * r,stride=r, padding=r//2 + r)
    )
    """
    def __init__(self, padding, channels, cache=False):
        super().__init__()
        self.padding = padding

        left_pad = torch.zeros(1, channels, padding)
        self.register_buffer("left_pad", left_pad)

        self.cache = cache

    def forward(self, x):
        if self.cache:
            padded_x = torch.cat([self.left_pad, x], -1)
            self.left_pad = padded_x[..., -self.padding:]
        else:
            padded_x = nn.functional.pad(
                x, (self.padding // 2, self.padding // 2))
        return padded_x

    def reset(self):
        self.left_pad.zero_()

    def __repr__(self):
        return f"CachedPadding(padding={self.padding}, cache={self.cache})"


class CachedConv1d(nn.Module):
    def __init__(self,
                 in_chan,
                 out_chan,
                 kernel,
                 stride,
                 padding,
                 dilation=(1, ),
                 cache=False):
        super().__init__()
        self.pas = cache_pad(2 * padding, in_chan, cache)
        self.conv = nn.Conv1d(in_chan,
                              out_chan,
                              kernel,
                              stride,
                              dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class CachedConvTranspose1d(nn.Module):
    def __init__(self,
                 in_chan,
                 out_chan,
                 kernel,
                 stride,
                 dilation=(1, ),
                 cache=False):
        super().__init__()
        assert kernel == 2 * stride, "WESH"
        self.cache = cache
        self.stride = stride
        self.pad = cache_pad(1, in_chan, cache)
        self.conv = nn.ConvTranspose1d(in_chan,
                                       out_chan,
                                       kernel_size=kernel,
                                       stride=stride,
                                       padding=0)

    def forward(self, x):
        if self.cache:
            x = self.pad(x)

        x = self.conv(x)

        if self.cache:
            x = x[..., self.stride:-self.stride]
        else:
            x = x[..., :-self.stride]

        return x