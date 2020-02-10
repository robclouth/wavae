import torch
import torch.nn as nn


class CachedPadding(nn.Module):
    def __init__(self, padding, channels, buffer_size, cache=False):
        super().__init__()
        shape = (1, channels, buffer_size)

        left_pad = torch.zeros(1, channels, padding)
        previous_sample = torch.zeros(shape)

        self.register_buffer("left_pad", left_pad)
        self.register_buffer("previous_sample", previous_sample)

        self.padding = padding
        self.cache = cache

    def forward(self, x):
        if self.cache:
            # assert x.shape == self.previous_sample.shape, f"Incoherence of input and cached shapes, expected input of shape {self.previous_sample.shape}, got {x.shape}"
            current = torch.cat(
                [self.left_pad, self.previous_sample, x[..., :self.padding]],
                -1)
            self.left_pad = self.previous_sample[:, :, -self.padding:]
            self.previous_sample = x

        else:
            current = nn.functional.pad(x, (self.padding, self.padding))
        return current