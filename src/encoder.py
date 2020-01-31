import torch
import torch.nn as nn
from . import config, Quantize
import librosa as li
import numpy as np

module = lambda x: torch.sqrt(x[..., 0]**2 + x[..., 1]**2)


class MelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hop = int(np.prod(config.RATIOS))
        self.nfft = 2048

        mel = li.filters.mel(16000, self.nfft, config.INPUT_SIZE, fmin=80)
        mel = torch.from_numpy(mel)

        self.register_buffer("mel", mel)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)

        S = torch.stft(x, self.nfft, self.hop, 512)
        S = 2 * module(S) / 512
        S_mel = self.mel.matmul(S)[..., :x.shape[-1] // self.hop]
        return torch.log10(torch.clamp(S_mel, min=1e-5))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(config.CHANNELS[i],
                      config.CHANNELS[i + 1],
                      config.KERNEL,
                      stride=config.RATIOS[i],
                      dilation=config.DILATION[i],
                      padding=((config.KERNEL - 1) * config.DILATION[i] + 1) //
                      2) for i in range(len(config.RATIOS))
        ])

        self.quantizer = Quantize(config.CHANNELS[-1], config.N_EMBED)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs) - 1:
                x = torch.relu(x)
        x = x.permute(0, 2, 1)

        embed, diff, index = self.quantizer(x)
        embed = embed.permute(0, 2, 1)

        return embed, diff, index
