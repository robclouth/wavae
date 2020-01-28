import torch
import torch.nn as nn
from . import config, Quantize

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(
                config.CHANNELS[i],
                config.CHANNELS[i+1],
                config.KERNEL,
                stride=config.RATIOS[i],
                dilation=config.DILATION[i],
                padding=((config.KERNEL-1)*config.DILATION[i] + 1)//2
            ) for i in range(len(config.RATIOS))
        ])

        self.quantizer = Quantize(config.CHANNELS[-1], config.N_EMBED)
    
    def forward(self, x):
        for i,conv in enumerate(self.convs):
            x = conv(x)
            if i != len(self.convs) -1:
                x = torch.relu(x)
        x = x.permute(0,2,1)
        
        embed, diff, index = self.quantizer(x)
        embed = embed.permute(0,2,1)

        return embed, diff, index