import torch.nn as nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np

from . import config, cache_pad


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, buffer_size=None):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            cache_pad(dilation, dim, buffer_size, True)
            if config.USE_CACHED_PADDING else nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)
        self.dilation = dilation

    def forward(self, x):
        # print(x.shape)
        # print(self.block)
        # print(self.shortcut)
        blockout = self.block(x)
        shortcut = self.shortcut(x)
        return blockout + shortcut


class Generator(nn.Module):
    def __init__(self,
                 input_size=config.INPUT_SIZE,
                 ngf=config.NGF,
                 n_residual_layers=config.N_RES_G,
                 ratios=config.RATIOS,
                 use_cached_padding=config.USE_CACHED_PADDING):

        super().__init__()
        self.hop_length = np.prod(ratios)
        mult = int(2**len(ratios))

        buf = int(config.BUFFER_SIZE // self.hop_length)

        model = [
            cache_pad(3, input_size, buf, True)
            if use_cached_padding else nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            buf = buf * r

            for j in range(n_residual_layers):
                model += [
                    ResnetBlock(mult * ngf // 2,
                                dilation=3**j,
                                buffer_size=buf)
                ]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            cache_pad(3, ngf, buf, True)
            if use_cached_padding else nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        # self.model = nn.ModuleList(model)
        self.apply(weights_init)

    def forward(self, x):
        # x = self.model(x)
        # for elm in self.model:
        #     print(elm.__class__.__name__)
        #     x = elm(x)
        return x


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(nf,
                                                      1,
                                                      kernel_size=3,
                                                      stride=1,
                                                      padding=1)

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self,
                 num_D=config.NUM_D,
                 ndf=config.NDF,
                 n_layers=config.N_LAYER_D,
                 downsampling_factor=config.DOWNSAMP_D):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor)

        self.downsample = nn.AvgPool1d(4,
                                       stride=2,
                                       padding=1,
                                       count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results
