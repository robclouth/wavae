import torch
torch.set_grad_enabled(False)
import torch.nn as nn
from src import TopVAE, Generator, MelEncoder, config, get_model
from os import path
import importlib
from termcolor import colored

config.parse_args()

NAME = config.NAME
ROOT = path.join("runs", NAME)

config_melgan = ".".join(path.join(ROOT, "melgan", "config").split("/"))
config_vanilla = ".".join(path.join(ROOT, "vanilla", "config").split("/"))


class BufferSTFT(nn.Module):
    def __init__(self, buffer_size, hop_length):
        super().__init__()
        buffer = torch.zeros(1, 2048 + 31 * hop_length)
        self.register_buffer("buffer", buffer)
        self.buffer_size = buffer_size

    def forward(self, x):
        self.buffer = torch.roll(self.buffer, -self.buffer_size, -1)
        self.buffer[:, -self.buffer_size:] = x
        return self.buffer


class TracedMelEncoder(nn.Module):
    def __init__(self, melencoder, buffer, use_buffer=True):
        super().__init__()
        self.melencoder = melencoder
        self.buffer = torch.jit.script(buffer)
        self.use_buffer = use_buffer

    def forward(self, x):
        if self.use_buffer:
            x = self.buffer(x)
        return self.melencoder(x)


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()

        # BUILDING MELGAN #################################################
        hparams = importlib.import_module(config_melgan).config
        hparams.override(USE_CACHED_PADDING=config.USE_CACHED_PADDING)
        melgan = get_model(hparams)

        pretrained_state_dict = torch.load(path.join(ROOT, "melgan",
                                                     "melgan_state.pth"),
                                           map_location="cpu")[0]
        state_dict = melgan.state_dict()
        state_dict.update(pretrained_state_dict)
        melgan.load_state_dict(state_dict)
        ###################################################################

        # BUILDING VANILLA ################################################
        hparams = importlib.import_module(config_vanilla).config
        hparams.override(USE_CACHED_PADDING=config.USE_CACHED_PADDING)
        vanilla = get_model(hparams)

        pretrained_state_dict = torch.load(path.join(ROOT, "vanilla",
                                                     "vanilla_state.pth"),
                                           map_location="cpu")
        state_dict = vanilla.state_dict()
        state_dict.update(pretrained_state_dict)
        vanilla.load_state_dict(state_dict)
        ###################################################################

        vanilla.eval()
        melgan.eval()

        # PRETRACE MODELS #################################################
        self.latent_size = int(config.CHANNELS[-1] // 2)
        self.mel_size = int(config.CHANNELS[0])

        if config.USE_CACHED_PADDING:
            test_wav = torch.randn(1, config.BUFFER_SIZE)
            test_mel = torch.randn(1, config.INPUT_SIZE, 2)
            test_z = torch.randn(1, self.latent_size, 1)

        else:
            test_wav = torch.randn(1, 8192)
            test_mel = torch.randn(1, config.INPUT_SIZE, 16)
            test_z = torch.randn(1, self.latent_size, 16)

        melencoder = TracedMelEncoder(
            vanilla.melencoder,
            BufferSTFT(config.BUFFER_SIZE, config.HOP_LENGTH),
            config.USE_CACHED_PADDING)

        self.trace_melencoder = torch.jit.trace(melencoder,
                                                test_wav,
                                                check_trace=False)
        self.trace_encoder = torch.jit.trace(vanilla.topvae.encoder,
                                             test_mel,
                                             check_trace=False)
        self.trace_decoder = torch.jit.trace(vanilla.topvae.decoder,
                                             test_z,
                                             check_trace=False)
        self.trace_melgan = torch.jit.trace(melgan.decoder,
                                            test_mel,
                                            check_trace=False)

    def forward(self, x):
        return self.decode_wav(self.decode_mel(self.encode(x)))

    @torch.jit.export
    def melencode(self, x):
        return self.trace_melencoder(x)

    @torch.jit.export
    def encode(self, x):
        mel = self.melencode(x)
        z = self.trace_encoder(mel)
        z = torch.split(z, self.latent_size, 1)[0]
        return z

    @torch.jit.export
    def decode_mel(self, z):
        mel = torch.sigmoid(self.trace_decoder(z))
        mel = torch.split(mel, self.mel_size, 1)[0]
        return mel

    @torch.jit.export
    def decode_wav(self, mel):
        waveform = self.trace_melgan(mel)
        return waveform


if __name__ == "__main__":
    wrapper = Wrapper()
    print(wrapper)
    torch.jit.script(wrapper).save(path.join(ROOT, "trace_model.ts"))