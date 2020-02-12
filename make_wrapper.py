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
        buffer = torch.zeros(1, 2048 + hop_length)
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

        # BUILDING MELGAN
        hparams = importlib.import_module(config_melgan).config
        hparams.override(USE_CACHED_PADDING=config.USE_CACHED_PADDING)
        self.melgan = get_model(hparams)

        pretrained_state_dict = torch.load(path.join(ROOT, "melgan",
                                                     "melgan_state.pth"),
                                           map_location="cpu")[0]
        pretrained_state_dict = {
            k: v
            for k, v in pretrained_state_dict.items()
            if ("previous_sample" not in k) and ("left_pad" not in k)
        }

        state_dict = self.melgan.state_dict()
        state_dict.update(pretrained_state_dict)
        self.melgan.load_state_dict(state_dict)

        # BUILDING VANILLA
        hparams = importlib.import_module(config_vanilla).config
        hparams.override(USE_CACHED_PADDING=config.USE_CACHED_PADDING)
        self.vanilla = get_model(hparams)

        pretrained_state_dict = torch.load(path.join(ROOT, "vanilla",
                                                     "vanilla_state.pth"),
                                           map_location="cpu")
        pretrained_state_dict = {
            k: v
            for k, v in pretrained_state_dict.items()
            if ("previous_sample" not in k) and ("left_pad" not in k)
        }

        state_dict = self.vanilla.state_dict()
        state_dict.update(pretrained_state_dict)
        self.vanilla.load_state_dict(state_dict)

        self.vanilla.eval()
        self.melgan.eval()

        if config.USE_CACHED_PADDING:
            self.vanilla.topvae.decoder.allow_spreading()
            self.melgan.decoder.allow_spreading()

        if config.USE_CACHED_PADDING:
            self.tracedmelencoder = torch.jit.trace(
                TracedMelEncoder(
                    self.vanilla.melencoder,
                    BufferSTFT(config.BUFFER_SIZE, config.HOP_LENGTH), True),
                torch.randn(1, config.BUFFER_SIZE))
        else:
            self.tracedmelencoder = torch.jit.trace(
                TracedMelEncoder(
                    self.vanilla.melencoder,
                    BufferSTFT(config.BUFFER_SIZE, config.HOP_LENGTH), False),
                torch.randn(1, 8192))

    def encode(self, x):
        mel = self.tracedmelencoder(x)
        z = self.vanilla.topvae.encode(mel)[1]
        return z

    def decode(self, z):
        mel = self.vanilla.topvae.decode(z)[1]
        waveform = self.melgan(mel, mel_encoded=True)
        return waveform


class Encoder(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, x):
        return self.wrapper.encode(x)


class Decoder(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, x):
        return self.wrapper.decode(x)


if __name__ == "__main__":
    wrapper = Wrapper()
    wrapper.eval()

    encoder = Encoder(wrapper)
    encoder.eval()

    decoder = Decoder(wrapper)
    decoder.eval()

    if config.USE_CACHED_PADDING:
        N = config.BUFFER_SIZE
    else:
        N = 8192

    input_waveform = torch.randn(1, N)

    # CHECK THAT EVERYTHING WORKS
    print(colored("TESTING MODEL"))
    print("\tencoding waveform... ", end="")
    z = wrapper.encode(input_waveform)
    print(colored(f"shape {z.shape}", "green"))
    print("\tdecoding latent... ", end="")
    y = wrapper.decode(z)
    print(colored(f"shape {y.shape}", "green"))
    print(colored("Successfuly reconstructed input !", "green"))

    # SCRIPTING TIME
    print(colored("Tracing model... ", "green"), end="")

    encoder_trace = torch.jit.trace(encoder, input_waveform, check_trace=False)
    decoder_trace = torch.jit.trace(decoder, z, check_trace=False)

    encoder_trace(input_waveform)
    decoder_trace(z)

    encoder_trace.save(path.join(ROOT, "encoder_trace.ts"))
    decoder_trace.save(path.join(ROOT, "decoder_trace.ts"))

    print(colored("Done !", "green"))
