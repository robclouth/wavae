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

        self.stft_buffer = torch.jit.script(
            BufferSTFT(config.BUFFER_SIZE, config.HOP_LENGTH))

        if config.USE_CACHED_PADDING:
            self.vanilla.topvae.decoder.allow_spreading()
            self.melgan.decoder.allow_spreading()

    def forward(self, x):
        y, mean_y, logvar_y, mean_z, logvar_z = self.vanilla(x)
        rec_waveform = self.melgan(y, mel_encoded=True)
        z = torch.randn_like(mean_z) * torch.exp(logvar_z) + mean_z
        return rec_waveform, z

    @torch.jit.export
    def encode(self, x):
        if config.USE_CACHED_PADDING:
            x = self.stft_buffer(x)
        mel = self.vanilla.melencoder(x)
        z = self.vanilla.topvae.encode(mel)[1]
        return z

    @torch.jit.export
    def decode(self, z):
        mel = self.vanilla.topvae.decode(z)[0]
        waveform = self.melgan(mel, mel_encoded=True)
        return waveform


if __name__ == "__main__":
    wrapper = Wrapper()
    wrapper.eval()

    if config.USE_CACHED_PADDING:
        N = config.BUFFER_SIZE
    else:
        N = 8192

    input_waveform = torch.randn(1, N)

    # CHECK THAT EVERYTHING WORKS
    print("Testing wrapper...")
    print("\tencoding waveform... ", end="")
    z = wrapper.encode(input_waveform)
    print(colored(f"shape {z.shape}", "green"))
    print("\tdecoding latent... ", end="")
    y = wrapper.decode(z)
    print(colored(f"shape {y.shape}", "green"))

    print(colored("Successfuly reconstructed input !", "green"))
    print("Tracing model...")

    torch.jit.trace(wrapper, input_waveform)