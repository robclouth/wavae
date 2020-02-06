import torch
torch.set_grad_enabled(False)
import torch.nn as nn
from src import TopVAE, Generator, MelEncoder, config, get_model
from os import path
import importlib

config.parse_args()

NAME = config.NAME
ROOT = path.join("runs", NAME)

config_melgan = ".".join(path.join(ROOT, "melgan", "config").split("/"))
config_vanilla = ".".join(path.join(ROOT, "vanilla", "config").split("/"))


class Wrapper(nn.Module):
    def __init__(self):
        super().__init__()

        config = importlib.import_module(config_melgan).config
        self.melgan = get_model(config)
        self.melgan.load_state_dict(
            torch.load(path.join(ROOT, "melgan", "melgan_state.pth"),
                       map_location="cpu")[0])

        config = importlib.import_module(config_vanilla).config
        self.vanilla = get_model(config)
        self.vanilla.load_state_dict(
            torch.load(path.join(ROOT, "vanilla", "vanilla_state.pth"),
                       map_location="cpu"))

    def forward(self, x):
        y, mean_y, logvar_y, mean_z, logvar_z = self.vanilla(x)
        rec_waveform = self.melgan(y, mel_encoded=True)

        z = torch.randn_like(mean_z) * torch.exp(logvar_z) + mean_z

        return rec_waveform, z


if __name__ == "__main__":
    wrapper = Wrapper()
    wrapper.eval()

    x = torch.randn(1, 8192)

    torch.jit.trace(wrapper, x, check_trace=False).save(f"{NAME}_traced.ts")