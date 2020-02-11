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

    def forward(self, x):
        y, mean_y, logvar_y, mean_z, logvar_z = self.vanilla(x)
        rec_waveform = self.melgan(y, mel_encoded=True)

        z = torch.randn_like(mean_z) * torch.exp(logvar_z) + mean_z

        return rec_waveform, z


class MelEncoderWrapper(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper.vanilla

    def forward(self, x):
        mel = self.wrapper.melencoder(x)
        return mel


class EncoderWrapper(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper.vanilla

    def forward(self, x):
        z = self.wrapper.topvae.encode(x)[1]
        return z


class DecoderWrapper(nn.Module):
    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper

    def forward(self, z):
        mel = self.wrapper.vanilla.topvae.decode(z)[0]
        waveform = self.wrapper.melgan(mel, mel_encoded=True)
        return waveform


if __name__ == "__main__":
    wrapper = Wrapper()
    wrapper.eval()

    melencoder = MelEncoderWrapper(wrapper)
    melencoder.eval()

    encoder = EncoderWrapper(wrapper)
    encoder.eval()

    decoder = DecoderWrapper(wrapper)
    decoder.eval()

    input_waveform = torch.randn(1, config.BUFFER_SIZE)

    # CHECK THAT EVERYTHING WORKS
    print("Checking melencoder... ", end="")
    mel = melencoder(input_waveform)[..., :config.BUFFER_SIZE //
                                     config.HOP_LENGTH]
    print(colored(f"melencoder is working, out shape {mel.shape}", "green"))

    print("Checking encoder... ", end="")
    input_z = encoder(mel)
    print(colored(f"encoder is working, out shape {input_z.shape}", "green"))

    print("Checking decoder... ", end="")
    rec = decoder(input_z)
    print(colored(f"decoder is working, out shape {rec.shape}", "green"))

    # TRACING TIME
    torch.jit.trace(melencoder, input_waveform, check_trace=False).save(
        path.join(ROOT, "melencoder_trace.ts"))

    torch.jit.trace(encoder, mel,
                    check_trace=False).save(path.join(ROOT,
                                                      "encoder_trace.ts"))
    torch.jit.trace(decoder, input_z,
                    check_trace=False).save(path.join(ROOT,
                                                      "decoder_trace.ts"))

    print("Traced wrapper created !")
