from udls import SimpleDataset
import librosa as li
from . import config
import numpy as np
import torch


def preprocess(name):
    try:
        x = li.load(name, config.SAMPRATE)[0]
    except:
        return None

    border = len(x) % config.N_SIGNAL

    if len(x) < config.N_SIGNAL:
        return None

    elif border:
        x = x[:-border]

    return x.reshape(-1, config.N_SIGNAL)


class Loader(torch.utils.data.Dataset):
    def __init__(self, cat):
        super().__init__()
        self.dataset = SimpleDataset(config.LMDB_LOC,
                                     config.WAV_LOC.split(","),
                                     preprocess,
                                     map_size=1e11)
        self.cat = cat

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return torch.cat([
            torch.from_numpy(self.dataset[(idx + i) % self.__len__()]).float()
            for i in range(self.cat)
        ], -1)
