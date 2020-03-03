from udls import SimpleDataset
import librosa as li
from . import config
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from scipy.special import erf
from tqdm import tqdm


def log_loudness(x, size, eps=1e-4):
    x_win = x.reshape(x.shape[0], -1, size)
    win = np.hanning(size)
    win /= np.mean(win)
    win = win.reshape(1, 1, -1)
    x_win = x_win * win
    log_rms = .5 * np.log(np.clip(np.mean(x_win**2, -1), eps, 1))
    log_rms = (np.log(eps) - 2 * log_rms) / np.log(eps)
    return log_rms.reshape(log_rms.shape[0], 1, -1)


def get_flattening_function(x, n_mixture=10):
    # FIT GMM ON DATA
    gmm = GaussianMixture(n_mixture).fit(x.reshape(-1, 1))
    weights, means, vars = gmm.weights_, gmm.means_, gmm.covariances_
    weights = weights.reshape(-1)
    means = means.reshape(-1)
    stds = np.sqrt(vars.reshape(-1))

    # COMPUTE CUMULATED DENSITY FUNCTION
    cdf = lambda x: np.sum([
        w * .5 * (1 + erf((x - m) / (s * np.sqrt(2))))
        for w, m, s in zip(weights, means, stds)
    ], 0)

    return cdf


def preprocess(name):
    try:
        x = li.load(name, config.SAMPRATE)[0]
    except KeyboardInterrupt:
        exit()
    except:
        return None

    border = len(x) % config.N_SIGNAL

    if len(x) < config.N_SIGNAL:
        x = np.pad(x, (0, config.N_SIGNAL - len(x)))

    elif border:
        x = x[:-border]

    x = x.reshape(-1, config.N_SIGNAL)

    if config.TYPE == "vanilla":
        log_rms = log_loudness(x, config.HOP_LENGTH * np.prod(config.RATIOS))
        x = zip(x, log_rms)

    return x


class Loader(torch.utils.data.Dataset):
    def __init__(self, cat, config=config):
        super().__init__()
        self.dataset = SimpleDataset(config.LMDB_LOC,
                                     config.WAV_LOC.split(","),
                                     preprocess,
                                     map_size=1e11)
        self.cat = cat

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if config.TYPE == "vanilla":
            sample = []
            loudness = []
            for i in range(self.cat):
                s, l = self.dataset[(idx + i) % self.__len__()]
                sample.append(torch.from_numpy(s).float())
                loudness.append(torch.from_numpy(l).float())
            sample = torch.cat(sample, -1)
            loudness = torch.cat(loudness, -1)
            return sample, loudness
        else:
            return self.dataset[idx][0]