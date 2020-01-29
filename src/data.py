import librosa as li
import lmdb
from tqdm import tqdm
from glob import glob
from os import path
import pickle
import torch

def preprocess(wavloc, samprate, outdb, n_signal):
    env = lmdb.open(outdb, map_size=10e9, lock=False)
    wavloc = wavloc if "." in wavloc else [path.join(wavloc, ext) for ext in ["*.wav", "*.aif"]]
    wavs = []
    for wav in wavloc:
        wavs.extend(glob(wav))

    wavs = tqdm(wavs)

    with env.begin(write=True) as txn:
        idx = 0
        for wav in wavs:
            wavs.set_description(path.basename(wav))
            x = li.load(wav, samprate)[0]
            N = len(x) // n_signal
            if N == 0:
                continue
            if len(x) % n_signal != 0:
                x = x[:-(len(x) % n_signal)].reshape([N, n_signal])
            else:
                x = x.reshape([N, n_signal])
            
            
            for elm in x:
                txn.put(f"{idx:08d}".encode("utf-8"), pickle.dumps(elm))
                idx += 1
        txn.put("length".encode("utf-8"), pickle.dumps(idx))

class Loader(torch.utils.data.Dataset):
    def __init__(self, database):
        super().__init__()
        self.env = lmdb.open(database, lock=False)
        with self.env.begin(write=False) as txn:
            self.len = txn.get("length".encode("utf-8"))
            if self.len is not None:
                self.len = pickle.loads(self.len)   

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        with self.env.begin(write=False) as txn:
            x = pickle.loads(txn.get(f"{i:08d}".encode("utf-8")))
        x = torch.from_numpy(x).unsqueeze(0)
        return x
