import librosa as li
import resampy
import lmdb
from tqdm import tqdm
from pathlib import Path
from os import path
import pickle
import torch


def preprocess(wavlocs, samprate, outdb, n_signal):
    env = lmdb.open(outdb, map_size=10e9, lock=False)

    wavs = []
    wavlocs = wavlocs.split(",")

    for wavloc in wavlocs:
        wavs += [str(elm.absolute()) for elm in Path(wavloc).rglob("*.wav")]
        wavs += [str(elm.absolute()) for elm in Path(wavloc).rglob("*.aif")]

    wavs = tqdm(wavs)

    with env.begin(write=True) as txn:
        idx = 0
        skip = 0
        for wav in wavs:
            wavs.set_description(path.basename(wav))
            try:
                x, sr = li.load(wav, None)
            except:
                skip += 1
                continue
            x = resampy.resample(x, sr, samprate)

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
        print(f"Skipped {skip} files during preprocess")
        txn.put("length".encode("utf-8"), pickle.dumps(idx))



class Loader(torch.utils.data.Dataset):
    def __init__(self, database, cat=1):
        super().__init__()
        self.env = lmdb.open(database, lock=False)
        with self.env.begin(write=False) as txn:
            self.len = txn.get("length".encode("utf-8"))
            if self.len is not None:
                self.len = pickle.loads(self.len)
        self.cat = cat

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        with self.env.begin(write=False) as txn:
            x = torch.cat([
                pickle.loads(txn.get(
                    f"{(i+t) % self.len:08d}".encode("utf-8")))
                for t in range(self.cat)
            ], -1)
        x = torch.from_numpy(x).unsqueeze(0)
        return x
