import torch
from . import Loader
from tqdm import tqdm


def compute_pca(model, lmdb_loc, batch_size):
    loader = Loader(lmdb_loc)
    dataloader = torch.utils.data.DataLoader(loader,
                                             batch_size=batch_size,
                                             drop_last=False,
                                             shuffle=True)

    z = []

    for elm in tqdm(dataloader, desc="parsing dataset..."):
        z_ = model.encode(elm.squeeze())  # SHAPE B x Z x T
        z_ = z_.permute(0, 2, 1).reshape(-1, z_.shape[1])  # SHAPE BT x Z
        z.append(z_)

    z = torch.cat(z, 0)
    z = z[torch.randperm(z.shape[0])].permute(1, 0)
    z = z[:10000]

    mean = torch.mean(z, -1, keepdim=True)
    std = 3 * torch.std(z)  # 99.7% of the range (normal law)
    U = torch.svd(z - mean, some=False)[0]
    return mean.reshape(1, 1, -1), std, U
