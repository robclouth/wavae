import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from effortless_config import Config
from src import get_model, Discriminator, Loader, preprocess, config
from tqdm import tqdm
from os import path

config.parse_args()

# PREPARE DATA
dataset = Loader(config.LMDB_LOC)
if dataset.len is None:
    preprocess(config.WAV_LOC, config.SAMPRATE, config.LMDB_LOC,
               config.N_SIGNAL)
dataset = Loader(config.LMDB_LOC)
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=config.BATCH,
                                         shuffle=True,
                                         drop_last=True)

# PREPARE MODELS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gen = get_model()
dis = Discriminator()

if config.CKPT is not None:
    ckptgen, ckptdis = torch.load(config.CKPT, map_location="cpu")
    gen.load_state_dict(ckptgen)
    dis.load_state_dict(ckptdis)

gen = gen.to(device)
dis = dis.to(device)

# PREPARE OPTIMIZERS
opt_gen = torch.optim.Adam(gen.parameters(), lr=config.LR, betas=[.5, .9])
opt_dis = torch.optim.Adam(dis.parameters(), lr=config.LR, betas=[.5, .9])

ROOT = path.join("runs", config.NAME)
writer = SummaryWriter(ROOT, flush_secs=20)

# TRAINING PROCESS
step = 0
for e in range(config.EPOCH):
    for batch in tqdm(dataloader):
        batch = batch.to(device)

        if config.TYPE == "autoencoder":
            y, zq, diff, idx = gen(batch)
        elif config.TYPE == "melgan":
            y = gen(batch)

        # TRAIN DISCRIMINATOR
        D_fake = dis(y.detach())
        D_real = dis(batch)

        loss_D = 0

        for scale in D_fake:
            loss_D += torch.relu(1 + scale[-1]).mean()
        for scale in D_real:
            loss_D += torch.relu(1 - scale[-1]).mean()

        opt_dis.zero_grad()
        loss_D.backward()
        opt_dis.step()

        # TRAIN GENERATOR
        D_fake = dis(y)

        loss_G = 0
        for scale in D_fake:
            loss_G += -scale[-1].mean()

        loss_feat = 0
        feat_weights = 4.0 / (config.N_LAYER_D + 1)
        D_weights = 1.0 / config.NUM_D
        wt = D_weights * feat_weights
        for i in range(config.NUM_D):
            for j in range(len(D_fake[i]) - 1):
                loss_feat += wt * F.l1_loss(D_fake[i][j],
                                            D_real[i][j].detach())

        loss_complete = loss_G + 10 * loss_feat

        if config.TYPE == "autoencoder":
            loss_complete += .01 * diff
            writer.add_scalar("loss regularization", diff, step)

        opt_gen.zero_grad()
        loss_complete.backward()
        opt_gen.step()

        writer.add_scalar("loss discriminator", loss_D, step)
        writer.add_scalar("loss adversarial", loss_G, step)
        writer.add_scalar("loss features", loss_feat, step)

        if step % config.BACKUP == 0:
            backup_name = path.join(
                ROOT, f"{gen.__class__.__name__}_{step//1000}k.pth")
            states = [gen.state_dict(), dis.state_dict()]
            torch.save(states, backup_name)

        if step % config.EVAL == 0:
            writer.add_audio("original", batch.reshape(-1), step,
                             config.SAMPRATE)
            writer.add_audio("generated", y.reshape(-1), step, config.SAMPRATE)

        step += 1

backup_name = path.join(ROOT, f"{model.__class__.__name__}_final.pth")
states = [gen.state_dict(), dis.state_dict()]
torch.save(states, backup_name)
