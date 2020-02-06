from effortless_config import Config, setting


class config(Config):
    groups = ["ae", "mel"]

    TYPE = setting(default="vanilla", ae="vanilla", mel="melgan")

    # MELGAN PARAMETERS
    INPUT_SIZE = 128
    NGF = 32
    N_RES_G = 3
    RATIOS = setting(default=[1, 2, 1, 2], ae=[1, 2, 1, 2], mel=[8, 8, 2, 2])

    NUM_D = 3
    NDF = 16
    N_LAYER_D = 4
    DOWNSAMP_D = 4

    # AUTOENCODER
    CHANNELS = [128, 96, 64, 32, 16]
    KERNEL = 9

    # TRAIN PARAMETERS
    SAMPRATE = 16000
    N_SIGNAL = setting(default=32000, ae=32000, mel=2**14)
    EPOCH = 1000
    BATCH = 1
    LR = 1e-4
    NAME = "untitled"
    CKPT = None

    WAV_LOC = "/Users/caillon/dev/vae-rnn/test wav/demo alexander/hivae"
    LMDB_LOC = "./preprocessed"

    BACKUP = 10000
    EVAL = 1000


if __name__ == "__main__":
    print(config)