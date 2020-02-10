from effortless_config import Config, setting


class config(Config):
    groups = ["vanilla", "melgan"]

    TYPE = setting(default="vanilla", vanilla="vanilla", melgan="melgan")

    # MELGAN PARAMETERS
    INPUT_SIZE = 128
    NGF = 32
    N_RES_G = 3

    HOP_LENGTH = 256

    RATIOS = setting(default=[1, 2, 1, 2],
                     vanilla=[1, 1, 1, 2],
                     melgan=[8, 8, 2, 2])

    NUM_D = 3
    NDF = 16
    N_LAYER_D = 4
    DOWNSAMP_D = 4

    # AUTOENCODER
    CHANNELS = [128, 96, 64, 32, 16]
    KERNEL = 3

    # TRAIN PARAMETERS
    SAMPRATE = 16000
    N_SIGNAL = setting(default=2**15, vanilla=2**15, melgan=2**14)
    EPOCH = 1000
    BATCH = 1
    LR = 1e-4
    NAME = "untitled"
    CKPT = None

    WAV_LOC = "/Users/caillon/dev/vae-rnn/test wav/demo alexander/hivae"
    LMDB_LOC = "./preprocessed"

    BACKUP = 10000
    EVAL = 1000

    # INCREMENTAL GENERATION
    BUFFER_SIZE = 1024
    USE_CACHED_PADDING = False


if __name__ == "__main__":
    print(config)
