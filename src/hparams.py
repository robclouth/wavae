from effortless_config import Config, setting

class config(Config):
    groups = ["ae", "mel"]

    TYPE       = setting(default="autoencoder",
                         ae="autoencoder",
                         mel="melgan")
    
    INPUT_SIZE = 128
    NGF        = 32
    N_RES_G    = 3
    RATIOS     = [8, 8, 2, 2]


    NUM_D      = 3
    NDF        = 16
    N_LAYER_D  = 4
    DOWNSAMP_D = 4

    N_EMBED    = 512

    CHANNELS   = [
        1, 32, 64, 128, 128
    ]
    KERNEL     = 9

    DILATION   = [
        2**(i%3) for i in range(4)
    ]

    # TRAIN PARAMETERS
    SAMPRATE = 16000
    N_SIGNAL = 8192
    EPOCH    = 1000
    BATCH    = 1
    LR       = 1e-4
    NAME     = "untitled"
    CKPT     = None

    WAV_LOC  = "/Users/caillon/dev/vae-rnn/test wav/demo alexander/hivae"
    LMDB_LOC = "./preprocessed"

    BACKUP   = 10000
    EVAL     = 1000

if __name__ == "__main__":
    print(config)