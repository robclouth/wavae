from effortless_config import Config

class config(Config):
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

if __name__ == "__main__":
    print(config)