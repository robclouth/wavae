import torch
from src import Generator, Discriminator, Quantize, Encoder

enc = Encoder()
x  = torch.randn(1,1,8192)

zq, diff, idx = enc(x)

gen = Generator()

out = gen(zq)

dis = Discriminator()

for elm in dis(out):
    if isinstance(elm, list):
        for elm_ in elm:
            print(elm_.shape)
        print("\n")
    else:
        print(elm.shape)
        print("\n")