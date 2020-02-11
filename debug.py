import torch
from src import get_model, config
config.parse_args()

model = get_model()
model.topvae.decoder.allow_spreading()
print(model)
x = torch.randn(1, 8192)
print(model(x)[0].shape)