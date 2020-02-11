import torch
torch.set_grad_enabled(False)
import matplotlib.pyplot as plt
from src import get_model, config
config.parse_args()

model = get_model()
model.topvae.decoder.allow_spreading()
print(model)
x = torch.randn(1, 2**14)
print(x.shape)
y, mean_y, logvar_y, mean_z, logvar_z = model(x)