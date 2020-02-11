import torch
torch.set_grad_enabled(False)
from src import get_model, config
config.parse_args()

model = get_model()
x = torch.randn(1, 2048)

print(model(x).shape)

model.decoder.allow_spreading()
