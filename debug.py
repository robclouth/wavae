import torch
from src import get_model, config
config.parse_args()

x = torch.randn(1, 8192)
model = get_model()
