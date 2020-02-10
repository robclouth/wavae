from src import get_model, config
import torch
config.parse_args()

model = get_model()
print(model)

x = torch.randn(1, 128, config.BUFFER_SIZE // config.HOP_LENGTH)

print(model(x, mel_encoded=True).shape)