from src import Loader, get_model
import torch
torch.set_grad_enabled(False)
import matplotlib.pyplot as plt

model = get_model()  # top vae
l = Loader("./preprocessed")

data = torch.stack([l[0], l[10], l[20]], 0)

mel = model.melencoder(data)

plt.imshow(mel[0])
plt.colorbar()
plt.show()