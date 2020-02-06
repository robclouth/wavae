# WaVAE

Cause I love naming models <3

Despite its name, its not a waveform based VAE, but a melspec one with a melGAN decoder. That's actually pretty cool ! You can even use it on your dataset, and train the whole thing in maybe 2 days on a single GPU.

This model has realtime generation (on CPU) and a highly-compressed and expressive latent representation.

## Usage

Train the spectral model
```bash
python train.py -c vanilla --wav-loc YOUR_DATA_FOLDER --name ENTER_A_COOL_NAME
```

Remember to delete the `preprocessed` folder between each training, as the models don't use the same preprocessing pipeline.

Train the waveform model
```bash
python train.py -c melgan --wav-loc YOUR_DATA_FOLDER --name ENTER_THE_SAME_COOL_NAME
```

The training scripts logs into the `runs` folder, you can visualize it using `tensorboard`.


Onced both models are trained, trace them using
```bash
python make_wrapper.py --name AGAIN_THE_SAME_COOL_NAME
```

It will produce two traced scripts in `runs/COOL_NAME/*.ts`. Those scripts can be deployed, used in a libtorch C++ environement, inside a Max/MSP playground that won't be named here, without having to use the source code

**Using Python**

```python
import torch

encoder = torch.jit.load("runs/cool/encoder_trace.ts")
decoder = torch.jit.load("runs/cool/decoder_trace.ts")

x = torch.randn(1,8192) # an audio signal

latent = encoder(x)
reconstruction = decoder(latent)
```

**Using C++**

```c++
#include <torch/script.h>
#include <vector>

int main(){
    torch::jit::script::Module encoder, decoder;

    encoder = torch::jit::load("runs/cool/encoder_trace.ts");
    decoder = torch::jit::load("runs/cool/decoder_trace.ts");

    std::vector<torch::jit::IValue> x;
    x.push_back(torch::zeros({1,8192})); // an audio signal

    auto latent = encoder.forward(x);
    auto reconstructon = decoder.forward(latent);
}
``` 