#pragma once
#include "deepAudioEngine.h"
#include <torch/script.h>
#include <torch/torch.h>

#define BUFFERSIZE 4096
#define LATENT_NUMBER 16
#define DIM_REDUCTION_FACTOR 512

namespace wavae {

class Encoder : public DeepAudioEngine {
public:
  Encoder();
  void perform(float *in_buffer, float *out_buffer);
  int load(std::string name);

protected:
  torch::jit::script::Module model;
};

class Decoder : public DeepAudioEngine {
public:
  Decoder();
  void perform(float *in_buffer, float *out_buffer);
  int load(std::string name);

protected:
  torch::jit::script::Module model;
};

} // namespace wavae
