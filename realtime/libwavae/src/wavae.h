#pragma once
#include "deepAudioEngine.h"
#include <torch/script.h>
#include <torch/torch.h>

namespace wavae {

class Encoder : public DeepAudioEngine {
public:
  Encoder();
  void perform(float *in_buffer, float *out_buffer);
  int load(std::string name);

protected:
  int model_loaded;
  torch::jit::script::Module model;
};

class Decoder : public DeepAudioEngine {
public:
  Decoder();
  void perform(float *in_buffer, float *out_buffer);
  int load(std::string name);

protected:
  int model_loaded;
  torch::jit::script::Module model;
};

} // namespace wavae
