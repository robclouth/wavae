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
  void perform(std::vector<float *> in_buffer, std::vector<float *> out_buffer,
               int n_in_channel, int n_out_channel, int n_signal);
  int load(std::string name);
  int getInputChannelNumber();
  int getOutputChannelNumber();

protected:
  torch::jit::script::Module model;
};

class Decoder : public DeepAudioEngine {
public:
  Decoder();
  void perform(std::vector<float *> in_buffer, std::vector<float *> out_buffer,
               int n_in_channel, int n_out_channel, int n_signal);
  int load(std::string name);
  int getInputChannelNumber();
  int getOutputChannelNumber();

protected:
  torch::jit::script::Module model;
};

} // namespace wavae
