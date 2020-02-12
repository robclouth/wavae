#pragma once
#include "deepAudioEngine.h"
#include <torch/script.h>
#include <torch/torch.h>

#define BUFFERSIZE 512
#define N_LATENT 16
#define SPREAD 403

class WaVAE_ENCODER : public DeepAudioEngine {
public:
  WaVAE_ENCODER();
  void addBuffer(float **buffer, int n_sample, int n_channel) override;
  void getBuffer(float **buffer, int n_sample, int n_channel) override;
  int load(std::string name) override;
  int getInputChannelNumber() override;
  int getOutputChannelNumber() override;

  void encode(float *input);

protected:
  torch::jit::script::Module melencoder, encoder;
  float input_buffer[BUFFERSIZE], *latent_out;
  int head_input_buffer;
  int is_loaded, z_available;
};