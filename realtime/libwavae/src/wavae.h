#pragma once
#include "deepAudioEngine.h"
#include <torch/script.h>
#include <torch/torch.h>

class WaVAE : public DeepAudioEngine {
public:
  WaVAE();
  void addBuffer(float *buffer, int n) override;
  void getBuffer(int n) override;
  int load(std::string name) override;
  int getLatentDimension() override;

protected:
  torch::jit::script::Module melencoder, encoder, decoder;
  torch::Tensor inputbuffer;
};