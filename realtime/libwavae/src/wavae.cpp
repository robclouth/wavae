#include "wavae.h"
#include <iostream>
#include <stdlib.h>

// ENCODER /////////////////////////////////////////////////////////

wavae::Encoder::Encoder() { at::init_num_threads(); }

void wavae::Encoder::perform(std::vector<float *> in_buffer,
                             std::vector<float *> out_buffer, int n_in_channel,
                             int n_out_channel, int n_signal) {
  torch::NoGradGuard no_grad;
  // CREATE INPUT
  torch::Tensor buffer = torch::zeros({n_signal});

  for (int i(0); i < n_signal; i++) {
    buffer[i] = in_buffer[0][i];
  }
  buffer = buffer.unsqueeze(0);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(buffer);

  // ENCODE INPUT

  auto out = model.get_method("encode")(std::move(inputs))
                 .toTensor()
                 .squeeze(0)
                 .data<float>();

  // UPLOAD LATENT PATH TO BUFFER
  for (int c(0); c < LATENT_NUMBER; c++) {
    for (int i(0); i < n_signal; i++) {
      out_buffer[c][i] = out[(c * n_signal + i) / DIM_REDUCTION_FACTOR];
    }
  }
}

int wavae::Encoder::load(std::string name) {
  try {
    model = torch::jit::load(name);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

int wavae::Encoder::getInputChannelNumber() { return 1; }

int wavae::Encoder::getOutputChannelNumber() { return LATENT_NUMBER; }

// DECODER /////////////////////////////////////////////////////////

wavae::Decoder::Decoder() { at::init_num_threads(); }

void wavae::Decoder::perform(std::vector<float *> in_buffer,
                             std::vector<float *> out_buffer, int n_in_channel,
                             int n_out_channel, int n_signal) {
  torch::NoGradGuard no_grad;

  // CREATE INPUT
  torch::Tensor buffer =
      torch::zeros({n_in_channel, n_signal / DIM_REDUCTION_FACTOR});

  for (int c(0); c < n_in_channel; c++) {
    for (int i(0); i < n_signal / DIM_REDUCTION_FACTOR; i++) {
      buffer[c][i] = in_buffer[c][i * DIM_REDUCTION_FACTOR];
    }
  }

  buffer = buffer / DIM_REDUCTION_FACTOR;
  buffer = buffer.unsqueeze(0);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(buffer);

  // DECODE INPUT

  auto out = model.get_method("decode")(std::move(inputs))
                 .toTensor()
                 .squeeze()
                 .data<float>();

  // UPLOAD WAVEFORM TO BUFFER
  memcpy(out_buffer[0], out, n_signal * sizeof(float));
}

int wavae::Decoder::load(std::string name) {
  try {
    model = torch::jit::load(name);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
int wavae::Decoder::getInputChannelNumber() { return LATENT_NUMBER; }

int wavae::Decoder::getOutputChannelNumber() { return 1; }

extern "C" {
DeepAudioEngine *get_encoder() { return new wavae::Encoder; }
DeepAudioEngine *get_decoder() { return new wavae::Decoder; }
}