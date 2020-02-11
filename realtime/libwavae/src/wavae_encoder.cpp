#include "wavae_encoder.h"
#include <algorithm>
#include <iostream>
#include <vector>

WaVAE_ENCODER::WaVAE_ENCODER() : head_input_buffer(0), is_loaded(0) {
  torch::NoGradGuard no_grad_guard;
}

float *WaVAE_ENCODER::encode(float *input) {
  // FILL TENSOR WITH INPUT BUFFER
  torch::Tensor waveform = torch::zeros({BUFFERSIZE});
  for (int i(0); i < BUFFERSIZE; i++) {
    waveform[i] = input_buffer[i];
  }
  waveform = waveform.reshape({1, -1});

  // PREPARE INPUTS TO BE ENCODED
  std::vector<torch::jit::IValue> inputs, melencoder_out;
  inputs.push_back(waveform);

  // ENCODING PROCESS
  auto mel = melencoder.forward(inputs);
  melencoder_out.push_back(mel);

  float *z =
      encoder.forward(melencoder_out).toTensor().reshape({-1}).data<float>();

  return z;
}

void WaVAE_ENCODER::addBuffer(float *buffer, int n) {
  while (n--) {
    input_buffer[head_input_buffer++] = *buffer++;
    if (head_input_buffer == BUFFERSIZE) {
      head_input_buffer = 0;
      latent_out = encode(input_buffer);
    }
  }
}

void WaVAE_ENCODER::getBuffer(float *buffer, int n) {
  // n = LATENT_DIM * BUFFER_SIZE
  for (int i(0); i < n; i++) {
    buffer[i] = latent_out[i / BUFFERSIZE];
  }
}

int WaVAE_ENCODER::load(std::string name) {
  std::string path_encoder, path_melencoder, path_decoder;

  path_encoder = name + "/" + "encoder_trace.ts";
  path_decoder = name + "/" + "decoder_trace.ts";
  path_melencoder = name + "/" + "melencoder_trace.ts";

  std::cout << path_melencoder << std::endl;
  std::cout << path_encoder << std::endl;

  try {
    melencoder = torch::jit::load(path_melencoder);
    encoder = torch::jit::load(path_encoder);
    is_loaded = 1;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    is_loaded = 0;
    return 1;
  }
}

int WaVAE_ENCODER::getInputChannelNumber() { return 1; }

int WaVAE_ENCODER::getOutputChannelNumber() { return N_LATENT; }