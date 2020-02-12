#include "wavae_encoder.h"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>

WaVAE_ENCODER::WaVAE_ENCODER()
    : head_input_buffer(0), is_loaded(0), z_available(0) {
  torch::NoGradGuard no_grad_guard;
}

void WaVAE_ENCODER::encode(float *input) {
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

  auto out =
      encoder.forward(melencoder_out).toTensor().reshape({-1}).data<float>();

  memcpy(latent_out, out, N_LATENT * sizeof(float));
}

void WaVAE_ENCODER::addBuffer(float **buffer, int n_sample, int n_channel) {
  while (n_sample--) {
    input_buffer[head_input_buffer++] = *(buffer[0])++;
    if (head_input_buffer == BUFFERSIZE) {
      head_input_buffer = 0;
      encode(input_buffer);
      z_available = 1;
    }
  }
}

void WaVAE_ENCODER::getBuffer(float **buffer, int n_sample, int n_channel) {
  if (z_available) {
    for (int i(0); i < n_sample; i++) {
      for (int l(0); l < N_LATENT; l++) {
        buffer[l][i] = latent_out[l];
      }
    }
  } else {
    for (int i(0); i < n_sample; i++) {
      for (int l(0); l < N_LATENT; l++) {
        buffer[l][i] = 0;
      }
    }
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