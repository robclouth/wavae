#include "wavae.h"
#include <iostream>
#include <stdlib.h>

// ENCODER /////////////////////////////////////////////////////////

wavae::Encoder::Encoder() { at::init_num_threads(); }

void wavae::Encoder::perform(float *in_buffer, float *out_buffer) {
  torch::NoGradGuard no_grad;

  std::vector<torch::jit::IValue> input;
  input.push_back(
      torch::from_blob(in_buffer, {1, BUFFERSIZE}).clone().to(torch::kFloat32));

  auto out =
      model.get_method("encode")(std::move(input)).toTensor().data<float>();

  memcpy(out_buffer, out, LATENT_NUMBER * BUFFERSIZE / DIM_REDUCTION_FACTOR);
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

// DECODER /////////////////////////////////////////////////////////

wavae::Decoder::Decoder() { at::init_num_threads(); }

void wavae::Decoder::perform(float *in_buffer, float *out_buffer) {

  torch::NoGradGuard no_grad;

  std::vector<torch::jit::IValue> input;
  input.push_back(
      torch::from_blob(in_buffer,
                       {1, LATENT_NUMBER, BUFFERSIZE / DIM_REDUCTION_FACTOR})
          .clone()
          .to(torch::kFloat32));

  auto out =
      model.get_method("decode")(std::move(input)).toTensor().data<float>();

  memcpy(out_buffer, out, BUFFERSIZE);
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

extern "C" {
DeepAudioEngine *get_encoder() { return new wavae::Encoder; }
DeepAudioEngine *get_decoder() { return new wavae::Decoder; }
}