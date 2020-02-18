#include "wavae.h"
#include <iostream>
#include <stdlib.h>

#define DEVICE torch::kCUDA

// ENCODER /////////////////////////////////////////////////////////

wavae::Encoder::Encoder() { at::init_num_threads(); }

void wavae::Encoder::perform(float *in_buffer, float *out_buffer) {
  torch::NoGradGuard no_grad;

  auto tensor = torch::from_blob(in_buffer, {1, BUFFERSIZE});
  tensor = tensor.to(DEVICE);

  std::vector<torch::jit::IValue> input;
  input.push_back(tensor);

  auto out_tensor = model.get_method("encode")(std::move(input)).toTensor();

  out_tensor = out_tensor.repeat_interleave(DIM_REDUCTION_FACTOR);
  out_tensor = out_tensor.to(torch::kCPU);

  auto out = out_tensor.contiguous().data<float>();

  for (int i(0); i < LATENT_NUMBER * BUFFERSIZE; i++) {
    out_buffer[i] = out[i];
  }
}

int wavae::Encoder::load(std::string name) {
  try {
    model = torch::jit::load(name);
    model.to(DEVICE);
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

  auto tensor = torch::from_blob(in_buffer, {1, LATENT_NUMBER, BUFFERSIZE});
  tensor =
      tensor.reshape({1, LATENT_NUMBER, -1, DIM_REDUCTION_FACTOR}).mean(-1);
  tensor = tensor.to(DEVICE);

  std::vector<torch::jit::IValue> input;
  input.push_back(tensor);

  auto out_tensor = model.get_method("decode")(std::move(input))
                        .toTensor()
                        .reshape({-1})
                        .contiguous();

  out_tensor = out_tensor.to(torch::kCPU);

  auto out = out_tensor.data<float>();

  for (int i(0); i < BUFFERSIZE; i++) {
    out_buffer[i] = out[i];
  }
}

int wavae::Decoder::load(std::string name) {
  try {
    model = torch::jit::load(name);
    model.to(DEVICE);
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