#include "wavae.h"
#include <iostream>
#include <vector>

WaVAE::WaVAE() {
  inputbuffer = torch::zeros({
      1,
  })
}

void WaVAE::addBuffer(float *buffer, int n) {}

void WaVAE::getBuffer(int n) {}

int WaVAE::load(std::string name) {
  std::string path_encoder, path_melencoder, path_decoder;

  path_encoder = name + "/" + "encoder_trace.ts";
  path_decoder = name + "/" + "decoder_trace.ts";
  path_melencoder = name + "/" + "melencoder_trace.ts";

  std::cout << path_encoder << std::endl;
  std::cout << path_decoder << std::endl;
  std::cout << path_melencoder << std::endl;

  try {
    encoder = torch::jit::load(path_encoder);
    decoder = torch::jit::load(path_decoder);
    melencoder = torch::jit::load(path_melencoder);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

int WaVAE::getLatentDimension() { return 16; }

extern "C" {
DeepAudioEngine *maker() { return new WaVAE; }
}