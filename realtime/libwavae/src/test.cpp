#include "wavae.h"
#include <dlfcn.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

int main(int argc, char const *argv[]) {

  DeepAudioEngine *encoder = new wavae::Encoder;
  int error = encoder->load("trace_model.ts");

  DeepAudioEngine *decoder = new wavae::Decoder;
  error = decoder->load("trace_model.ts");

  std::vector<float *> input_buffer, latent_buffer, output_buffer;

  input_buffer.push_back(new float[4096]);
  output_buffer.push_back(new float[4096]);

  for (int i(0); i < encoder->getOutputChannelNumber(); i++) {
    latent_buffer.push_back(new float[4096]);
  }

  // LOOP TEST
  for (int i(0); i < 100; i++) {
    std::cout << i << std::endl;
    encoder->perform(input_buffer, latent_buffer, 1, 16, 4096);
    decoder->perform(latent_buffer, output_buffer, 16, 1, 4096);
  }

  return 0;
}
