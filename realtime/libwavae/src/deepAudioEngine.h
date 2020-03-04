#pragma once
#include <string>
#include <vector>

#define BUFFERSIZE 2048
#define LATENT_NUMBER 17
#define DIM_REDUCTION_FACTOR 512

class DeepAudioEngine {
public:
  virtual void perform(float *in_buffer, float *out_buffer) = 0;
  virtual int load(std::string name) = 0;
};