#pragma once
#include <string>
#include <vector>

class DeepAudioEngine {
public:
  virtual void perform(float *in_buffer, float *out_buffer) = 0;
  virtual int load(std::string name) = 0;
};