#pragma once
#include <string>
#include <vector>

class DeepAudioEngine {
public:
  virtual void perform(std::vector<float *> in_buffer,
                       std::vector<float *> out_buffer, int n_in_channel,
                       int n_out_channel, int n_signal) = 0;
  virtual int load(std::string name) = 0;
  virtual int getInputChannelNumber() = 0;
  virtual int getOutputChannelNumber() = 0;
};