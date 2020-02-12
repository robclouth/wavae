#pragma once
#include <string>

class DeepAudioEngine {
public:
  virtual void addBuffer(float **buffer, int n_sample, int n_channel) = 0;
  virtual void getBuffer(float **buffer, int n_sample, int n_channel) = 0;
  virtual int load(std::string name) = 0;
  virtual int getInputChannelNumber() = 0;
  virtual int getOutputChannelNumber() = 0;
};