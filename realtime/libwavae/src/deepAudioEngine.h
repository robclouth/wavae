#pragma once
#include <string>
#define BUFFERSIZE 512
#define SPREAD 403

class DeepAudioEngine {
public:
  virtual void addBuffer(float *buffer, int n) = 0;
  virtual void getBuffer(int n) = 0;
  virtual int load(std::string name) = 0;
  virtual int getLatentDimension() = 0;

protected:
  float previous[BUFFERSIZE], current[BUFFERSIZE], rightspread[SPREAD];
};