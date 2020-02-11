#include "wavae_encoder.h"

extern "C" {
DeepAudioEngine *build_encoder() { return new WaVAE_ENCODER; }
}