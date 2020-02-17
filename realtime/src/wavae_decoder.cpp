#include "../libwavae/src/deepAudioEngine.h"
#include "m_pd.h"
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <vector>

#define LATENT_NUMBER 16
#define BUFFER_SIZE 4096

static t_class *wavae_decoder_tilde_class;

typedef struct _wavae_decoder_tilde {
  t_object x_obj;
  t_sample f;

  DeepAudioEngine *model;
  std::vector<float *> in_buffer, out_buffer;
  std::thread *worker;

  t_inlet *x_in[LATENT_NUMBER - 1];
  t_outlet *x_out;
} t_wavae_decoder_tilde;

void perform(t_wavae_decoder_tilde *x, int n_signal) {
  x->model->perform(x->in_buffer, x->out_buffer, x->in_buffer.size(),
                    x->out_buffer.size(), n_signal);
}

t_int *wavae_decoder_tilde_perform(t_int *w) {

  t_wavae_decoder_tilde *x = (t_wavae_decoder_tilde *)w[1];
  std::vector<float *> input, output;

  // WAIT FOR THREAD TO END
  if (x->worker) {
    x->worker->join();
  }

  // PUT INPUT AND OUT BUFFER INTO VECTORS
  for (int i(0); i < LATENT_NUMBER; i++) {
    input.push_back((float *)(w[i + 2]));
  }
  output.push_back((float *)(w[LATENT_NUMBER + 2]));

  int n = (int)(w[LATENT_NUMBER + 3]);

  // COPY PREVIOUS COMPUTATIONS
  for (int t(0); t < n; t++) {
    output[0][t] = x->out_buffer[0][t];
  }

  // COPY CURRENT BUFFER INTO X
  for (int d(0); d < n; d++) {
    for (int t(0); t < n; t++) {
      x->in_buffer[d][t] = input[d][t];
    }
  }

  // START THREAD TO COMPUTE NEXT BUFFER
  x->worker = new std::thread(perform, x, n);

  return (w + LATENT_NUMBER + 4);
}

void wavae_decoder_tilde_dsp(t_wavae_decoder_tilde *x, t_signal **sp) {
  dsp_add(wavae_decoder_tilde_perform, 19, x, sp[0]->s_vec, sp[1]->s_vec,
          sp[2]->s_vec, sp[3]->s_vec, sp[4]->s_vec, sp[5]->s_vec, sp[6]->s_vec,
          sp[7]->s_vec, sp[8]->s_vec, sp[9]->s_vec, sp[10]->s_vec,
          sp[11]->s_vec, sp[12]->s_vec, sp[13]->s_vec, sp[14]->s_vec,
          sp[15]->s_vec, sp[16]->s_vec, sp[0]->s_n);
}

void wavae_decoder_tilde_free(t_wavae_decoder_tilde *x) {
  for (int i(0); i < LATENT_NUMBER - 1; i++) {
    inlet_free(x->x_in[i]);
  }
  outlet_free(x->x_out);
}

void *wavae_decoder_tilde_new(t_floatarg *f) {
  t_wavae_decoder_tilde *x =
      (t_wavae_decoder_tilde *)pd_new(wavae_decoder_tilde_class);

  x->worker = NULL;

  // INITIALIZATION OF THE LATENT_NUMBER LATENT INPUTS
  for (int i(0); i < LATENT_NUMBER - 1; i++) {
    x->x_in[i] = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  }
  x->x_out = outlet_new(&x->x_obj, &s_signal);

  // INITIALIZE BUFFER
  x->out_buffer.push_back(new float[BUFFER_SIZE]);
  for (int i(0); i < LATENT_NUMBER; i++) {
    x->in_buffer.push_back(new float[BUFFER_SIZE]);
  }

  // LOAD LIBWAVAE.SO ///////////////////////////////////
  void *hndl = dlopen("./libwavae/libwavae.so", RTLD_LAZY);
  if (!hndl) {
    std::cout << "Failed to load libwavae..." << std::endl;
  }

  void *symbol = dlsym(hndl, "get_decoder");
  if (!symbol) {
    std::cout << "Could not find symbol..." << std::endl;
  }
  auto build_decoder = reinterpret_cast<DeepAudioEngine *(*)()>(symbol);

  x->model = (*build_decoder)();
  int error = x->model->load("trace_model.ts");

  if (error) {
    std::cout << "could not load model" << std::endl;
  } else {
    std::cout << "model loaded successfuly!" << std::endl;
  }
  return (void *)x;
}

extern "C" {
void wavae_decoder_tilde_setup(void) {
  wavae_decoder_tilde_class =
      class_new(gensym("wavae_decoder~"), (t_newmethod)wavae_decoder_tilde_new,
                0, sizeof(t_wavae_decoder_tilde), CLASS_DEFAULT, A_DEFFLOAT, 0);

  class_addmethod(wavae_decoder_tilde_class, (t_method)wavae_decoder_tilde_dsp,
                  gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(wavae_decoder_tilde_class, t_wavae_decoder_tilde, f);
}
}
