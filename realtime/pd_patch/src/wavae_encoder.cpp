#include "../../libwavae/src/deepAudioEngine.h"
#include "m_pd.h"
#include <dlfcn.h>
#include <iostream>

static t_class *wavae_encoder_tilde_class;

typedef struct _wavae_encoder_tilde {
  t_object x_obj;
  t_sample f;

  DeepAudioEngine *model;

  t_outlet *x_out[16];
} t_wavae_encoder_tilde;

t_int *wavae_encoder_tilde_perform(t_int *w) {
  // IN OUT INITIALIZATION
  t_wavae_encoder_tilde *x = (t_wavae_encoder_tilde *)(w[1]);
  t_sample *in_buffer = (t_sample *)w[2];
  t_sample *outs[16];
  for (int i(0); i < 16; i++) {
    outs[i] = (t_sample *)(w[i + 3]);
  }
  int n = (int)(w[19]);

  // UPLOAD BUFFER TO MODEL
  x->model->addBuffer(&in_buffer, n, 1);

  // DOWNLOAD DATA FROM MODEL
  x->model->getBuffer(outs, n, 16);
  return (w + 20);
}

void wavae_encoder_tilde_dsp(t_wavae_encoder_tilde *x, t_signal **sp) {
  dsp_add(wavae_encoder_tilde_perform, 19, x, sp[0]->s_vec, sp[1]->s_vec,
          sp[2]->s_vec, sp[3]->s_vec, sp[4]->s_vec, sp[5]->s_vec, sp[6]->s_vec,
          sp[7]->s_vec, sp[8]->s_vec, sp[9]->s_vec, sp[10]->s_vec,
          sp[11]->s_vec, sp[12]->s_vec, sp[13]->s_vec, sp[14]->s_vec,
          sp[15]->s_vec, sp[16]->s_vec, sp[0]->s_n);
}

void wavae_encoder_tilde_free(t_wavae_encoder_tilde *x) {
  for (int i(0); i < 16; i++) {
    outlet_free(x->x_out[i]);
  }
}

void *wavae_encoder_tilde_new(t_floatarg *f) {
  t_wavae_encoder_tilde *x =
      (t_wavae_encoder_tilde *)pd_new(wavae_encoder_tilde_class);

  // INITIALIZATION OF THE 16 LATENT OUTPUTS
  for (int i(0); i < 16; i++) {
    x->x_out[i] = outlet_new(&x->x_obj, &s_signal);
  }

  // LOAD LIBWAVAE.SO
  void *hndl = dlopen("./libwavae.so", RTLD_LAZY);
  if (!hndl) {
    std::cout << "Failed to load libwavae..." << std::endl;
  }

  void *symbol = dlsym(hndl, "build_encoder");
  if (!symbol) {
    std::cout << "Could not find symbol..." << std::endl;
  }
  auto build_encoder = reinterpret_cast<DeepAudioEngine *(*)()>(symbol);

  x->model = (*build_encoder)();
  int error = x->model->load("alexander");

  if (error) {
    std::cout << "could not load model" << std::endl;
  } else {
    std::cout << "model loaded successfuly!" << std::endl;
  }
  return (void *)x;
}

extern "C" {
void wavae_encoder_tilde_setup(void) {
  wavae_encoder_tilde_class =
      class_new(gensym("wavae_encoder~"), (t_newmethod)wavae_encoder_tilde_new,
                0, sizeof(t_wavae_encoder_tilde), CLASS_DEFAULT, A_DEFFLOAT, 0);

  class_addmethod(wavae_encoder_tilde_class, (t_method)wavae_encoder_tilde_dsp,
                  gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(wavae_encoder_tilde_class, t_wavae_encoder_tilde, f);
}
}
