#include "../libwavae/src/deepAudioEngine.h"
#include "cstring"
#include "dlfcn.h"
#include "m_pd.h"
#include "pthread.h"
#include "sched.h"
#include "thread"
#include <iostream>

#define DAE DeepAudioEngine

static t_class *decoder_tilde_class;

typedef struct _decoder_tilde {
  t_object x_obj;
  t_sample f;

  t_inlet *x_in[LATENT_NUMBER - 1];
  t_outlet *x_out;

  float *in_buffer, *out_buffer;

  std::thread *worker;

  DAE *model;

} t_decoder_tilde;

void perform(t_decoder_tilde *x) {
  // SET THREAD TO REALTIME PRIORITY
  pthread_t this_thread = pthread_self();
  struct sched_param params;
  params.sched_priority = sched_get_priority_max(SCHED_FIFO);
  int ret = pthread_setschedparam(this_thread, SCHED_FIFO, &params);

  // COMPUTATION
  x->model->perform(x->in_buffer, x->out_buffer);
}

t_int *decoder_tilde_perform(t_int *w) {
  t_decoder_tilde *x = (t_decoder_tilde *)w[1];
  int n = (int)w[2];

  // WAIT FOR PREVIOUS PROCESS TO END
  if (x->worker) {
    x->worker->join();
  }

  // COPY INPUT BUFFER TO OBJECT
  for (int d(0); d < LATENT_NUMBER; d++) {
    memcpy(x->in_buffer + (d * BUFFERSIZE), (float *)w[d + 3],
           BUFFERSIZE * sizeof(float));
  }

  // COPY PREVIOUS OUTPUT BUFFER TO PD
  memcpy((float *)w[LATENT_NUMBER + 3], x->out_buffer,
         BUFFERSIZE * sizeof(float));

  // START NEXT COMPUTATION
  x->worker = new std::thread(perform, x);
  return w + 20;
}

void decoder_tilde_dsp(t_decoder_tilde *x, t_signal **sp) {
  dsp_add(decoder_tilde_perform, 19, x, sp[0]->s_n, sp[0]->s_vec, sp[1]->s_vec,
          sp[2]->s_vec, sp[3]->s_vec, sp[4]->s_vec, sp[5]->s_vec, sp[6]->s_vec,
          sp[7]->s_vec, sp[8]->s_vec, sp[9]->s_vec, sp[10]->s_vec,
          sp[11]->s_vec, sp[12]->s_vec, sp[13]->s_vec, sp[14]->s_vec,
          sp[15]->s_vec, sp[16]->s_vec);
}

void decoder_tilde_free(t_decoder_tilde *x) {
  outlet_free(x->x_out);
  for (int i(0); i < LATENT_NUMBER - 1; i++) {
    inlet_free(x->x_in[i]);
  }
  if (x->worker) {
    x->worker->join();
  }
}

void *decoder_tilde_new(t_floatarg f) {
  t_decoder_tilde *x = (t_decoder_tilde *)pd_new(decoder_tilde_class);

  x->x_out = outlet_new(&x->x_obj, &s_signal);
  for (int i(0); i < LATENT_NUMBER - 1; i++) {
    x->x_in[i] = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  }

  x->in_buffer = new float[LATENT_NUMBER * BUFFERSIZE];
  x->out_buffer = new float[BUFFERSIZE];

  x->worker = NULL;

  void *hndl = dlopen("./libwavae/libwavae.so", RTLD_LAZY);
  x->model = reinterpret_cast<DAE *(*)()>(dlsym(hndl, "get_decoder"))();
  // x->model->load("trace_model.ts");

  return (void *)x;
}

void decoder_tilde_load(t_decoder_tilde *x, t_symbol *sym) {
  x->model->load(sym->s_name);
  post("decoder loaded");
}

extern "C" {
void decoder_tilde_setup(void) {
  decoder_tilde_class =
      class_new(gensym("decoder~"), (t_newmethod)decoder_tilde_new, 0,
                sizeof(t_decoder_tilde), CLASS_DEFAULT, A_DEFFLOAT, 0);

  class_addmethod(decoder_tilde_class, (t_method)decoder_tilde_dsp,
                  gensym("dsp"), A_CANT, 0);
  class_addmethod(decoder_tilde_class, (t_method)decoder_tilde_load,
                  gensym("load"), A_SYMBOL, A_NULL);

  CLASS_MAINSIGNALIN(decoder_tilde_class, t_decoder_tilde, f);
}
}
