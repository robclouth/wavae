#include "../libwavae/src/deepAudioEngine.h"
#include "cstring"
#include "dlfcn.h"
#include "m_pd.h"
#include "pthread.h"
#include "sched.h"
#include "thread"
#include <iostream>

#define BUFFER_SIZE 4096
#define LATENT_NUMBER 16
#define DIM_REDUCTION BUFFER_SIZE / LATENT_SIZE

#define DAE DeepAudioEngine

static t_class *encoder_tilde_class;

typedef struct _encoder_tilde {
  t_object x_obj;
  t_sample f;

  t_outlet *x_out[LATENT_NUMBER];

  float *in_buffer, *out_buffer;

  std::thread *worker;

  DAE *model;

} t_encoder_tilde;

void perform(t_encoder_tilde *x) {
  // SET THREAD TO REALTIME PRIORITY
  pthread_t this_thread = pthread_self();
  struct sched_param params;
  params.sched_priority = sched_get_priority_max(SCHED_FIFO);
  int ret = pthread_setschedparam(this_thread, SCHED_FIFO, &params);

  // COMPUTATION
  x->model->perform(x->in_buffer, x->out_buffer);
}

t_int *encoder_tilde_perform(t_int *w) {
  t_encoder_tilde *x = (t_encoder_tilde *)w[1];
  int n = (int)w[2];

  // WAIT FOR PREVIOUS PROCESS TO END
  if (x->worker) {
    x->worker->join();
  }

  // COPY INPUT BUFFER TO OBJECT
  memcpy(x->in_buffer, (float *)w[3], BUFFER_SIZE * sizeof(float));

  // COPY PREVIOUS OUTPUT BUFFER TO PD
  for (int d(0); d < LATENT_NUMBER; d++) {
    memcpy((float *)w[d + 4], x->out_buffer + (d * BUFFER_SIZE),
           BUFFER_SIZE * sizeof(float));
  }

  // START NEXT COMPUTATION
  x->worker = new std::thread(perform, x);
  return w + 20;
}

void encoder_tilde_dsp(t_encoder_tilde *x, t_signal **sp) {
  dsp_add(encoder_tilde_perform, 19, x, sp[0]->s_n, sp[0]->s_vec, sp[1]->s_vec,
          sp[2]->s_vec, sp[3]->s_vec, sp[4]->s_vec, sp[5]->s_vec, sp[6]->s_vec,
          sp[7]->s_vec, sp[8]->s_vec, sp[9]->s_vec, sp[10]->s_vec,
          sp[11]->s_vec, sp[12]->s_vec, sp[13]->s_vec, sp[14]->s_vec,
          sp[15]->s_vec, sp[16]->s_vec);
}

void encoder_tilde_free(t_encoder_tilde *x) {
  for (int i(0); i < LATENT_NUMBER; i++) {
    outlet_free(x->x_out[i]);
  }
  if (x->worker) {
    x->worker->join();
  }
}

void *encoder_tilde_new(t_floatarg f) {
  t_encoder_tilde *x = (t_encoder_tilde *)pd_new(encoder_tilde_class);

  for (int i(0); i < LATENT_NUMBER; i++) {
    x->x_out[i] = outlet_new(&x->x_obj, &s_signal);
  }

  x->in_buffer = new float[BUFFER_SIZE];
  x->out_buffer = new float[LATENT_NUMBER * BUFFER_SIZE];

  x->worker = NULL;

  void *hndl = dlopen("./libwavae/libwavae.so", RTLD_LAZY);
  x->model = reinterpret_cast<DAE *(*)()>(dlsym(hndl, "get_encoder"))();
  x->model->load("trace_model.ts");

  return (void *)x;
}

extern "C" {
void encoder_tilde_setup(void) {
  encoder_tilde_class =
      class_new(gensym("encoder~"), (t_newmethod)encoder_tilde_new, 0,
                sizeof(t_encoder_tilde), CLASS_DEFAULT, A_DEFFLOAT, 0);

  class_addmethod(encoder_tilde_class, (t_method)encoder_tilde_dsp,
                  gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(encoder_tilde_class, t_encoder_tilde, f);
}
}
