#include "distributions.h"

uint32_t random_uint32(void *void_state) {
  prng_t *prng_state = (prng_t *)void_state;
  prng_uint32 next_uint32 = (prng_uint32)prng_state->next_uint32;
  return next_uint32(prng_state->state);
}

float random_float(void *void_state) {
  prng_t *prng_state = (prng_t *)void_state;
  prng_uint32 next_uint32 = (prng_uint32)(prng_state->next_uint32);
  uint32_t next_value = next_uint32(prng_state->state);
  return (next_value >> 9) * (1.0f / 8388608.0f);
}

double random_double(void *void_state) {
  prng_t *prng_state = (prng_t *)void_state;
  prng_double next_double = (prng_double)prng_state->next_double;
  return next_double(prng_state->state);
}

double random_standard_exponential(void *void_state) {
  prng_t *prng_state = (prng_t *)void_state;
  prng_double next_double = (prng_double)prng_state->next_double;
  return -log(1.0 - next_double(prng_state->state));
}
