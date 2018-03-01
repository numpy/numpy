#include "distributions.h"

uint32_t random_uint32(prng_t *prng_state) {
  return prng_state->next_uint32(prng_state->state);
}

float random_float(prng_t *prng_state) {
  uint32_t next_32 = prng_state->next_uint32(prng_state->state);
  return (next_32 >> 9) * (1.0f / 8388608.0f);
}

double random_double(prng_t *prng_state) {
  return prng_state->next_double(prng_state->state);
}

double random_standard_exponential(prng_t *prng_state) {
  return -log(1.0 - prng_state->next_double(prng_state->state));
}
