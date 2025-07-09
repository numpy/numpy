#ifndef _RANDOMDGEN__SFC64_H_
#define _RANDOMDGEN__SFC64_H_

#include "numpy/npy_common.h"
#include <inttypes.h>
#ifdef _WIN32
#include <stdlib.h>
#endif

typedef struct s_sfc64_state {
  uint64_t s[4];
  int has_uint32;
  uint32_t uinteger;
} sfc64_state;


static inline uint64_t rotl(const uint64_t value, unsigned int rot) {
#ifdef _WIN32
  return _rotl64(value, rot);
#else
  return (value << rot) | (value >> ((-rot) & 63));
#endif
}

static inline uint64_t sfc64_next(uint64_t *s) {
  const uint64_t tmp = s[0] + s[1] + s[3]++;

  s[0] = s[1] ^ (s[1] >> 11);
  s[1] = s[2] + (s[2] << 3);
  s[2] = rotl(s[2], 24) + tmp;

  return tmp;
}


static inline uint64_t sfc64_next64(sfc64_state *state) {
  return sfc64_next(&state->s[0]);
}

static inline uint32_t sfc64_next32(sfc64_state *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = sfc64_next(&state->s[0]);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

void sfc64_set_seed(sfc64_state *state, uint64_t *seed);

void sfc64_get_state(sfc64_state *state, uint64_t *state_arr, int *has_uint32,
                     uint32_t *uinteger);

void sfc64_set_state(sfc64_state *state, uint64_t *state_arr, int has_uint32,
                     uint32_t uinteger);

#endif
