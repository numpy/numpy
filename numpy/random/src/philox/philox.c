#include "philox.h"

extern NPY_INLINE uint64_t philox_next64(philox_state *state);

extern NPY_INLINE uint32_t philox_next32(philox_state *state);

extern void philox_jump(philox_state *state) {
  /* Advances state as-if 2^128 draws were made */
  state->ctr->v[2]++;
  if (state->ctr->v[2] == 0) {
    state->ctr->v[3]++;
  }
}

extern void philox_advance(uint64_t *step, philox_state *state) {
  int i, carry = 0;
  uint64_t v_orig;
  for (i = 0; i < 4; i++) {
    if (carry == 1) {
      state->ctr->v[i]++;
      carry = state->ctr->v[i] == 0 ? 1 : 0;
    }
    v_orig = state->ctr->v[i];
    state->ctr->v[i] += step[i];
    if (state->ctr->v[i] < v_orig && carry == 0) {
      carry = 1;
    }
  }
}
