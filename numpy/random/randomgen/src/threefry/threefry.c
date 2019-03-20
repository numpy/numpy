#include "threefry.h"

extern INLINE uint64_t threefry_next64(threefry_state *state);

extern INLINE uint32_t threefry_next32(threefry_state *state);

extern void threefry_jump(threefry_state *state) {
  /* Advances state as-if 2^128 draws were made */
  state->ctr->v[2]++;
  if (state->ctr->v[2] == 0) {
    state->ctr->v[3]++;
  }
}

extern void threefry_advance(uint64_t *step, threefry_state *state) {
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
