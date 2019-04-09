#include "threefry32.h"

extern INLINE uint64_t threefry32_next64(threefry32_state *state);

extern INLINE uint32_t threefry32_next32(threefry32_state *state);

extern void threefry32_jump(threefry32_state *state) {
  /* Advances state as-if 2^64 draws were made */
  state->ctr->v[2]++;
  if (state->ctr->v[2] == 0) {
    state->ctr->v[3]++;
  }
}

extern void threefry32_advance(uint32_t *step, threefry32_state *state) {
  int i, carry = 0;
  uint32_t v_orig;
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
