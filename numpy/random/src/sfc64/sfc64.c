#include "sfc64.h"

extern void sfc64_set_seed(sfc64_state *state, uint64_t *seed) {
  /* Conservatively stick with the original formula. With SeedSequence, it
   * might be fine to just set the state with 4 uint64s and be done.
   */
  int i;

  state->s[0] = seed[0];
  state->s[1] = seed[1];
  state->s[2] = seed[2];
  state->s[3] = 1;

  for (i=0; i<12; i++) {
    (void)sfc64_next(state->s);
  }
}

extern void sfc64_get_state(sfc64_state *state, uint64_t *state_arr, int *has_uint32,
                            uint32_t *uinteger) {
  int i;

  for (i=0; i<4; i++) {
    state_arr[i] = state->s[i];
  }
  has_uint32[0] = state->has_uint32;
  uinteger[0] = state->uinteger;
}

extern void sfc64_set_state(sfc64_state *state, uint64_t *state_arr, int has_uint32,
                            uint32_t uinteger) {
  int i;

  for (i=0; i<4; i++) {
    state->s[i] = state_arr[i];
  }
  state->has_uint32 = has_uint32;
  state->uinteger = uinteger;
}
