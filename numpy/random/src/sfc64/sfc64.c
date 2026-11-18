#include "sfc64.h"

static inline void sfc64_copy_state_arr(uint64_t *dst, const uint64_t *src) {
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
  dst[3] = src[3];
}

void sfc64_fill_uint32(sfc64_state *state, size_t count, uint32_t *out) {
  size_t i;
  uint64_t next = 0;
  uint64_t s[4];

  if (count == 0) {
    return;
  }

  if (state->has_uint32) {
    out[0] = state->uinteger;
    out++;
    state->has_uint32 = 0;
    count--;
    if (count == 0) {
      return;
    }
  }

  sfc64_copy_state_arr(s, state->s);
  for (i = 0; i < count / 2; i++) {
    next = sfc64_next(s);
    out[2 * i] = (uint32_t)(next & 0xffffffff);
    out[2 * i + 1] = (uint32_t)(next >> 32);
  }
  if (count & 1) {
    next = sfc64_next(s);
    out[count - 1] = (uint32_t)(next & 0xffffffff);
    state->has_uint32 = 1;
  }

  state->uinteger = (uint32_t)(next >> 32);
  sfc64_copy_state_arr(state->s, s);
}

void sfc64_fill_uint64(sfc64_state *state, size_t count, uint64_t *out) {
  size_t i;
  uint64_t s[4];

  if (count == 0) {
    return;
  }
  sfc64_copy_state_arr(s, state->s);
  for (i = 0; i < count; i++) {
    out[i] = sfc64_next(s);
  }
  sfc64_copy_state_arr(state->s, s);
}

void sfc64_fill_double(sfc64_state *state, size_t count, double *out) {
  size_t i;
  uint64_t s[4];

  if (count == 0) {
    return;
  }
  sfc64_copy_state_arr(s, state->s);
  for (i = 0; i < count; i++) {
    out[i] = (sfc64_next(s) >> 11) * (1.0 / 9007199254740992.0);
  }
  sfc64_copy_state_arr(state->s, s);
}

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
