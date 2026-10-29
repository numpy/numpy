#include "philox.h"

void philox_fill_uint32(philox_state *state, size_t count, uint32_t *out) {
  size_t i;
  uint64_t next = 0;
  philox_state rng;

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

  rng = *state;
  for (i = 0; i < count / 2; i++) {
    next = philox_next(&rng);
    out[2 * i] = (uint32_t)(next & 0xffffffff);
    out[2 * i + 1] = (uint32_t)(next >> 32);
  }
  if (count & 1) {
    next = philox_next(&rng);
    out[count - 1] = (uint32_t)(next & 0xffffffff);
    rng.has_uint32 = 1;
  }

  rng.uinteger = (uint32_t)(next >> 32);
  *state = rng;
}

void philox_fill_uint64(philox_state *state, size_t count, uint64_t *out) {
  size_t i;
  philox_state rng;

  if (count == 0) {
    return;
  }
  rng = *state;
  for (i = 0; i < count; i++) {
    out[i] = philox_next(&rng);
  }
  *state = rng;
}

void philox_fill_double(philox_state *state, size_t count, double *out) {
  size_t i;
  philox_state rng;

  if (count == 0) {
    return;
  }
  rng = *state;
  for (i = 0; i < count; i++) {
    out[i] = (philox_next(&rng) >> 11) * (1.0 / 9007199254740992.0);
  }
  *state = rng;
}

extern inline uint64_t philox_next64(philox_state *state);

extern inline uint32_t philox_next32(philox_state *state);

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
