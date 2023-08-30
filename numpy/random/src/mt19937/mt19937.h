#pragma once
#include <math.h>
#include <stdint.h>

#if defined(_WIN32) && !defined (__MINGW32__)
#define inline __forceinline
#endif

#define RK_STATE_LEN 624

#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

typedef struct s_mt19937_state {
  uint32_t key[RK_STATE_LEN];
  int pos;
} mt19937_state;

extern void mt19937_seed(mt19937_state *state, uint32_t seed);

extern void mt19937_gen(mt19937_state *state);

/* Slightly optimized reference implementation of the Mersenne Twister */
static inline uint32_t mt19937_next(mt19937_state *state) {
  uint32_t y;

  if (state->pos == RK_STATE_LEN) {
    // Move to function to help inlining
    mt19937_gen(state);
  }
  y = state->key[state->pos++];

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
}

extern void mt19937_init_by_array(mt19937_state *state, uint32_t *init_key,
                                  int key_length);

static inline uint64_t mt19937_next64(mt19937_state *state) {
  return (uint64_t)mt19937_next(state) << 32 | mt19937_next(state);
}

static inline uint32_t mt19937_next32(mt19937_state *state) {
  return mt19937_next(state);
}

static inline double mt19937_next_double(mt19937_state *state) {
  int32_t a = mt19937_next(state) >> 5, b = mt19937_next(state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

void mt19937_jump(mt19937_state *state);
