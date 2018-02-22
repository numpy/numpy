#include <stdint.h>

typedef struct s_xoroshiro128_state {
  uint64_t s[2];
} xoroshiro128_state;

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoroshiro128_next(xoroshiro128_state *state) {
  const uint64_t s0 = state->s[0];
  uint64_t s1 = state->s[1];
  const uint64_t result = s0 + s1;

  s1 ^= s0;
  state->s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
  state->s[1] = rotl(s1, 36);                   // c

  return result;
}

void xoroshiro128_jump(xoroshiro128_state *state);
