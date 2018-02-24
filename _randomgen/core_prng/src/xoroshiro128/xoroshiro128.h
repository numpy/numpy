#include <stdint.h>

static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoroshiro128_next(uint64_t *s) {
  const uint64_t s0 = s[0];
  uint64_t s1 = s[1];
  const uint64_t result = s0 + s1;

  s1 ^= s0;
  s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
  s[1] = rotl(s1, 36);                   // c

  return result;
}

void xoroshiro128_jump(uint64_t *s);
