#ifdef _WIN32
#if _MSC_VER == 1500
#include "../common/inttypes.h"
#define INLINE __forceinline
#else
#include <inttypes.h>
#define INLINE __inline __forceinline
#endif
#else
#include <inttypes.h>
#define INLINE inline
#endif

typedef struct s_xoroshiro128_state {
  uint64_t s[2];
  int has_uint32;
  uint32_t uinteger;
} xoroshiro128_state;

static INLINE uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static INLINE uint64_t xoroshiro128_next(uint64_t *s) {
  const uint64_t s0 = s[0];
  uint64_t s1 = s[1];
  const uint64_t result = s0 + s1;

  s1 ^= s0;
  s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
  s[1] = rotl(s1, 36);                   // c

  return result;
}

static INLINE uint64_t xoroshiro128_next64(xoroshiro128_state *state) {
  return xoroshiro128_next(&state->s[0]);
}

static INLINE uint32_t xoroshiro128_next32(xoroshiro128_state *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = xoroshiro128_next(&state->s[0]);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

void xoroshiro128_jump(xoroshiro128_state *state);
