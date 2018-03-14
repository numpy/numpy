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

typedef struct s_xorshift1024_state {
  uint64_t s[16];
  int p;
  int has_uint32;
  uint32_t uinteger;
} xorshift1024_state;

static INLINE uint64_t xorshift1024_next(xorshift1024_state *state) {
  const uint64_t s0 = state->s[state->p];
  uint64_t s1 = state->s[state->p = ((state->p) + 1) & 15];
  s1 ^= s1 << 31;                                         // a
  state->s[state->p] = s1 ^ s0 ^ (s1 >> 11) ^ (s0 >> 30); // b,c
  return state->s[state->p] * 0x9e3779b97f4a7c13;
}

static INLINE uint64_t xorshift1024_next64(xorshift1024_state *state) {
  return xorshift1024_next(state);
}

static INLINE uint32_t xorshift1024_next32(xorshift1024_state *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = xorshift1024_next(state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

void xorshift1024_jump(xorshift1024_state *state);
