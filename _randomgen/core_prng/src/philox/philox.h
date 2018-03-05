#include <inttypes.h>

#ifdef _WIN32
#define INLINE __inline _forceinline
#else
#define INLINE inline
#endif

struct r123array2x64 {
  uint64_t v[2];
};
struct r123array4x64 {
  uint64_t v[4];
};

enum r123_enum_philox4x64 { philox4x64_rounds = 10 };
typedef struct r123array4x64 philox4x64_ctr_t;
typedef struct r123array2x64 philox4x64_key_t;
typedef struct r123array2x64 philox4x64_ukey_t;

static __inline struct r123array2x64
_philox4x64bumpkey(struct r123array2x64 key) {
  key.v[0] += (0x9E3779B97F4A7C15ULL);
  key.v[1] += (0xBB67AE8584CAA73BULL);
  return key;
}

#ifdef _WIN32
/* TODO: This isn't correct for many platforms */
#include <intrin.h>
#pragma intrinsic(_umul128)

static __inline uint64_t mulhilo64(uint64_t a, uint64_t b, uint64_t *hip) {
  return _umul128(a, b, hip);
}
#else
static inline uint64_t mulhilo64(uint64_t a, uint64_t b, uint64_t *hip) {
  __uint128_t product = ((__uint128_t)a) * ((__uint128_t)b);
  *hip = product >> 64;
  return (uint64_t)product;
}
#endif

// static __inline _forceinline struct r123array4x64 _philox4x64round(struct
// r123array4x64 ctr, struct r123array2x64 key);

static INLINE struct r123array4x64
_philox4x64round(struct r123array4x64 ctr, struct r123array2x64 key) {
  uint64_t hi0;
  uint64_t hi1;
  uint64_t lo0 = mulhilo64((0xD2E7470EE14C6C93ULL), ctr.v[0], &hi0);
  uint64_t lo1 = mulhilo64((0xCA5A826395121157ULL), ctr.v[2], &hi1);
  struct r123array4x64 out = {
      {hi1 ^ ctr.v[1] ^ key.v[0], lo1, hi0 ^ ctr.v[3] ^ key.v[1], lo0}};
  return out;
}

static INLINE philox4x64_key_t philox4x64keyinit(philox4x64_ukey_t uk) {
  return uk;
}
// static __inline _forceinline philox4x64_ctr_t philox4x64_R(unsigned int R,
// philox4x64_ctr_t ctr, philox4x64_key_t key);

static INLINE philox4x64_ctr_t
philox4x64_R(unsigned int R, philox4x64_ctr_t ctr, philox4x64_key_t key) {
  if (R > 0) {
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 1) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 2) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 3) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 4) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 5) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 6) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 7) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 8) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 9) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 10) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 11) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 12) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 13) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 14) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  if (R > 15) {
    key = _philox4x64bumpkey(key);
    ctr = _philox4x64round(ctr, key);
  }
  return ctr;
}

typedef struct s_philox_state {
  philox4x64_ctr_t *ctr;
  philox4x64_key_t *key;
  int buffer_pos;
  uint64_t buffer[4];
  int has_uint32;
  uint32_t uinteger;
} philox_state;

static inline uint64_t philox_next(philox_state *state) {
  /* TODO: This 4 should be a constant somewhere */
  if (state->buffer_pos < 4) {
    uint64_t out = state->buffer[state->buffer_pos];
    state->buffer_pos++;
    return out;
  }
  /* generate 4 new uint64_t */
  int i;
  philox4x64_ctr_t ct;
  state->ctr->v[0]++;
  /* Handle carry */
  if (state->ctr->v[0] == 0) {
    state->ctr->v[1]++;
    if (state->ctr->v[1] == 0) {
      state->ctr->v[2]++;
      if (state->ctr->v[2] == 0) {
        state->ctr->v[3]++;
      }
    }
  }
  ct = philox4x64_R(philox4x64_rounds, *state->ctr, *state->key);
  for (i = 0; i < 4; i++) {
    state->buffer[i] = ct.v[i];
  }
  state->buffer_pos = 1;
  return state->buffer[0];
}

static inline uint64_t philox_next64(philox_state *state) {
  return philox_next(state);
}

static inline uint64_t philox_next32(philox_state *state) {
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  uint64_t next = philox_next(state);

  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next & 0xffffffff);

  return (uint32_t)(next >> 32);
}

extern void philox_jump(philox_state *state);

extern void philox_advance(uint64_t *step, philox_state *state);
