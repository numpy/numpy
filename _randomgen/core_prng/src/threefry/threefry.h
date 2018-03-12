/*
Adapted from random123's threefry.h
*/

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

#define THREEFRY_BUFFER_SIZE 4L

enum r123_enum_threefry64x4 {
  /* These are the R_256 constants from the Threefish reference sources
     with names changed to R_64x4... */
  R_64x4_0_0 = 14,
  R_64x4_0_1 = 16,
  R_64x4_1_0 = 52,
  R_64x4_1_1 = 57,
  R_64x4_2_0 = 23,
  R_64x4_2_1 = 40,
  R_64x4_3_0 = 5,
  R_64x4_3_1 = 37,
  R_64x4_4_0 = 25,
  R_64x4_4_1 = 33,
  R_64x4_5_0 = 46,
  R_64x4_5_1 = 12,
  R_64x4_6_0 = 58,
  R_64x4_6_1 = 22,
  R_64x4_7_0 = 32,
  R_64x4_7_1 = 32
};

struct r123array4x64 {
  uint64_t v[4];
}; /* r123array4x64 */

typedef struct r123array4x64 threefry4x64_key_t;
typedef struct r123array4x64 threefry4x64_ctr_t;

static INLINE uint64_t RotL_64(uint64_t x, unsigned int N);
static INLINE uint64_t RotL_64(uint64_t x, unsigned int N) {
  return (x << (N & 63)) | (x >> ((64 - N) & 63));
}

static INLINE threefry4x64_ctr_t threefry4x64_R(unsigned int Nrounds,
                                                threefry4x64_ctr_t in,
                                                threefry4x64_key_t k);
static INLINE threefry4x64_ctr_t threefry4x64_R(unsigned int Nrounds,
                                                threefry4x64_ctr_t in,
                                                threefry4x64_key_t k) {
  threefry4x64_ctr_t X;
  uint64_t ks[4 + 1];
  int i;
  ks[4] = ((0xA9FC1A22) + (((uint64_t)(0x1BD11BDA)) << 32));
  for (i = 0; i < 4; i++) {
    ks[i] = k.v[i];
    X.v[i] = in.v[i];
    ks[4] ^= k.v[i];
  }
  X.v[0] += ks[0];
  X.v[1] += ks[1];
  X.v[2] += ks[2];
  X.v[3] += ks[3];
  if (Nrounds > 0) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 1) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 2) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 3) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 3) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 1;
  }
  if (Nrounds > 4) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 5) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 6) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 7) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 7) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 2;
  }
  if (Nrounds > 8) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 9) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 10) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 11) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 11) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 3;
  }
  if (Nrounds > 12) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 13) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 14) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 15) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 15) {
    X.v[0] += ks[4];
    X.v[1] += ks[0];
    X.v[2] += ks[1];
    X.v[3] += ks[2];
    X.v[4 - 1] += 4;
  }
  if (Nrounds > 16) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 17) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 18) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 19) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 19) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    X.v[4 - 1] += 5;
  }
  /* Maximum of 20 rounds */
  if (Nrounds > 20) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
    X.v[3] ^= X.v[2];
  }
  return X;
}
enum r123_enum_threefry4x64 { threefry4x64_rounds = 20 };

typedef struct s_threefry_state {
  threefry4x64_key_t *ctr;
  threefry4x64_ctr_t *key;
  int buffer_pos;
  uint64_t buffer[THREEFRY_BUFFER_SIZE];
  int has_uint32;
  uint32_t uinteger;
} threefry_state;

static INLINE uint64_t threefry_next(threefry_state *state) {
  int i;
  threefry4x64_ctr_t ct;
  uint64_t out;
  if (state->buffer_pos < THREEFRY_BUFFER_SIZE) {
    out = state->buffer[state->buffer_pos];
    state->buffer_pos++;
    return out;
  }
  /* generate 4 new uint64_t */
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
  ct = threefry4x64_R(threefry4x64_rounds, *state->ctr, *state->key);
  for (i = 0; i < 4; i++) {
    state->buffer[i] = ct.v[i];
  }
  state->buffer_pos = 1;
  return state->buffer[0];
}

static INLINE uint64_t threefry_next64(threefry_state *state) {
  return threefry_next(state);
}

static INLINE uint32_t threefry_next32(threefry_state *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = threefry_next(state);

  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

extern void threefry_jump(threefry_state *state);

extern void threefry_advance(uint64_t *step, threefry_state *state);
