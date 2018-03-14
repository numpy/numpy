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

static INLINE uint32_t RotL_32(uint32_t x, unsigned int N);
static INLINE uint32_t RotL_32(uint32_t x, unsigned int N) {
  return (x << (N & 31)) | (x >> ((32 - N) & 31));
}

struct r123array4x32 {
  uint32_t v[4];
};

enum r123_enum_threefry32x4 {

  R_32x4_0_0 = 10,
  R_32x4_0_1 = 26,
  R_32x4_1_0 = 11,
  R_32x4_1_1 = 21,
  R_32x4_2_0 = 13,
  R_32x4_2_1 = 27,
  R_32x4_3_0 = 23,
  R_32x4_3_1 = 5,
  R_32x4_4_0 = 6,
  R_32x4_4_1 = 20,
  R_32x4_5_0 = 17,
  R_32x4_5_1 = 11,
  R_32x4_6_0 = 25,
  R_32x4_6_1 = 10,
  R_32x4_7_0 = 18,
  R_32x4_7_1 = 20

};

typedef struct r123array4x32 threefry4x32_ctr_t;
typedef struct r123array4x32 threefry4x32_key_t;
typedef struct r123array4x32 threefry4x32_ukey_t;
static INLINE threefry4x32_key_t threefry4x32keyinit(threefry4x32_ukey_t uk) {
  return uk;
};
static INLINE threefry4x32_ctr_t threefry4x32_R(unsigned int Nrounds,
                                                threefry4x32_ctr_t in,
                                                threefry4x32_key_t k);
static INLINE threefry4x32_ctr_t threefry4x32_R(unsigned int Nrounds,
                                                threefry4x32_ctr_t in,
                                                threefry4x32_key_t k) {
  threefry4x32_ctr_t X;
  uint32_t ks[4 + 1];
  int i;
  ks[4] = 0x1BD11BDA;
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
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 1) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 2) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 3) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
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
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 5) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 6) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 7) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
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
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 9) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 10) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 11) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
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
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 13) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 14) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 15) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
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
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 17) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 18) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 19) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 19) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    X.v[4 - 1] += 5;
  }
  if (Nrounds > 20) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 21) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 22) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 23) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 23) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 6;
  }
  if (Nrounds > 24) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 25) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 26) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 27) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 27) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 7;
  }
  if (Nrounds > 28) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 29) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 30) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 31) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 31) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 8;
  }
  if (Nrounds > 32) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 33) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 34) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 35) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 35) {
    X.v[0] += ks[4];
    X.v[1] += ks[0];
    X.v[2] += ks[1];
    X.v[3] += ks[2];
    X.v[4 - 1] += 9;
  }
  if (Nrounds > 36) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 37) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 38) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 39) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 39) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    X.v[4 - 1] += 10;
  }
  if (Nrounds > 40) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 41) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 42) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 43) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 43) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 11;
  }
  if (Nrounds > 44) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 45) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 46) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 47) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 47) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 12;
  }
  if (Nrounds > 48) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 49) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 50) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 51) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 51) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 13;
  }
  if (Nrounds > 52) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 53) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 54) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 55) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 55) {
    X.v[0] += ks[4];
    X.v[1] += ks[0];
    X.v[2] += ks[1];
    X.v[3] += ks[2];
    X.v[4 - 1] += 14;
  }
  if (Nrounds > 56) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 57) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 58) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 59) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 59) {
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    X.v[4 - 1] += 15;
  }
  if (Nrounds > 60) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 61) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 62) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 63) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 63) {
    X.v[0] += ks[1];
    X.v[1] += ks[2];
    X.v[2] += ks[3];
    X.v[3] += ks[4];
    X.v[4 - 1] += 16;
  }
  if (Nrounds > 64) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 65) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 66) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 67) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 67) {
    X.v[0] += ks[2];
    X.v[1] += ks[3];
    X.v[2] += ks[4];
    X.v[3] += ks[0];
    X.v[4 - 1] += 17;
  }
  if (Nrounds > 68) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 69) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 70) {
    X.v[0] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
    X.v[1] ^= X.v[0];
    X.v[2] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
    X.v[3] ^= X.v[2];
  }
  if (Nrounds > 71) {
    X.v[0] += X.v[3];
    X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
    X.v[3] ^= X.v[0];
    X.v[2] += X.v[1];
    X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
    X.v[1] ^= X.v[2];
  }
  if (Nrounds > 71) {
    X.v[0] += ks[3];
    X.v[1] += ks[4];
    X.v[2] += ks[0];
    X.v[3] += ks[1];
    X.v[4 - 1] += 18;
  }
  return X;
}
enum r123_enum_threefry4x32 { threefry4x32_rounds = 20 };
static INLINE threefry4x32_ctr_t threefry4x32(threefry4x32_ctr_t in,
                                              threefry4x32_key_t k);
static INLINE threefry4x32_ctr_t threefry4x32(threefry4x32_ctr_t in,
                                              threefry4x32_key_t k) {
  return threefry4x32_R(threefry4x32_rounds, in, k);
}

typedef struct s_threefry32_state {
  threefry4x32_key_t *ctr;
  threefry4x32_ctr_t *key;
  int buffer_pos;
  uint32_t buffer[THREEFRY_BUFFER_SIZE];
} threefry32_state;

static INLINE uint32_t threefry32_next(threefry32_state *state) {
  int i;
  threefry4x32_ctr_t ct;
  uint32_t out;
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
  ct = threefry4x32_R(threefry4x32_rounds, *state->ctr, *state->key);
  for (i = 0; i < 4; i++) {
    state->buffer[i] = ct.v[i];
  }
  state->buffer_pos = 1;
  return state->buffer[0];
}

static INLINE uint64_t threefry32_next64(threefry32_state *state) {
  return ((uint64_t)threefry32_next(state) << 32) | threefry32_next(state);
}

static INLINE uint32_t threefry32_next32(threefry32_state *state) {
  return threefry32_next(state);
}

static INLINE double threefry32_next_double(threefry32_state *state) {
  int32_t a = threefry32_next(state) >> 5, b = threefry32_next(state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

extern void threefry32_jump(threefry32_state *state);

extern void threefry32_advance(uint32_t *step, threefry32_state *state);
