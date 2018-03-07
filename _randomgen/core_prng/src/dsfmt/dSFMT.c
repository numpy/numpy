/**
 * @file dSFMT.c
 * @brief double precision SIMD-oriented Fast Mersenne Twister (dSFMT)
 * based on IEEE 754 format.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (Hiroshima University)
 *
 * Copyright (C) 2007,2008 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University. All rights reserved.
 *
 * The new BSD License is applied to this software, see LICENSE.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dSFMT-params.h"

#include "dSFMT-common.h"
#include "dSFMT-jump.h"
#include "dSFMT-poly.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** dsfmt internal state vector */
dsfmt_t dsfmt_global_data;
/** dsfmt mexp for check */
static const int dsfmt_mexp = DSFMT_MEXP;

/*----------------
  STATIC FUNCTIONS
  ----------------*/
inline static uint32_t ini_func1(uint32_t x);
inline static uint32_t ini_func2(uint32_t x);
inline static void gen_rand_array_c1o2(dsfmt_t *dsfmt, w128_t *array, int size);
inline static void gen_rand_array_c0o1(dsfmt_t *dsfmt, w128_t *array, int size);
inline static void gen_rand_array_o0c1(dsfmt_t *dsfmt, w128_t *array, int size);
inline static void gen_rand_array_o0o1(dsfmt_t *dsfmt, w128_t *array, int size);
inline static int idxof(int i);
static void initial_mask(dsfmt_t *dsfmt);
static void period_certification(dsfmt_t *dsfmt);

#if defined(HAVE_SSE2)
/** 1 in 64bit for sse2 */
static const union X128I_T sse2_int_one = {{1, 1}};
/** 2.0 double for sse2 */
static const union X128D_T sse2_double_two = {{2.0, 2.0}};
/** -1.0 double for sse2 */
static const union X128D_T sse2_double_m_one = {{-1.0, -1.0}};
#endif

/**
 * This function simulate a 32-bit array index overlapped to 64-bit
 * array of LITTLE ENDIAN in BIG ENDIAN machine.
 */
#if defined(DSFMT_BIG_ENDIAN)
inline static int idxof(int i) { return i ^ 1; }
#else
inline static int idxof(int i) { return i; }
#endif

#if defined(HAVE_SSE2)
/**
 * This function converts the double precision floating point numbers which
 * distribute uniformly in the range [1, 2) to those which distribute uniformly
 * in the range [0, 1).
 * @param w 128bit stracture of double precision floating point numbers (I/O)
 */
inline static void convert_c0o1(w128_t *w) {
  w->sd = _mm_add_pd(w->sd, sse2_double_m_one.d128);
}

/**
 * This function converts the double precision floating point numbers which
 * distribute uniformly in the range [1, 2) to those which distribute uniformly
 * in the range (0, 1].
 * @param w 128bit stracture of double precision floating point numbers (I/O)
 */
inline static void convert_o0c1(w128_t *w) {
  w->sd = _mm_sub_pd(sse2_double_two.d128, w->sd);
}

/**
 * This function converts the double precision floating point numbers which
 * distribute uniformly in the range [1, 2) to those which distribute uniformly
 * in the range (0, 1).
 * @param w 128bit stracture of double precision floating point numbers (I/O)
 */
inline static void convert_o0o1(w128_t *w) {
  w->si = _mm_or_si128(w->si, sse2_int_one.i128);
  w->sd = _mm_add_pd(w->sd, sse2_double_m_one.d128);
}
#else /* standard C and altivec */
/**
 * This function converts the double precision floating point numbers which
 * distribute uniformly in the range [1, 2) to those which distribute uniformly
 * in the range [0, 1).
 * @param w 128bit stracture of double precision floating point numbers (I/O)
 */
inline static void convert_c0o1(w128_t *w) {
  w->d[0] -= 1.0;
  w->d[1] -= 1.0;
}

/**
 * This function converts the double precision floating point numbers which
 * distribute uniformly in the range [1, 2) to those which distribute uniformly
 * in the range (0, 1].
 * @param w 128bit stracture of double precision floating point numbers (I/O)
 */
inline static void convert_o0c1(w128_t *w) {
  w->d[0] = 2.0 - w->d[0];
  w->d[1] = 2.0 - w->d[1];
}

/**
 * This function converts the double precision floating point numbers which
 * distribute uniformly in the range [1, 2) to those which distribute uniformly
 * in the range (0, 1).
 * @param w 128bit stracture of double precision floating point numbers (I/O)
 */
inline static void convert_o0o1(w128_t *w) {
  w->u[0] |= 1;
  w->u[1] |= 1;
  w->d[0] -= 1.0;
  w->d[1] -= 1.0;
}
#endif

/**
 * This function fills the user-specified array with double precision
 * floating point pseudorandom numbers of the IEEE 754 format.
 * @param dsfmt dsfmt state vector.
 * @param array an 128-bit array to be filled by pseudorandom numbers.
 * @param size number of 128-bit pseudorandom numbers to be generated.
 */
inline static void gen_rand_array_c1o2(dsfmt_t *dsfmt, w128_t *array,
                                       int size) {
  int i, j;
  w128_t lung;

  lung = dsfmt->status[DSFMT_N];
  do_recursion(&array[0], &dsfmt->status[0], &dsfmt->status[DSFMT_POS1], &lung);
  for (i = 1; i < DSFMT_N - DSFMT_POS1; i++) {
    do_recursion(&array[i], &dsfmt->status[i], &dsfmt->status[i + DSFMT_POS1],
                 &lung);
  }
  for (; i < DSFMT_N; i++) {
    do_recursion(&array[i], &dsfmt->status[i], &array[i + DSFMT_POS1 - DSFMT_N],
                 &lung);
  }
  for (; i < size - DSFMT_N; i++) {
    do_recursion(&array[i], &array[i - DSFMT_N],
                 &array[i + DSFMT_POS1 - DSFMT_N], &lung);
  }
  for (j = 0; j < 2 * DSFMT_N - size; j++) {
    dsfmt->status[j] = array[j + size - DSFMT_N];
  }
  for (; i < size; i++, j++) {
    do_recursion(&array[i], &array[i - DSFMT_N],
                 &array[i + DSFMT_POS1 - DSFMT_N], &lung);
    dsfmt->status[j] = array[i];
  }
  dsfmt->status[DSFMT_N] = lung;
}

/**
 * This function fills the user-specified array with double precision
 * floating point pseudorandom numbers of the IEEE 754 format.
 * @param dsfmt dsfmt state vector.
 * @param array an 128-bit array to be filled by pseudorandom numbers.
 * @param size number of 128-bit pseudorandom numbers to be generated.
 */
inline static void gen_rand_array_c0o1(dsfmt_t *dsfmt, w128_t *array,
                                       int size) {
  int i, j;
  w128_t lung;

  lung = dsfmt->status[DSFMT_N];
  do_recursion(&array[0], &dsfmt->status[0], &dsfmt->status[DSFMT_POS1], &lung);
  for (i = 1; i < DSFMT_N - DSFMT_POS1; i++) {
    do_recursion(&array[i], &dsfmt->status[i], &dsfmt->status[i + DSFMT_POS1],
                 &lung);
  }
  for (; i < DSFMT_N; i++) {
    do_recursion(&array[i], &dsfmt->status[i], &array[i + DSFMT_POS1 - DSFMT_N],
                 &lung);
  }
  for (; i < size - DSFMT_N; i++) {
    do_recursion(&array[i], &array[i - DSFMT_N],
                 &array[i + DSFMT_POS1 - DSFMT_N], &lung);
    convert_c0o1(&array[i - DSFMT_N]);
  }
  for (j = 0; j < 2 * DSFMT_N - size; j++) {
    dsfmt->status[j] = array[j + size - DSFMT_N];
  }
  for (; i < size; i++, j++) {
    do_recursion(&array[i], &array[i - DSFMT_N],
                 &array[i + DSFMT_POS1 - DSFMT_N], &lung);
    dsfmt->status[j] = array[i];
    convert_c0o1(&array[i - DSFMT_N]);
  }
  for (i = size - DSFMT_N; i < size; i++) {
    convert_c0o1(&array[i]);
  }
  dsfmt->status[DSFMT_N] = lung;
}

/**
 * This function fills the user-specified array with double precision
 * floating point pseudorandom numbers of the IEEE 754 format.
 * @param dsfmt dsfmt state vector.
 * @param array an 128-bit array to be filled by pseudorandom numbers.
 * @param size number of 128-bit pseudorandom numbers to be generated.
 */
inline static void gen_rand_array_o0o1(dsfmt_t *dsfmt, w128_t *array,
                                       int size) {
  int i, j;
  w128_t lung;

  lung = dsfmt->status[DSFMT_N];
  do_recursion(&array[0], &dsfmt->status[0], &dsfmt->status[DSFMT_POS1], &lung);
  for (i = 1; i < DSFMT_N - DSFMT_POS1; i++) {
    do_recursion(&array[i], &dsfmt->status[i], &dsfmt->status[i + DSFMT_POS1],
                 &lung);
  }
  for (; i < DSFMT_N; i++) {
    do_recursion(&array[i], &dsfmt->status[i], &array[i + DSFMT_POS1 - DSFMT_N],
                 &lung);
  }
  for (; i < size - DSFMT_N; i++) {
    do_recursion(&array[i], &array[i - DSFMT_N],
                 &array[i + DSFMT_POS1 - DSFMT_N], &lung);
    convert_o0o1(&array[i - DSFMT_N]);
  }
  for (j = 0; j < 2 * DSFMT_N - size; j++) {
    dsfmt->status[j] = array[j + size - DSFMT_N];
  }
  for (; i < size; i++, j++) {
    do_recursion(&array[i], &array[i - DSFMT_N],
                 &array[i + DSFMT_POS1 - DSFMT_N], &lung);
    dsfmt->status[j] = array[i];
    convert_o0o1(&array[i - DSFMT_N]);
  }
  for (i = size - DSFMT_N; i < size; i++) {
    convert_o0o1(&array[i]);
  }
  dsfmt->status[DSFMT_N] = lung;
}

/**
 * This function fills the user-specified array with double precision
 * floating point pseudorandom numbers of the IEEE 754 format.
 * @param dsfmt dsfmt state vector.
 * @param array an 128-bit array to be filled by pseudorandom numbers.
 * @param size number of 128-bit pseudorandom numbers to be generated.
 */
inline static void gen_rand_array_o0c1(dsfmt_t *dsfmt, w128_t *array,
                                       int size) {
  int i, j;
  w128_t lung;

  lung = dsfmt->status[DSFMT_N];
  do_recursion(&array[0], &dsfmt->status[0], &dsfmt->status[DSFMT_POS1], &lung);
  for (i = 1; i < DSFMT_N - DSFMT_POS1; i++) {
    do_recursion(&array[i], &dsfmt->status[i], &dsfmt->status[i + DSFMT_POS1],
                 &lung);
  }
  for (; i < DSFMT_N; i++) {
    do_recursion(&array[i], &dsfmt->status[i], &array[i + DSFMT_POS1 - DSFMT_N],
                 &lung);
  }
  for (; i < size - DSFMT_N; i++) {
    do_recursion(&array[i], &array[i - DSFMT_N],
                 &array[i + DSFMT_POS1 - DSFMT_N], &lung);
    convert_o0c1(&array[i - DSFMT_N]);
  }
  for (j = 0; j < 2 * DSFMT_N - size; j++) {
    dsfmt->status[j] = array[j + size - DSFMT_N];
  }
  for (; i < size; i++, j++) {
    do_recursion(&array[i], &array[i - DSFMT_N],
                 &array[i + DSFMT_POS1 - DSFMT_N], &lung);
    dsfmt->status[j] = array[i];
    convert_o0c1(&array[i - DSFMT_N]);
  }
  for (i = size - DSFMT_N; i < size; i++) {
    convert_o0c1(&array[i]);
  }
  dsfmt->status[DSFMT_N] = lung;
}

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t ini_func1(uint32_t x) {
  return (x ^ (x >> 27)) * (uint32_t)1664525UL;
}

/**
 * This function represents a function used in the initialization
 * by init_by_array
 * @param x 32-bit integer
 * @return 32-bit integer
 */
static uint32_t ini_func2(uint32_t x) {
  return (x ^ (x >> 27)) * (uint32_t)1566083941UL;
}

/**
 * This function initializes the internal state array to fit the IEEE
 * 754 format.
 * @param dsfmt dsfmt state vector.
 */
static void initial_mask(dsfmt_t *dsfmt) {
  int i;
  uint64_t *psfmt;

  psfmt = &dsfmt->status[0].u[0];
  for (i = 0; i < DSFMT_N * 2; i++) {
    psfmt[i] = (psfmt[i] & DSFMT_LOW_MASK) | DSFMT_HIGH_CONST;
  }
}

/**
 * This function certificate the period of 2^{SFMT_MEXP}-1.
 * @param dsfmt dsfmt state vector.
 */
static void period_certification(dsfmt_t *dsfmt) {
  uint64_t pcv[2] = {DSFMT_PCV1, DSFMT_PCV2};
  uint64_t tmp[2];
  uint64_t inner;
  int i;
#if (DSFMT_PCV2 & 1) != 1
  int j;
  uint64_t work;
#endif

  tmp[0] = (dsfmt->status[DSFMT_N].u[0] ^ DSFMT_FIX1);
  tmp[1] = (dsfmt->status[DSFMT_N].u[1] ^ DSFMT_FIX2);

  inner = tmp[0] & pcv[0];
  inner ^= tmp[1] & pcv[1];
  for (i = 32; i > 0; i >>= 1) {
    inner ^= inner >> i;
  }
  inner &= 1;
  /* check OK */
  if (inner == 1) {
    return;
  }
    /* check NG, and modification */
#if (DSFMT_PCV2 & 1) == 1
  dsfmt->status[DSFMT_N].u[1] ^= 1;
#else
  for (i = 1; i >= 0; i--) {
    work = 1;
    for (j = 0; j < 64; j++) {
      if ((work & pcv[i]) != 0) {
        dsfmt->status[DSFMT_N].u[i] ^= work;
        return;
      }
      work = work << 1;
    }
  }
#endif
  return;
}

/*----------------
  PUBLIC FUNCTIONS
  ----------------*/
/**
 * This function returns the identification string.  The string shows
 * the Mersenne exponent, and all parameters of this generator.
 * @return id string.
 */
const char *dsfmt_get_idstring(void) { return DSFMT_IDSTR; }

/**
 * This function returns the minimum size of array used for \b
 * fill_array functions.
 * @return minimum size of array used for fill_array functions.
 */
int dsfmt_get_min_array_size(void) { return DSFMT_N64; }

/**
 * This function fills the internal state array with double precision
 * floating point pseudorandom numbers of the IEEE 754 format.
 * @param dsfmt dsfmt state vector.
 */
void dsfmt_gen_rand_all(dsfmt_t *dsfmt) {
  int i;
  w128_t lung;

  lung = dsfmt->status[DSFMT_N];
  do_recursion(&dsfmt->status[0], &dsfmt->status[0], &dsfmt->status[DSFMT_POS1],
               &lung);
  for (i = 1; i < DSFMT_N - DSFMT_POS1; i++) {
    do_recursion(&dsfmt->status[i], &dsfmt->status[i],
                 &dsfmt->status[i + DSFMT_POS1], &lung);
  }
  for (; i < DSFMT_N; i++) {
    do_recursion(&dsfmt->status[i], &dsfmt->status[i],
                 &dsfmt->status[i + DSFMT_POS1 - DSFMT_N], &lung);
  }
  dsfmt->status[DSFMT_N] = lung;
}

/**
 * This function generates double precision floating point
 * pseudorandom numbers which distribute in the range [1, 2) to the
 * specified array[] by one call. The number of pseudorandom numbers
 * is specified by the argument \b size, which must be at least (SFMT_MEXP
 * / 128) * 2 and a multiple of two.  The function
 * get_min_array_size() returns this minimum size.  The generation by
 * this function is much faster than the following fill_array_xxx functions.
 *
 * For initialization, init_gen_rand() or init_by_array() must be called
 * before the first call of this function. This function can not be
 * used after calling genrand_xxx functions, without initialization.
 *
 * @param dsfmt dsfmt state vector.
 * @param array an array where pseudorandom numbers are filled
 * by this function.  The pointer to the array must be "aligned"
 * (namely, must be a multiple of 16) in the SIMD version, since it
 * refers to the address of a 128-bit integer.  In the standard C
 * version, the pointer is arbitrary.
 *
 * @param size the number of 64-bit pseudorandom integers to be
 * generated.  size must be a multiple of 2, and greater than or equal
 * to (SFMT_MEXP / 128) * 2.
 *
 * @note \b memalign or \b posix_memalign is available to get aligned
 * memory. Mac OSX doesn't have these functions, but \b malloc of OSX
 * returns the pointer to the aligned memory block.
 */
void dsfmt_fill_array_close1_open2(dsfmt_t *dsfmt, double array[], int size) {
  assert(size % 2 == 0);
  assert(size >= DSFMT_N64);
  gen_rand_array_c1o2(dsfmt, (w128_t *)array, size / 2);
}

/**
 * This function generates double precision floating point
 * pseudorandom numbers which distribute in the range (0, 1] to the
 * specified array[] by one call. This function is the same as
 * fill_array_close1_open2() except the distribution range.
 *
 * @param dsfmt dsfmt state vector.
 * @param array an array where pseudorandom numbers are filled
 * by this function.
 * @param size the number of pseudorandom numbers to be generated.
 * see also \sa fill_array_close1_open2()
 */
void dsfmt_fill_array_open_close(dsfmt_t *dsfmt, double array[], int size) {
  assert(size % 2 == 0);
  assert(size >= DSFMT_N64);
  gen_rand_array_o0c1(dsfmt, (w128_t *)array, size / 2);
}

/**
 * This function generates double precision floating point
 * pseudorandom numbers which distribute in the range [0, 1) to the
 * specified array[] by one call. This function is the same as
 * fill_array_close1_open2() except the distribution range.
 *
 * @param array an array where pseudorandom numbers are filled
 * by this function.
 * @param dsfmt dsfmt state vector.
 * @param size the number of pseudorandom numbers to be generated.
 * see also \sa fill_array_close1_open2()
 */
void dsfmt_fill_array_close_open(dsfmt_t *dsfmt, double array[], int size) {
  assert(size % 2 == 0);
  assert(size >= DSFMT_N64);
  gen_rand_array_c0o1(dsfmt, (w128_t *)array, size / 2);
}

/**
 * This function generates double precision floating point
 * pseudorandom numbers which distribute in the range (0, 1) to the
 * specified array[] by one call. This function is the same as
 * fill_array_close1_open2() except the distribution range.
 *
 * @param dsfmt dsfmt state vector.
 * @param array an array where pseudorandom numbers are filled
 * by this function.
 * @param size the number of pseudorandom numbers to be generated.
 * see also \sa fill_array_close1_open2()
 */
void dsfmt_fill_array_open_open(dsfmt_t *dsfmt, double array[], int size) {
  assert(size % 2 == 0);
  assert(size >= DSFMT_N64);
  gen_rand_array_o0o1(dsfmt, (w128_t *)array, size / 2);
}

#if defined(__INTEL_COMPILER)
#pragma warning(disable : 981)
#endif
/**
 * This function initializes the internal state array with a 32-bit
 * integer seed.
 * @param dsfmt dsfmt state vector.
 * @param seed a 32-bit integer used as the seed.
 * @param mexp caller's mersenne expornent
 */
void dsfmt_chk_init_gen_rand(dsfmt_t *dsfmt, uint32_t seed, int mexp) {
  int i;
  uint32_t *psfmt;

  /* make sure caller program is compiled with the same MEXP */
  if (mexp != dsfmt_mexp) {
    fprintf(stderr, "DSFMT_MEXP doesn't match with dSFMT.c\n");
    exit(1);
  }
  psfmt = &dsfmt->status[0].u32[0];
  psfmt[idxof(0)] = seed;
  for (i = 1; i < (DSFMT_N + 1) * 4; i++) {
    psfmt[idxof(i)] =
        1812433253UL * (psfmt[idxof(i - 1)] ^ (psfmt[idxof(i - 1)] >> 30)) + i;
  }
  initial_mask(dsfmt);
  period_certification(dsfmt);
  dsfmt->idx = DSFMT_N64;
}

/**
 * This function initializes the internal state array,
 * with an array of 32-bit integers used as the seeds
 * @param dsfmt dsfmt state vector.
 * @param init_key the array of 32-bit integers, used as a seed.
 * @param key_length the length of init_key.
 * @param mexp caller's mersenne expornent
 */
void dsfmt_chk_init_by_array(dsfmt_t *dsfmt, uint32_t init_key[],
                             int key_length, int mexp) {
  int i, j, count;
  uint32_t r;
  uint32_t *psfmt32;
  int lag;
  int mid;
  int size = (DSFMT_N + 1) * 4; /* pulmonary */

  /* make sure caller program is compiled with the same MEXP */
  if (mexp != dsfmt_mexp) {
    fprintf(stderr, "DSFMT_MEXP doesn't match with dSFMT.c\n");
    exit(1);
  }
  if (size >= 623) {
    lag = 11;
  } else if (size >= 68) {
    lag = 7;
  } else if (size >= 39) {
    lag = 5;
  } else {
    lag = 3;
  }
  mid = (size - lag) / 2;

  psfmt32 = &dsfmt->status[0].u32[0];
  memset(dsfmt->status, 0x8b, sizeof(dsfmt->status));
  if (key_length + 1 > size) {
    count = key_length + 1;
  } else {
    count = size;
  }
  r = ini_func1(psfmt32[idxof(0)] ^ psfmt32[idxof(mid % size)] ^
                psfmt32[idxof((size - 1) % size)]);
  psfmt32[idxof(mid % size)] += r;
  r += key_length;
  psfmt32[idxof((mid + lag) % size)] += r;
  psfmt32[idxof(0)] = r;
  count--;
  for (i = 1, j = 0; (j < count) && (j < key_length); j++) {
    r = ini_func1(psfmt32[idxof(i)] ^ psfmt32[idxof((i + mid) % size)] ^
                  psfmt32[idxof((i + size - 1) % size)]);
    psfmt32[idxof((i + mid) % size)] += r;
    r += init_key[j] + i;
    psfmt32[idxof((i + mid + lag) % size)] += r;
    psfmt32[idxof(i)] = r;
    i = (i + 1) % size;
  }
  for (; j < count; j++) {
    r = ini_func1(psfmt32[idxof(i)] ^ psfmt32[idxof((i + mid) % size)] ^
                  psfmt32[idxof((i + size - 1) % size)]);
    psfmt32[idxof((i + mid) % size)] += r;
    r += i;
    psfmt32[idxof((i + mid + lag) % size)] += r;
    psfmt32[idxof(i)] = r;
    i = (i + 1) % size;
  }
  for (j = 0; j < size; j++) {
    r = ini_func2(psfmt32[idxof(i)] + psfmt32[idxof((i + mid) % size)] +
                  psfmt32[idxof((i + size - 1) % size)]);
    psfmt32[idxof((i + mid) % size)] ^= r;
    r -= i;
    psfmt32[idxof((i + mid + lag) % size)] ^= r;
    psfmt32[idxof(i)] = r;
    i = (i + 1) % size;
  }
  initial_mask(dsfmt);
  period_certification(dsfmt);
  dsfmt->idx = DSFMT_N64;
}
#if defined(__INTEL_COMPILER)
#pragma warning(default : 981)
#endif

#if defined(__cplusplus)
}
#endif

extern inline double dsfmt_next_double(dsfmt_state *state);

extern inline uint64_t dsfmt_next64(dsfmt_state *state);

extern inline uint32_t dsfmt_next32(dsfmt_state *state);

void dsfmt_jump(dsfmt_state *state) { dSFMT_jump(state->state, poly_128); };