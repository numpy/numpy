/*
 * Generate testing csv files
 *
 * GCC only
 *
 * gcc  pcg64-test-data-gen.c pcg64.orig.c ../splitmix64/splitmix64.c -o
 * pgc64-test-data-gen
 */

#include "pcg64.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  pcg64_random_t rng;
  uint64_t state, seed = 0xDEADBEAF;
  state = seed;
  __uint128_t temp, s, inc;
  int i;
  uint64_t store[N];
  s = (__uint128_t)seed;
  inc = (__uint128_t)0;
  pcg64_srandom_r(&rng, s, inc);
  printf("0x%" PRIx64, (uint64_t)(rng.state >> 64));
  printf("%" PRIx64 "\n", (uint64_t)rng.state);
  printf("0x%" PRIx64, (uint64_t)(rng.inc >> 64));
  printf("%" PRIx64 "\n", (uint64_t)rng.inc);
  for (i = 0; i < N; i++) {
    store[i] = pcg64_random_r(&rng);
  }

  FILE *fp;
  fp = fopen("pcg64-testset-1.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);

  state = seed = 0;
  s = (__uint128_t)seed;
  i = (__uint128_t)0;
  pcg64_srandom_r(&rng, s, i);
  printf("0x%" PRIx64, (uint64_t)(rng.state >> 64));
  printf("%" PRIx64 "\n", (uint64_t)rng.state);
  printf("0x%" PRIx64, (uint64_t)(rng.inc >> 64));
  printf("%" PRIx64 "\n", (uint64_t)rng.inc);
  for (i = 0; i < N; i++) {
    store[i] = pcg64_random_r(&rng);
  }
  fp = fopen("pcg64-testset-2.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);
}
