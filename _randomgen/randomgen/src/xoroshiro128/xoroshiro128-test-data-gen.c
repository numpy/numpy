/*
 * Generate testing csv files
 *
 *  cl xoroshiro128-test-data-gen.c xoroshiro128plus.orig.c /
 * ../splitmix64/splitmix64.c /Ox
 * xoroshiro128-test-data-gen.exe *
 *
 *  gcc xoroshiro128-test-data-gen.c xoroshiro128plus.orig.c /
 * ../splitmix64/splitmix64.c -o xoroshiro128-test-data-gen
 *  ./xoroshiro128-test-data-gen
 *
 * Requres the Random123 directory containing header files to be located in the
 * same directory (not included).
 *
 */

#include "../splitmix64/splitmix64.h"
#include "xoroshiro128plus.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  uint64_t sum = 0;
  uint64_t state, seed = 0xDEADBEAF;
  state = seed;
  int i;
  for (i = 0; i < 2; i++) {
    s[i] = splitmix64_next(&state);
  }
  uint64_t store[N];
  for (i = 0; i < N; i++) {
    store[i] = next();
  }

  FILE *fp;
  fp = fopen("xoroshiro128-testset-1.csv", "w");
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

  seed = state = 0;
  for (i = 0; i < 2; i++) {
    s[i] = splitmix64_next(&state);
  }
  for (i = 0; i < N; i++) {
    store[i] = next();
  }
  fp = fopen("xoroshiro128-testset-2.csv", "w");
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
