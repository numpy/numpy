/*
 * Generate testing csv files
 *
 *
 * gcc  pcg32-test-data-gen.c pcg32.orig.c ../splitmix64/splitmix64.c -o
 * pgc64-test-data-gen
 */

#include "pcg_variants.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  pcg32_random_t rng;
  uint64_t inc, seed = 0xDEADBEAF;
  inc = 0;
  int i;
  uint64_t store[N];
  pcg32_srandom_r(&rng, seed, inc);
  for (i = 0; i < N; i++) {
    store[i] = pcg32_random_r(&rng);
  }

  FILE *fp;
  fp = fopen("pcg32-testset-1.csv", "w");
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

  seed = 0;
  pcg32_srandom_r(&rng, seed, inc);
  for (i = 0; i < N; i++) {
    store[i] = pcg32_random_r(&rng);
  }
  fp = fopen("pcg32-testset-2.csv", "w");
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
