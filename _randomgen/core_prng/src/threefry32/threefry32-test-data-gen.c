/*
 * Generate testing csv files
 *
 *  cl threefry32-test-data-gen.c /Ox ../splitmix64/splitmix64.c /Ox
 *  threefry32-test-data-gen.exe
 *
 *  gcc threefry32-test-data-gen.c ../splitmix64/splitmix64.c /Ox -o
 * threefry32-test-data-gen
 *  ./threefry32-test-data-gen
 *
 * Requires the Random123 directory containing header files to be located in the
 * same directory (not included).
 *
 */

#include "../splitmix64/splitmix64.h"
#include "Random123/threefry.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  threefry4x32_key_t ctr = {{0, 0, 0, 0}};
  uint64_t state, seed = 0xDEADBEAF;
  state = seed;
  threefry4x32_ctr_t key = {{0}};
  threefry4x32_ctr_t out;
  uint64_t store[N];
  uint64_t seed_val;
  int i, j;
  for (i = 0; i < 4; i++) {
    seed_val = splitmix64_next(&state);
    key.v[2*i] = (uint32_t)seed_val;
    key.v[2*i+1] = (uint32_t)(seed_val >> 32);
  }
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;
    out = threefry4x32_R(threefry4x32_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];
    }
  }

  FILE *fp;
  fp = fopen("threefry32-testset-1.csv", "w");
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

  ctr.v[0] = 0;
  state = seed = 0;
  for (i = 0; i < 4; i++) {
    seed_val = splitmix64_next(&state);
    key.v[2*i] = (uint32_t)seed_val;
    key.v[2*i+1] = (uint32_t)(seed_val >> 32);
  }
  for (i = 0; i < N / 4; i++) {
    ctr.v[0]++;
    out = threefry4x32_R(threefry4x32_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];
    }
  }

  fp = fopen("threefry32-testset-2.csv", "w");
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
