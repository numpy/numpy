/*
 * Generate testing csv files
 *
 *  cl threefry-test-data-gen.c /Ox ../splitmix64/splitmix64.c /Ox
 *  threefry-test-data-gen.exe
 *
 *  gcc threefry-test-data-gen.c ../splitmix64/splitmix64.c /Ox -o
 * threefry-test-data-gen
 *  ./threefry-test-data-gen
 *
 * Requres the Random123 directory containing header files to be located in the
 * same directory (not included).
 *
 */

#include "../splitmix64/splitmix64.h"
#include "Random123/threefry.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  threefry4x64_key_t ctr = {{0, 0, 0, 0}};
  uint64_t state, seed = 0xDEADBEAF;
  state = seed;
  threefry4x64_ctr_t key = {{0}};
  threefry4x64_ctr_t out;
  uint64_t store[N];
  int i, j;
  for (i = 0; i < 4; i++) {
    key.v[i] = splitmix64_next(&state);
  }
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;
    out = threefry4x64_R(threefry4x64_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];
    }
  }

  FILE *fp;
  fp = fopen("threefry-testset-1.csv", "w");
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
    key.v[i] = splitmix64_next(&state);
  }
  for (i = 0; i < N / 4; i++) {
    ctr.v[0]++;
    out = threefry4x64_R(threefry4x64_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];
    }
  }

  fp = fopen("threefry-testset-2.csv", "w");
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
