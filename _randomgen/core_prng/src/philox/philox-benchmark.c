/*
 * Simple benchamrk command
 *
 *  cl philox-benchmark.c /Ox
 *  Measure-Command { .\philox-benchmark.exe }
 *
 *  gcc philox-benchmark.c -O3 -o philox-benchmark
 *  time ./philox-benchmark
 *
 * Requres the Random123 directory containing header files to be located in the
 * same directory (not included).
 */
#include "Random123/philox.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000000000

int main() {
  philox4x64_ctr_t ctr = {{0, 0, 0, 0}};
  philox4x64_key_t key = {{0, 0xDEADBEAF}};
  philox4x64_ctr_t out;
  uint64_t sum = 0;
  int i, j;
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;
    out = philox4x64_R(philox4x64_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      sum += out.v[j];
    }
  }
  printf("%" PRIu64 "\n", sum);
}