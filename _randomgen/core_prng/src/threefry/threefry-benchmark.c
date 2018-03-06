/*
 * Simple benchamrk command
 *
 *  cl threefry-benchmark.c /Ox
 *  Measure-Command { .\threefry-benchmark.exe }
 *
 *  gcc threefry-benchmark.c -O3 -o threefry-benchmark
 *  time ./threefry-benchmark
 *
 * Requres the Random123 directory containing header files to be located in the
 * same directory (not included).
 */
#include "Random123/threefry.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000000000

int main() {

  threefry4x64_key_t ctr = {{0, 0, 0, 0}};
  threefry4x64_ctr_t key = {{0xDEADBEAF, 0, 0, 0}};
  threefry4x64_ctr_t out;
  uint64_t sum = 0;
  int i, j;
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;
    out = threefry4x64_R(threefry4x64_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      sum += out.v[j];
    }
  }
  printf("%" PRIu64 "\n", sum);
}