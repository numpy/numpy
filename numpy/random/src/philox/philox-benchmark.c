/*
 * Simple benchamrk command
 *
 *  cl philox-benchmark.c /Ox
 *
 *  gcc philox-benchmark.c -O3 -o philox-benchmark
 *
 * Requires the Random123 directory containing header files to be located in the
 * same directory (not included).
 */
#include "Random123/philox.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#define N 1000000000

int main() {
  philox4x64_ctr_t ctr = {{0, 0, 0, 0}};
  philox4x64_key_t key = {{0, 0xDEADBEAF}};
  philox4x64_ctr_t out;
  uint64_t count = 0, sum = 0;
  int i, j;
  clock_t begin = clock();
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;
    out = philox4x64_R(philox4x64_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      sum += out.v[j];
      count++;
    }
  }
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("0x%" PRIx64 "\ncount: %" PRIu64 "\n", sum, count);
  printf("%" PRIu64 " randoms per second\n",
         (uint64_t)(N / time_spent) / 1000000 * 1000000);
}
