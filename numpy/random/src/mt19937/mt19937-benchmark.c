/*
 * cl mt19937-benchmark.c mt19937.c /Ox
 * Measure-Command { .\mt19937-benchmark.exe }
 *
 * gcc mt19937-benchmark.c mt19937.c -O3 -o mt19937-benchmark
 * time ./mt19937-benchmark
 */
#include "mt19937.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#define Q 1000000000

int main() {
  int i;
  uint32_t seed = 0x0;
  uint64_t sum = 0, count = 0;
  mt19937_state state;
  mt19937_seed(&state, seed);
  clock_t begin = clock();
  for (i = 0; i < Q; i++) {
    sum += mt19937_next64(&state);
    count++;
  }
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("0x%" PRIx64 "\ncount: %" PRIu64 "\n", sum, count);
  printf("%" PRIu64 " randoms per second\n",
         (uint64_t)(Q / time_spent) / 1000000 * 1000000);
}
