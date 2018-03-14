/*
 * cl pcg64-benchmark.c pcg64.c ../splitmix64/splitmix64.c /Ox
 * Measure-Command { .\xoroshiro128-benchmark.exe }
 *
 * gcc pcg64-benchmark.c pcg64.c ../splitmix64/splitmix64.c -O3 -o
 * pcg64-benchmark
 * time ./pcg64-benchmark
 */
#include "../splitmix64/splitmix64.h"
#include "pcg64.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#define N 1000000000

int main() {
  pcg64_random_t rng;
  uint64_t sum = 0, count = 0;
  uint64_t seed = 0xDEADBEAF;
  int i;
#if __SIZEOF_INT128__ && !defined(PCG_FORCE_EMULATED_128BIT_MATH)
  rng.state = (__uint128_t)splitmix64_next(&seed) << 64;
  rng.state |= splitmix64_next(&seed);
  rng.inc = (__uint128_t)1;
#else
  rng.state.high = splitmix64_next(&seed);
  rng.state.low = splitmix64_next(&seed);
  rng.inc.high = 0;
  rng.inc.low = 1;
#endif
  clock_t begin = clock();
  for (i = 0; i < N; i++) {
    sum += pcg64_random_r(&rng);
    count++;
  }
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("0x%" PRIx64 "\ncount: %" PRIu64 "\n", sum, count);
  printf("%" PRIu64 " randoms per second\n",
         (uint64_t)(N / time_spent) / 1000000 * 1000000);
}
