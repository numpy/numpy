/*
 * cl xoroshiro128-benchmark.c xoroshiro128plus.orig.c \
 * ../splitmix64/splitmix64.c /Ox
 *
 * gcc -O3 xoroshiro128-benchmark.c xoroshiro128plus.orig.c \
 * ../splitmix64/splitmix64.c -o  xoroshiro128-benchmark
 *
 */
#include "../splitmix64/splitmix64.h"
#include "xoroshiro128plus.orig.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#define N 1000000000

int main()
{
  uint64_t count = 0, sum = 0;
  uint64_t seed = 0xDEADBEAF;
  s[0] = splitmix64_next(&seed);
  s[1] = splitmix64_next(&seed);
  int i;
  clock_t begin = clock();
  for (i = 0; i < N; i++)
  {
    sum += next();
    count++;
  }
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("0x%" PRIx64 "\ncount: %" PRIu64 "\n", sum, count);
  printf("%" PRIu64 " randoms per second\n",
         (uint64_t)(N / time_spent) / 1000000 * 1000000);
}
