/*
 * cl xorshift1024-benchmark.c xorshift2014.orig.c
 *     ../splitmix64/splitmix64.c /Ox
 *
 * gcc -O3 xorshift1024-benchmark.c xorshift2014.orig.c /
 * ../splitmix64/splitmix64.c -o xorshift1024-benchmark
 *
 */
#include "../splitmix64/splitmix64.h"
#include "xorshift1024.orig.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

#define N 1000000000

int main() {
  uint64_t count = 0, sum = 0;
  uint64_t seed = 0xDEADBEAF;
  int i;
  for (i = 0; i < 16; i++) {
    s[i] = splitmix64_next(&seed);
  }
  p = 0;
  clock_t begin = clock();
  for (i = 0; i < N; i++) {
    sum += next();
    count++;
  }
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("0x%" PRIx64 "\ncount: %" PRIu64 "\n", sum, count);
  printf("%" PRIu64 " randoms per second\n",
         (uint64_t)(N / time_spent) / 1000000 * 1000000);
}
