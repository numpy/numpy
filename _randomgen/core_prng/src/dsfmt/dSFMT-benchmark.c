/*
 *
 * cl dsfmt-benchmark.c dSFMT.c /Ox -DHAVE_SSE2
 *
 * gcc dSFMT-benchmark.c dSFMT.c -O3 -DHAVE_SSE2 -DDSFMT_MEXP=19937 -o
 * dSFMT-benchmark
 */
#include <inttypes.h>
#include <time.h>

#include "dSFMT.h"


#define N 1000000000

int main() {
  int i, j;
  uint32_t seed = 0xDEADBEAF;
  uint64_t count = 0, sum = 0;
  dsfmt_t state;
  double buffer[DSFMT_N64];

  uint64_t out;
  uint64_t *tmp;
  dsfmt_init_gen_rand(&state, seed);
  clock_t begin = clock();
  for (i = 0; i < N / (DSFMT_N64 / 2); i++) {
    dsfmt_fill_array_close_open(&state, &buffer[0], DSFMT_N64);
    for (j = 0; j < DSFMT_N64; j += 2) {
      tmp = (uint64_t *)&buffer[j];
      out = (*tmp >> 16) << 32;
      tmp = (uint64_t *)&buffer[j + 1];
      out |= (*tmp >> 16) & 0xffffffff;
      sum += out;
      count++;
    }
  }
  clock_t end = clock();
  double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  printf("0x%" PRIx64 "\ncount: %" PRIu64 "\n", sum, count);
  printf("%" PRIu64 " randoms per second\n",
         (uint64_t)(N / time_spent) / 1000000 * 1000000);
}
