#include <stddef.h>
#ifdef _WIN32
#if _MSC_VER == 1500
#include "../common/stdint.h"
typedef int bool;
#define false 0
#define true 1
#else
#include <stdbool.h>
#include <stdint.h>
#endif
#else
#include <stdbool.h>
#include <stdint.h>
#endif

#include <math.h>

typedef double (*random_double_0)(void *st);
typedef float (*random_float_0)(void *st);

typedef struct prng {
  void *state;
  uint64_t (*next_uint64)(void *st);
  uint32_t (*next_uint32)(void *st);
  double (*next_double)(void *st);
} prng_t;

float random_float(prng_t *prng_state);

double random_double(prng_t *prng_state);

uint32_t random_uint32(prng_t *prng_state);

double random_standard_exponential(prng_t *prng_state);