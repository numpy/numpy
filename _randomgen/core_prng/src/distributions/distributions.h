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

typedef double (*prng_double)(void *st);
typedef float (*prng_float)(void *st);
typedef uint32_t (*prng_uint32)(void *st);
typedef uint64_t (*prng_uint64)(void *st);

typedef struct prng {
  void *state;
  void *next_uint64;
  void *next_uint32;
  void *next_double;
} prng_t;

float random_float(void *void_state);

double random_double(void *void_state);

uint32_t random_uint32(void *void_state);

double random_standard_exponential(void *void_state);