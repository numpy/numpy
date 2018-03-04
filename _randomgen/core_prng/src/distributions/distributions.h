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
#include "numpy/npy_common.h"

typedef double (*random_double_0)(void *st);
typedef float (*random_float_0)(void *st);

typedef struct s_binomial_t
{
    int has_binomial; /* !=0: following parameters initialized for binomial */
    double psave;
    long nsave;
    double r;
    double q;
    double fm;
    long m;
    double p1;
    double xm;
    double xl;
    double xr;
    double c;
    double laml;
    double lamr;
    double p2;
    double p3;
    double p4;
} binomial_t;

typedef struct prng {
  void *state;
  uint64_t (*next_uint64)(void *st);
  uint32_t (*next_uint32)(void *st);
  double (*next_double)(void *st);
  int has_gauss;
  double gauss;
  int has_gauss_f;
  float gauss_f;
  binomial_t *binomial;
} prng_t;

float random_float(prng_t *prng_state);

double random_double(prng_t *prng_state);

uint32_t random_uint32(prng_t *prng_state);

double random_standard_exponential(prng_t *prng_state);

float random_standard_exponential_float(prng_t *prng_state);

double random_gauss(prng_t *prng_state);

float random_gauss_float(prng_t *prng_state);

double random_standard_exponential_zig_double(prng_t *prng_state);