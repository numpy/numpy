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

#include "Python.h"
#include "numpy/npy_common.h"
#include <math.h>

#ifdef _WIN32
#if _MSC_VER == 1500

static NPY_INLINE int64_t llabs(int64_t x) {
  int64_t o;
  if (x < 0) {
    o = -x;
  } else {
    o = x;
  }
  return o;
}
#endif
#endif

#ifdef DLL_EXPORT
#define DECLDIR __declspec(dllexport)
#else
#define DECLDIR extern
#endif

#ifndef min
#define min(x, y) ((x < y) ? x : y)
#define max(x, y) ((x > y) ? x : y)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

typedef struct s_binomial_t {
  int has_binomial; /* !=0: following parameters initialized for binomial */
  double psave;
  int64_t nsave;
  double r;
  double q;
  double fm;
  int64_t m;
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

typedef struct brng {
  void *state;
  uint64_t (*next_uint64)(void *st);
  uint32_t (*next_uint32)(void *st);
  double (*next_double)(void *st);
  uint64_t (*next_raw)(void *st);
} brng_t;

/* Inline generators for internal use */
static NPY_INLINE uint32_t next_uint32(brng_t *brng_state) {
  return brng_state->next_uint32(brng_state->state);
}

static NPY_INLINE uint64_t next_uint64(brng_t *brng_state) {
  return brng_state->next_uint64(brng_state->state);
}

static NPY_INLINE float next_float(brng_t *brng_state) {
  return (next_uint32(brng_state) >> 9) * (1.0f / 8388608.0f);
}

static NPY_INLINE double next_double(brng_t *brng_state) {
  return brng_state->next_double(brng_state->state);
}

DECLDIR float random_float(brng_t *brng_state);
DECLDIR double random_double(brng_t *brng_state);
DECLDIR void random_double_fill(brng_t* brng_state, npy_intp cnt, double *out);

DECLDIR int64_t random_positive_int64(brng_t *brng_state);
DECLDIR int32_t random_positive_int32(brng_t *brng_state);
DECLDIR int64_t random_positive_int(brng_t *brng_state);
DECLDIR uint64_t random_uint(brng_t *brng_state);

DECLDIR double random_standard_exponential(brng_t *brng_state);
DECLDIR void random_standard_exponential_fill(brng_t *brng_state, npy_intp cnt, double *out);
DECLDIR float random_standard_exponential_f(brng_t *brng_state);
DECLDIR double random_standard_exponential_zig(brng_t *brng_state);
DECLDIR void random_standard_exponential_zig_fill(brng_t *brng_state, npy_intp cnt, double *out);
DECLDIR float random_standard_exponential_zig_f(brng_t *brng_state);

/*
DECLDIR double random_gauss(brng_t *brng_state);
DECLDIR float random_gauss_f(brng_t *brng_state);
*/
DECLDIR double random_gauss_zig(brng_t *brng_state);
DECLDIR float random_gauss_zig_f(brng_t *brng_state);
DECLDIR void random_gauss_zig_fill(brng_t *brng_state, npy_intp cnt, double *out);

/*
DECLDIR double random_standard_gamma(brng_t *brng_state, double shape);
DECLDIR float random_standard_gamma_f(brng_t *brng_state, float shape);
*/
DECLDIR double random_standard_gamma_zig(brng_t *brng_state, double shape);
DECLDIR float random_standard_gamma_zig_f(brng_t *brng_state, float shape);

/*
DECLDIR double random_normal(brng_t *brng_state, double loc, double scale);
*/
DECLDIR double random_normal_zig(brng_t *brng_state, double loc, double scale);

DECLDIR double random_gamma(brng_t *brng_state, double shape, double scale);
DECLDIR float random_gamma_float(brng_t *brng_state, float shape, float scale);

DECLDIR double random_exponential(brng_t *brng_state, double scale);
DECLDIR double random_uniform(brng_t *brng_state, double lower, double range);
DECLDIR double random_beta(brng_t *brng_state, double a, double b);
DECLDIR double random_chisquare(brng_t *brng_state, double df);
DECLDIR double random_f(brng_t *brng_state, double dfnum, double dfden);
DECLDIR double random_standard_cauchy(brng_t *brng_state);
DECLDIR double random_pareto(brng_t *brng_state, double a);
DECLDIR double random_weibull(brng_t *brng_state, double a);
DECLDIR double random_power(brng_t *brng_state, double a);
DECLDIR double random_laplace(brng_t *brng_state, double loc, double scale);
DECLDIR double random_gumbel(brng_t *brng_state, double loc, double scale);
DECLDIR double random_logistic(brng_t *brng_state, double loc, double scale);
DECLDIR double random_lognormal(brng_t *brng_state, double mean, double sigma);
DECLDIR double random_rayleigh(brng_t *brng_state, double mode);
DECLDIR double random_standard_t(brng_t *brng_state, double df);
DECLDIR double random_noncentral_chisquare(brng_t *brng_state, double df,
                                           double nonc);
DECLDIR double random_noncentral_f(brng_t *brng_state, double dfnum,
                                   double dfden, double nonc);
DECLDIR double random_wald(brng_t *brng_state, double mean, double scale);
DECLDIR double random_vonmises(brng_t *brng_state, double mu, double kappa);
DECLDIR double random_triangular(brng_t *brng_state, double left, double mode,
                                 double right);

DECLDIR int64_t random_poisson(brng_t *brng_state, double lam);
DECLDIR int64_t random_negative_binomial(brng_t *brng_state, double n,
                                         double p);
DECLDIR int64_t random_binomial(brng_t *brng_state, double p, int64_t n,
                                binomial_t *binomial);
DECLDIR int64_t random_logseries(brng_t *brng_state, double p);
DECLDIR int64_t random_geometric_search(brng_t *brng_state, double p);
DECLDIR int64_t random_geometric_inversion(brng_t *brng_state, double p);
DECLDIR int64_t random_geometric(brng_t *brng_state, double p);
DECLDIR int64_t random_zipf(brng_t *brng_state, double a);
DECLDIR int64_t random_hypergeometric(brng_t *brng_state, int64_t good,
                                      int64_t bad, int64_t sample);

DECLDIR uint64_t random_interval(brng_t *brng_state, uint64_t max);
DECLDIR uint64_t random_bounded_uint64(brng_t *brng_state, uint64_t off,
                                       uint64_t rng, uint64_t mask);
DECLDIR uint32_t random_buffered_bounded_uint32(brng_t *brng_state,
                                                uint32_t off, uint32_t rng,
                                                uint32_t mask, int *bcnt,
                                                uint32_t *buf);

DECLDIR uint16_t random_buffered_bounded_uint16(brng_t *brng_state,
                                                uint16_t off, uint16_t rng,
                                                uint16_t mask, int *bcnt,
                                                uint32_t *buf);
DECLDIR uint8_t random_buffered_bounded_uint8(brng_t *brng_state, uint8_t off,
                                              uint8_t rng, uint8_t mask,
                                              int *bcnt, uint32_t *buf);
DECLDIR npy_bool random_buffered_bounded_bool(brng_t *brng_state, npy_bool off,
                                              npy_bool rng, npy_bool mask,
                                              int *bcnt, uint32_t *buf);
DECLDIR void random_bounded_uint64_fill(brng_t *brng_state, uint64_t off,
                                        uint64_t rng, npy_intp cnt,
                                        uint64_t *out);
DECLDIR void random_bounded_uint32_fill(brng_t *brng_state, uint32_t off,
                                        uint32_t rng, npy_intp cnt,
                                        uint32_t *out);
DECLDIR void random_bounded_uint16_fill(brng_t *brng_state, uint16_t off,
                                        uint16_t rng, npy_intp cnt,
                                        uint16_t *out);
DECLDIR void random_bounded_uint8_fill(brng_t *brng_state, uint8_t off,
                                       uint8_t rng, npy_intp cnt, uint8_t *out);
DECLDIR void random_bounded_bool_fill(brng_t *brng_state, npy_bool off,
                                      npy_bool rng, npy_intp cnt,
                                      npy_bool *out);
