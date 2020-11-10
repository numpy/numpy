#ifndef _RANDOMDGEN__DISTRIBUTIONS_H_
#define _RANDOMDGEN__DISTRIBUTIONS_H_

#include "Python.h"
#include "numpy/npy_common.h"
#include <stddef.h>
#include <stdbool.h>
#ifdef __VMS
#if __CRTL_VER > 80400000
#include <stdint.h>
#else
#include <inttypes.h>
typedef signed char             int8_t;
typedef signed short            int16_t;
typedef signed int              int32_t;
typedef signed char             int_least8_t;
typedef signed short            int_least16_t;
typedef signed int              int_least32_t;
typedef signed long long int    int_least64_t;
typedef signed char             int_fast8_t;
typedef signed int              int_fast16_t;
typedef signed int              int_fast32_t;
typedef signed long long int    int_fast64_t;
typedef signed long long int    intmax_t;
typedef unsigned char           uint8_t;
typedef unsigned short          uint16_t;
typedef unsigned int            uint32_t;
typedef unsigned char           uint_least8_t;
typedef unsigned short          uint_least16_t;
typedef unsigned int            uint_least32_t;
typedef unsigned long long int  uint_least64_t;
typedef unsigned char           uint_fast8_t;
typedef unsigned int            uint_fast16_t;
typedef unsigned int            uint_fast32_t;
typedef unsigned long long int  uint_fast64_t;
typedef unsigned long long int  uintmax_t;

#define INT8_MIN        (-127-1)
#define INT16_MIN       (-32767-1)
#define INT32_MIN       (-2147483647-1)
#define INT64_MIN       (-9223372036854775807LL-1)
#define INTMAX_MIN      INT64_MIN

#define INT_LEAST8_MIN  INT8_MIN
#define INT_LEAST16_MIN INT16_MIN
#define INT_LEAST32_MIN INT32_MIN
#define INT_LEAST64_MIN INT64_MIN

#define INT_FAST8_MIN   INT8_MIN
#define INT_FAST16_MIN  INT32_MIN
#define INT_FAST32_MIN  INT32_MIN
#define INT_FAST64_MIN  INT64_MIN

#define INT8_MAX        (127)
#define INT16_MAX       (32767)
#define INT32_MAX       (2147483647)
#define INT64_MAX       (9223372036854775807LL)
#define INTMAX_MAX      INT64_MAX

#define INT_LEAST8_MAX  INT8_MAX
#define INT_LEAST16_MAX INT16_MAX
#define INT_LEAST32_MAX INT32_MAX
#define INT_LEAST64_MAX INT64_MAX

#define INT_FAST8_MAX   INT8_MAX
#define INT_FAST16_MAX  INT32_MAX
#define INT_FAST32_MAX  INT32_MAX
#define INT_FAST64_MAX  INT64_MAX

#define UINT8_MAX       (255)
#define UINT16_MAX      (65535)
#define UINT32_MAX      (4294967295UL)
#define UINT64_MAX      (18446744073709551615ULL)
#define UINTMAX_MAX     UINT64_MAX

#define UINT_LEAST8_MAX     UINT8_MAX
#define UINT_LEAST16_MAX    UINT16_MAX
#define UINT_LEAST32_MAX    UINT32_MAX
#define UINT_LEAST64_MAX    UINT64_MAX

#define UINT_FAST8_MAX      UINT8_MAX
#define UINT_FAST16_MAX     UINT32_MAX
#define UINT_FAST32_MAX     UINT32_MAX
#define UINT_FAST64_MAX     UINT64_MAX

#endif
#else
#include <stdint.h>
#endif


#include "numpy/npy_math.h"
#include "numpy/random/bitgen.h"

/*
 * RAND_INT_TYPE is used to share integer generators with RandomState which
 * used long in place of int64_t. If changing a distribution that uses
 * RAND_INT_TYPE, then the original unmodified copy must be retained for
 * use in RandomState by copying to the legacy distributions source file.
 */
#ifdef NP_RANDOM_LEGACY
#define RAND_INT_TYPE long
#define RAND_INT_MAX LONG_MAX
#else
#define RAND_INT_TYPE int64_t
#define RAND_INT_MAX INT64_MAX
#endif

#ifdef _MSC_VER
#define DECLDIR __declspec(dllexport)
#else
#define DECLDIR extern
#endif

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? x : y)
#define MAX(x, y) (((x) > (y)) ? x : y)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338328
#endif

typedef struct s_binomial_t {
  int has_binomial; /* !=0: following parameters initialized for binomial */
  double psave;
  RAND_INT_TYPE nsave;
  double r;
  double q;
  double fm;
  RAND_INT_TYPE m;
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

DECLDIR float random_standard_uniform_f(bitgen_t *bitgen_state);
DECLDIR double random_standard_uniform(bitgen_t *bitgen_state);
DECLDIR void random_standard_uniform_fill(bitgen_t *, npy_intp, double *);
DECLDIR void random_standard_uniform_fill_f(bitgen_t *, npy_intp, float *);

DECLDIR int64_t random_positive_int64(bitgen_t *bitgen_state);
DECLDIR int32_t random_positive_int32(bitgen_t *bitgen_state);
DECLDIR int64_t random_positive_int(bitgen_t *bitgen_state);
DECLDIR uint64_t random_uint(bitgen_t *bitgen_state);

DECLDIR double random_standard_exponential(bitgen_t *bitgen_state);
DECLDIR float random_standard_exponential_f(bitgen_t *bitgen_state);
DECLDIR void random_standard_exponential_fill(bitgen_t *, npy_intp, double *);
DECLDIR void random_standard_exponential_fill_f(bitgen_t *, npy_intp, float *);
DECLDIR void random_standard_exponential_inv_fill(bitgen_t *, npy_intp, double *);
DECLDIR void random_standard_exponential_inv_fill_f(bitgen_t *, npy_intp, float *);

DECLDIR double random_standard_normal(bitgen_t *bitgen_state);
DECLDIR float random_standard_normal_f(bitgen_t *bitgen_state);
DECLDIR void random_standard_normal_fill(bitgen_t *, npy_intp, double *);
DECLDIR void random_standard_normal_fill_f(bitgen_t *, npy_intp, float *);
DECLDIR double random_standard_gamma(bitgen_t *bitgen_state, double shape);
DECLDIR float random_standard_gamma_f(bitgen_t *bitgen_state, float shape);

DECLDIR double random_normal(bitgen_t *bitgen_state, double loc, double scale);

DECLDIR double random_gamma(bitgen_t *bitgen_state, double shape, double scale);
DECLDIR float random_gamma_f(bitgen_t *bitgen_state, float shape, float scale);

DECLDIR double random_exponential(bitgen_t *bitgen_state, double scale);
DECLDIR double random_uniform(bitgen_t *bitgen_state, double lower, double range);
DECLDIR double random_beta(bitgen_t *bitgen_state, double a, double b);
DECLDIR double random_chisquare(bitgen_t *bitgen_state, double df);
DECLDIR double random_f(bitgen_t *bitgen_state, double dfnum, double dfden);
DECLDIR double random_standard_cauchy(bitgen_t *bitgen_state);
DECLDIR double random_pareto(bitgen_t *bitgen_state, double a);
DECLDIR double random_weibull(bitgen_t *bitgen_state, double a);
DECLDIR double random_power(bitgen_t *bitgen_state, double a);
DECLDIR double random_laplace(bitgen_t *bitgen_state, double loc, double scale);
DECLDIR double random_gumbel(bitgen_t *bitgen_state, double loc, double scale);
DECLDIR double random_logistic(bitgen_t *bitgen_state, double loc, double scale);
DECLDIR double random_lognormal(bitgen_t *bitgen_state, double mean, double sigma);
DECLDIR double random_rayleigh(bitgen_t *bitgen_state, double mode);
DECLDIR double random_standard_t(bitgen_t *bitgen_state, double df);
DECLDIR double random_noncentral_chisquare(bitgen_t *bitgen_state, double df,
                                           double nonc);
DECLDIR double random_noncentral_f(bitgen_t *bitgen_state, double dfnum,
                                   double dfden, double nonc);
DECLDIR double random_wald(bitgen_t *bitgen_state, double mean, double scale);
DECLDIR double random_vonmises(bitgen_t *bitgen_state, double mu, double kappa);
DECLDIR double random_triangular(bitgen_t *bitgen_state, double left, double mode,
                                 double right);

DECLDIR RAND_INT_TYPE random_poisson(bitgen_t *bitgen_state, double lam);
DECLDIR RAND_INT_TYPE random_negative_binomial(bitgen_t *bitgen_state, double n,
                                 double p);

DECLDIR int64_t random_binomial(bitgen_t *bitgen_state, double p,
                                int64_t n, binomial_t *binomial);

DECLDIR RAND_INT_TYPE random_logseries(bitgen_t *bitgen_state, double p);
DECLDIR RAND_INT_TYPE random_geometric(bitgen_t *bitgen_state, double p);
DECLDIR RAND_INT_TYPE random_zipf(bitgen_t *bitgen_state, double a);
DECLDIR int64_t random_hypergeometric(bitgen_t *bitgen_state,
                                      int64_t good, int64_t bad, int64_t sample);
DECLDIR uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max);

/* Generate random uint64 numbers in closed interval [off, off + rng]. */
DECLDIR uint64_t random_bounded_uint64(bitgen_t *bitgen_state, uint64_t off,
                                       uint64_t rng, uint64_t mask,
                                       bool use_masked);

/* Generate random uint32 numbers in closed interval [off, off + rng]. */
DECLDIR uint32_t random_buffered_bounded_uint32(bitgen_t *bitgen_state,
                                                uint32_t off, uint32_t rng,
                                                uint32_t mask, bool use_masked,
                                                int *bcnt, uint32_t *buf);
DECLDIR uint16_t random_buffered_bounded_uint16(bitgen_t *bitgen_state,
                                                uint16_t off, uint16_t rng,
                                                uint16_t mask, bool use_masked,
                                                int *bcnt, uint32_t *buf);
DECLDIR uint8_t random_buffered_bounded_uint8(bitgen_t *bitgen_state, uint8_t off,
                                              uint8_t rng, uint8_t mask,
                                              bool use_masked, int *bcnt,
                                              uint32_t *buf);
DECLDIR npy_bool random_buffered_bounded_bool(bitgen_t *bitgen_state, npy_bool off,
                                              npy_bool rng, npy_bool mask,
                                              bool use_masked, int *bcnt,
                                              uint32_t *buf);

DECLDIR void random_bounded_uint64_fill(bitgen_t *bitgen_state, uint64_t off,
                                        uint64_t rng, npy_intp cnt,
                                        bool use_masked, uint64_t *out);
DECLDIR void random_bounded_uint32_fill(bitgen_t *bitgen_state, uint32_t off,
                                        uint32_t rng, npy_intp cnt,
                                        bool use_masked, uint32_t *out);
DECLDIR void random_bounded_uint16_fill(bitgen_t *bitgen_state, uint16_t off,
                                        uint16_t rng, npy_intp cnt,
                                        bool use_masked, uint16_t *out);
DECLDIR void random_bounded_uint8_fill(bitgen_t *bitgen_state, uint8_t off,
                                       uint8_t rng, npy_intp cnt,
                                       bool use_masked, uint8_t *out);
DECLDIR void random_bounded_bool_fill(bitgen_t *bitgen_state, npy_bool off,
                                      npy_bool rng, npy_intp cnt,
                                      bool use_masked, npy_bool *out);

DECLDIR void random_multinomial(bitgen_t *bitgen_state, RAND_INT_TYPE n, RAND_INT_TYPE *mnix,
                                double *pix, npy_intp d, binomial_t *binomial);

/* multivariate hypergeometric, "count" method */
DECLDIR int random_multivariate_hypergeometric_count(bitgen_t *bitgen_state,
                              int64_t total,
                              size_t num_colors, int64_t *colors,
                              int64_t nsample,
                              size_t num_variates, int64_t *variates);

/* multivariate hypergeometric, "marginals" method */
DECLDIR void random_multivariate_hypergeometric_marginals(bitgen_t *bitgen_state,
                                   int64_t total,
                                   size_t num_colors, int64_t *colors,
                                   int64_t nsample,
                                   size_t num_variates, int64_t *variates);

/* Common to legacy-distributions.c and distributions.c but not exported */

RAND_INT_TYPE random_binomial_btpe(bitgen_t *bitgen_state,
                                   RAND_INT_TYPE n,
                                   double p,
                                   binomial_t *binomial);
RAND_INT_TYPE random_binomial_inversion(bitgen_t *bitgen_state,
                                        RAND_INT_TYPE n,
                                        double p,
                                        binomial_t *binomial);
double random_loggam(double x);
static NPY_INLINE double next_double(bitgen_t *bitgen_state) {
    return bitgen_state->next_double(bitgen_state->state);
}

#endif
