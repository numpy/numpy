#ifndef _RANDOMDGEN__DISTRIBUTIONS_LEGACY_H_
#define _RANDOMDGEN__DISTRIBUTIONS_LEGACY_H_


#include "numpy/random/distributions.h"

typedef struct aug_bitgen {
  bitgen_t *bit_generator;
  int has_gauss;
  double gauss;
} aug_bitgen_t;

extern double legacy_gauss(aug_bitgen_t *aug_state);
extern double legacy_standard_exponential(aug_bitgen_t *aug_state);
extern double legacy_pareto(aug_bitgen_t *aug_state, double a);
extern double legacy_weibull(aug_bitgen_t *aug_state, double a);
extern double legacy_power(aug_bitgen_t *aug_state, double a);
extern double legacy_gamma(aug_bitgen_t *aug_state, double shape, double scale);
extern double legacy_chisquare(aug_bitgen_t *aug_state, double df);
extern double legacy_rayleigh(bitgen_t *bitgen_state, double mode);
extern double legacy_noncentral_chisquare(aug_bitgen_t *aug_state, double df,
                                          double nonc);
extern double legacy_noncentral_f(aug_bitgen_t *aug_state, double dfnum,
                                  double dfden, double nonc);
extern double legacy_wald(aug_bitgen_t *aug_state, double mean, double scale);
extern double legacy_lognormal(aug_bitgen_t *aug_state, double mean,
                               double sigma);
extern double legacy_standard_t(aug_bitgen_t *aug_state, double df);
extern double legacy_standard_cauchy(aug_bitgen_t *state);
extern double legacy_beta(aug_bitgen_t *aug_state, double a, double b);
extern double legacy_f(aug_bitgen_t *aug_state, double dfnum, double dfden);
extern double legacy_normal(aug_bitgen_t *aug_state, double loc, double scale);
extern double legacy_standard_gamma(aug_bitgen_t *aug_state, double shape);
extern double legacy_exponential(aug_bitgen_t *aug_state, double scale);
extern double legacy_vonmises(bitgen_t *bitgen_state, double mu, double kappa);
extern int64_t legacy_random_binomial(bitgen_t *bitgen_state, double p,
                                      int64_t n, binomial_t *binomial);
extern int64_t legacy_negative_binomial(aug_bitgen_t *aug_state, double n,
                                        double p);
extern int64_t legacy_random_hypergeometric(bitgen_t *bitgen_state,
                                            int64_t good, int64_t bad,
                                            int64_t sample);
extern int64_t legacy_logseries(bitgen_t *bitgen_state, double p);
extern int64_t legacy_random_poisson(bitgen_t *bitgen_state, double lam);
extern int64_t legacy_random_zipf(bitgen_t *bitgen_state, double a);
extern int64_t legacy_random_geometric(bitgen_t *bitgen_state, double p);
void legacy_random_multinomial(bitgen_t *bitgen_state, RAND_INT_TYPE n,
                               RAND_INT_TYPE *mnix, double *pix, npy_intp d,
                               binomial_t *binomial);

#endif
