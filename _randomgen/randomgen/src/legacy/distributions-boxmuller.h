#ifndef _RANDOMDGEN__DISTRIBUTIONS_LEGACY_H_
#define _RANDOMDGEN__DISTRIBUTIONS_LEGACY_H_


#include "../distributions/distributions.h"

typedef struct aug_brng {
  brng_t *basicrng;
  int has_gauss;
  double gauss;
} aug_brng_t;

extern double legacy_gauss(aug_brng_t *aug_state);
extern double legacy_standard_exponential(aug_brng_t *aug_state);
extern double legacy_pareto(aug_brng_t *aug_state, double a);
extern double legacy_weibull(aug_brng_t *aug_state, double a);
extern double legacy_power(aug_brng_t *aug_state, double a);
extern double legacy_gamma(aug_brng_t *aug_state, double shape, double scale);
extern double legacy_pareto(aug_brng_t *aug_state, double a);
extern double legacy_weibull(aug_brng_t *aug_state, double a);
extern double legacy_chisquare(aug_brng_t *aug_state, double df);
extern double legacy_noncentral_chisquare(aug_brng_t *aug_state, double df,
                                          double nonc);

extern double legacy_noncentral_f(aug_brng_t *aug_state, double dfnum,
                                  double dfden, double nonc);
extern double legacy_wald(aug_brng_t *aug_state, double mean, double scale);
extern double legacy_lognormal(aug_brng_t *aug_state, double mean,
                               double sigma);
extern double legacy_standard_t(aug_brng_t *aug_state, double df);
extern int64_t legacy_negative_binomial(aug_brng_t *aug_state, double n,
                                        double p);
extern double legacy_standard_cauchy(aug_brng_t *state);
extern double legacy_beta(aug_brng_t *aug_state, double a, double b);
extern double legacy_f(aug_brng_t *aug_state, double dfnum, double dfden);
extern double legacy_normal(aug_brng_t *aug_state, double loc, double scale);
extern double legacy_standard_gamma(aug_brng_t *aug_state, double shape);
extern double legacy_exponential(aug_brng_t *aug_state, double scale);

#endif
