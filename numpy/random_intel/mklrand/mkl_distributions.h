#include <stddef.h>
#include "randomkit.h"

#ifndef _MKL_DISTRIBUTIONS_H_
#define _MKL_DISTRIBUTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MATRIX = 0,
    PACKED = 1,
    DIAGONAL = 2
} ch_st_enum;

extern void vrk_double_vec(vrk_state *state, const int len, double *res);
extern void vrk_uniform_vec(vrk_state *state, const int len, double *res, const double low, const double high);
extern void vrk_standard_normal_vec_ICDF(vrk_state *state, const int len, double *res);
extern void vrk_standard_normal_vec_BM1(vrk_state *state, const int len, double *res);
extern void vrk_standard_normal_vec_BM2(vrk_state *state, const int len, double *res);

extern void vrk_normal_vec_ICDF(vrk_state *state, const int len, double *res, const double loc, const double scale);
extern void vrk_normal_vec_BM1(vrk_state *state, const int len, double *res, const double loc, const double scale);
extern void vrk_normal_vec_BM2(vrk_state *state, const int len, double *res, const double loc, const double scale);

extern void vrk_standard_t_vec(vrk_state *state, const int len, double *res, const double df);
extern void vrk_chisquare_vec(vrk_state *state, const int len, double *res, const double df);

extern void vrk_standard_exponential_vec(vrk_state *state, int len, double *res);
extern void vrk_standard_cauchy_vec(vrk_state *state, int len, double *res);

extern void vrk_standard_gamma_vec(vrk_state *state, const int len, double *res, const double shape);
extern void vrk_exponential_vec(vrk_state *state, const int len, double *res, const double scale);
extern void vrk_gamma_vec(vrk_state *state, const int len, double *res, const double shape, const double scale);

extern void vrk_pareto_vec(vrk_state *state, const int len, double *res, const double alph);
extern void vrk_power_vec(vrk_state *state, const int len, double *res, const double alph);

extern void vrk_weibull_vec(vrk_state *state, const int len, double *res, const double alph);

extern void vrk_rayleigh_vec(vrk_state *state, const int len, double *res, const double scale);

extern void vrk_beta_vec(vrk_state *state, const int len, double *res, const double p, const double q);
extern void vrk_f_vec(vrk_state *state, const int len, double *res, const double df_num, const double df_den);

extern void vrk_noncentral_chisquare_vec(vrk_state *state, const int len, double *res, const double df, const double nonc);

extern void vrk_laplace_vec(vrk_state *vec, const int len, double *res, const double loc, const double scale);
extern void vrk_gumbel_vec(vrk_state *vec, const int len, double *res, const double loc, const double scale);
extern void vrk_logistic_vec(vrk_state *vec, const int len, double *res, const double loc, const double scale);

extern void vrk_lognormal_vec_ICDF(vrk_state *state, const int len, double *res, const double mean, const double sigma);
extern void vrk_lognormal_vec_BM(vrk_state *state, const int len, double *res, const double mean, const double sigma);

extern void vrk_wald_vec(vrk_state *state, const int len, double *res, const double mean, const double scale);

extern void vrk_vonmises_vec(vrk_state *state, const int len, double *res, const double mu, const double kappa);

extern void vrk_noncentral_f_vec(vrk_state *state, int len, double *res, double df_num, double df_den, double nonc);
extern void vrk_triangular_vec(vrk_state *state, int len, double *res, double left, double mode, double right);

extern void vrk_binomial_vec(vrk_state *state, const int len, int *res, const int n, const double p);
extern void vrk_geometric_vec(vrk_state *state, const int len, int *res, const double p);
extern void vrk_negbinomial_vec(vrk_state *state, const int len, int *res, const double a, const double p);
extern void vrk_hypergeometric_vec(vrk_state *state, const int len, int *res, const int ls, const int ss, const int ms);
extern void vrk_poisson_vec_PTPE(vrk_state *state, const int len, int *res, const double lambda);
extern void vrk_poisson_vec_POISNORM(vrk_state *state, const int len, int *res, const double lambda);

extern void vrk_poisson_vec_V(vrk_state *state, const int len, int *res, double *lambdas);

extern void vrk_zipf_long_vec(vrk_state *state, const int len, long *res, const double alp);

extern void vrk_logseries_vec(vrk_state *state, const int len, int *res, const double alp);

extern void vrk_discrete_uniform_vec(vrk_state *state, const int len, int *res, const int low, const int high);

extern void vrk_discrete_uniform_long_vec(vrk_state *state, const int len, long *res, const long low, const long high);

extern void vrk_rand_int64_vec(vrk_state *state, const int len, npy_int64 *res, const npy_int64 lo, const npy_int64 hi);
extern void vrk_rand_uint64_vec(vrk_state *state, const int len, npy_uint64 *res, const npy_uint64 lo, const npy_uint64 hi);
extern void vrk_rand_int32_vec(vrk_state *state, const int len, npy_int32 *res, const npy_int32 lo, const npy_int32 hi);
extern void vrk_rand_uint32_vec(vrk_state *state, const int len, npy_uint32 *res, const npy_uint32 lo, const npy_uint32 hi);
extern void vrk_rand_int16_vec(vrk_state *state, const int len, npy_int16 *res, const npy_int16 lo, const npy_int16 hi);
extern void vrk_rand_uint16_vec(vrk_state *state, const int len, npy_uint16 *res, const npy_uint16 lo, const npy_uint16 hi);
extern void vrk_rand_int8_vec(vrk_state *state, const int len, npy_int8 *res, const npy_int8 lo, const npy_int8 hi);
extern void vrk_rand_uint8_vec(vrk_state *state, const int len, npy_uint8 *res, const npy_uint8 lo, const npy_uint8 hi);
extern void vrk_rand_bool_vec(vrk_state *state, const int len, npy_bool *res, const npy_bool lo, const npy_bool hi);

extern void vrk_ulong_vec(vrk_state *state, const int len, unsigned long *res);
extern void vrk_long_vec(vrk_state *state, const int len, long *res);

extern void vrk_multinormal_vec_ICDF(vrk_state *state, const int len, double *res, const int dim,
    double *mean_vec, double *ch, const ch_st_enum storage_mode);

extern void vrk_multinormal_vec_BM1(vrk_state *state, const int len, double *res, const int dim,
    double *mean_vec, double *ch, const ch_st_enum storage_mode);

extern void vrk_multinormal_vec_BM2(vrk_state *state, const int len, double *res, const int dim,
    double *mean_vec, double *ch, const ch_st_enum storage_mode);

#ifdef __cplusplus
}
#endif


#endif
