from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t, intptr_t)
import numpy as np
cimport numpy as np 

cdef extern from "src/distributions/distributions.h":

    struct s_binomial_t:
        int has_binomial
        double psave
        int64_t nsave
        double r
        double q
        double fm
        int64_t m
        double p1
        double xm
        double xl
        double xr
        double c
        double laml
        double lamr
        double p2
        double p3
        double p4

    ctypedef s_binomial_t binomial_t

    struct brng:
        void *state
        uint64_t (*next_uint64)(void *st) nogil
        uint32_t (*next_uint32)(void *st) nogil
        double (*next_double)(void *st) nogil
        uint64_t (*next_raw)(void *st) nogil

    ctypedef brng brng_t

    double random_double(brng_t *brng_state) nogil
    void random_double_fill(brng_t* brng_state, np.npy_intp cnt, double *out)
    double random_standard_exponential(brng_t *brng_state) nogil
    void random_standard_exponential_fill(brng_t *brng_state, np.npy_intp cnt, double *out)
    double random_standard_exponential_zig(brng_t *brng_state) nogil
    void random_standard_exponential_zig_fill(brng_t *brng_state, np.npy_intp cnt, double *out)
    double random_gauss_zig(brng_t* brng_state) nogil
    void random_gauss_zig_fill(brng_t *brng_state, np.npy_intp count, double *out) nogil
    double random_standard_gamma_zig(brng_t *brng_state, double shape) nogil

    float random_float(brng_t *brng_state) nogil
    float random_standard_exponential_f(brng_t *brng_state) nogil
    float random_standard_exponential_zig_f(brng_t *brng_state) nogil
    float random_gauss_zig_f(brng_t* brng_state) nogil
    float random_standard_gamma_f(brng_t *brng_state, float shape) nogil
    float random_standard_gamma_zig_f(brng_t *brng_state, float shape) nogil

    int64_t random_positive_int64(brng_t *brng_state) nogil
    int32_t random_positive_int32(brng_t *brng_state) nogil
    int64_t random_positive_int(brng_t *brng_state) nogil
    uint64_t random_uint(brng_t *brng_state) nogil

    double random_normal_zig(brng_t *brng_state, double loc, double scale) nogil

    double random_gamma(brng_t *brng_state, double shape, double scale) nogil
    float random_gamma_float(brng_t *brng_state, float shape, float scale) nogil

    double random_exponential(brng_t *brng_state, double scale) nogil
    double random_uniform(brng_t *brng_state, double lower, double range) nogil
    double random_beta(brng_t *brng_state, double a, double b) nogil
    double random_chisquare(brng_t *brng_state, double df) nogil
    double random_f(brng_t *brng_state, double dfnum, double dfden) nogil
    double random_standard_cauchy(brng_t *brng_state) nogil
    double random_pareto(brng_t *brng_state, double a) nogil
    double random_weibull(brng_t *brng_state, double a) nogil
    double random_power(brng_t *brng_state, double a) nogil
    double random_laplace(brng_t *brng_state, double loc, double scale) nogil
    double random_gumbel(brng_t *brng_state, double loc, double scale) nogil
    double random_logistic(brng_t *brng_state, double loc, double scale) nogil
    double random_lognormal(brng_t *brng_state, double mean, double sigma) nogil
    double random_rayleigh(brng_t *brng_state, double mode) nogil
    double random_standard_t(brng_t *brng_state, double df) nogil
    double random_noncentral_chisquare(brng_t *brng_state, double df,
                                            double nonc) nogil
    double random_noncentral_f(brng_t *brng_state, double dfnum,
                                    double dfden, double nonc) nogil
    double random_wald(brng_t *brng_state, double mean, double scale) nogil
    double random_vonmises(brng_t *brng_state, double mu, double kappa) nogil
    double random_triangular(brng_t *brng_state, double left, double mode,
                                    double right) nogil

    int64_t random_poisson(brng_t *brng_state, double lam) nogil
    int64_t random_negative_binomial(brng_t *brng_state, double n, double p) nogil
    int64_t random_binomial(brng_t *brng_state, double p, int64_t n, binomial_t *binomial) nogil
    int64_t random_logseries(brng_t *brng_state, double p) nogil
    int64_t random_geometric_search(brng_t *brng_state, double p) nogil
    int64_t random_geometric_inversion(brng_t *brng_state, double p) nogil
    int64_t random_geometric(brng_t *brng_state, double p) nogil
    int64_t random_zipf(brng_t *brng_state, double a) nogil
    int64_t random_hypergeometric(brng_t *brng_state, int64_t good, int64_t bad,
                                    int64_t sample) nogil
    uint64_t random_interval(brng_t *brng_state, uint64_t max) nogil
    uint64_t random_bounded_uint64(brng_t *brng_state, uint64_t off,
                                        uint64_t rng, uint64_t mask) nogil
    uint32_t random_buffered_bounded_uint32(brng_t *brng_state, uint32_t off,
                                                uint32_t rng, uint32_t mask,
                                                int *bcnt, uint32_t *buf) nogil

    uint16_t random_buffered_bounded_uint16(brng_t *brng_state, uint16_t off,
                                                uint16_t rng, uint16_t mask,
                                                int *bcnt, uint32_t *buf) nogil
    uint8_t random_buffered_bounded_uint8(brng_t *brng_state, uint8_t off,
                                                uint8_t rng, uint8_t mask,
                                                int *bcnt, uint32_t *buf) nogil
    np.npy_bool random_buffered_bounded_bool(brng_t *brng_state, np.npy_bool off,
                                                np.npy_bool rng, np.npy_bool mask,
                                                int *bcnt, uint32_t *buf) nogil
    void random_bounded_uint64_fill(brng_t *brng_state, uint64_t off,
                                        uint64_t rng, np.npy_intp cnt,
                                        uint64_t *out) nogil
    void random_bounded_uint32_fill(brng_t *brng_state, uint32_t off,
                                        uint32_t rng, np.npy_intp cnt,
                                        uint32_t *out) nogil
    void random_bounded_uint16_fill(brng_t *brng_state, uint16_t off,
                                        uint16_t rng, np.npy_intp cnt,
                                        uint16_t *out) nogil
    void random_bounded_uint8_fill(brng_t *brng_state, uint8_t off,
                                        uint8_t rng, np.npy_intp cnt, uint8_t *out) nogil
    void random_bounded_bool_fill(brng_t *brng_state, np.npy_bool off,
                                        np.npy_bool rng, np.npy_intp cnt, np.npy_bool *out) nogil
