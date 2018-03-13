from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t, intptr_t)
import numpy as np
cimport numpy as np 

cdef extern from "src/distributions/distributions.h":

    struct s_binomial_t:
        int has_binomial
        double psave
        long nsave
        double r
        double q
        double fm
        long m
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

    struct prng:
        void *state
        uint64_t (*next_uint64)(void *st) nogil
        uint32_t (*next_uint32)(void *st) nogil
        double (*next_double)(void *st) nogil
        uint64_t (*next_raw)(void *st) nogil

    ctypedef prng prng_t

    double random_sample(prng_t *prng_state) nogil
    double random_standard_exponential(prng_t *prng_state) nogil
    double random_standard_exponential_zig(prng_t *prng_state) nogil
    double random_gauss_zig(prng_t* prng_state) nogil
    double random_standard_gamma_zig(prng_t *prng_state, double shape) nogil

    float random_sample_f(prng_t *prng_state) nogil
    float random_standard_exponential_f(prng_t *prng_state) nogil
    float random_standard_exponential_zig_f(prng_t *prng_state) nogil
    float random_gauss_zig_f(prng_t* prng_state) nogil
    float random_standard_gamma_f(prng_t *prng_state, float shape) nogil
    float random_standard_gamma_zig_f(prng_t *prng_state, float shape) nogil

    int64_t random_positive_int64(prng_t *prng_state) nogil
    int32_t random_positive_int32(prng_t *prng_state) nogil
    long random_positive_int(prng_t *prng_state) nogil
    unsigned long random_uint(prng_t *prng_state) nogil

    double random_normal_zig(prng_t *prng_state, double loc, double scale) nogil

    double random_gamma(prng_t *prng_state, double shape, double scale) nogil
    float random_gamma_float(prng_t *prng_state, float shape, float scale) nogil

    double random_exponential(prng_t *prng_state, double scale) nogil
    double random_uniform(prng_t *prng_state, double lower, double range) nogil
    double random_beta(prng_t *prng_state, double a, double b) nogil
    double random_chisquare(prng_t *prng_state, double df) nogil
    double random_f(prng_t *prng_state, double dfnum, double dfden) nogil
    double random_standard_cauchy(prng_t *prng_state) nogil
    double random_pareto(prng_t *prng_state, double a) nogil
    double random_weibull(prng_t *prng_state, double a) nogil
    double random_power(prng_t *prng_state, double a) nogil
    double random_laplace(prng_t *prng_state, double loc, double scale) nogil
    double random_gumbel(prng_t *prng_state, double loc, double scale) nogil
    double random_logistic(prng_t *prng_state, double loc, double scale) nogil
    double random_lognormal(prng_t *prng_state, double mean, double sigma) nogil
    double random_rayleigh(prng_t *prng_state, double mode) nogil
    double random_standard_t(prng_t *prng_state, double df) nogil
    double random_noncentral_chisquare(prng_t *prng_state, double df,
                                            double nonc) nogil
    double random_noncentral_f(prng_t *prng_state, double dfnum,
                                    double dfden, double nonc) nogil
    double random_wald(prng_t *prng_state, double mean, double scale) nogil
    double random_vonmises(prng_t *prng_state, double mu, double kappa) nogil
    double random_triangular(prng_t *prng_state, double left, double mode,
                                    double right) nogil

    long random_poisson(prng_t *prng_state, double lam) nogil
    long random_negative_binomial(prng_t *prng_state, double n, double p) nogil
    long random_binomial(prng_t *prng_state, double p, long n, binomial_t *binomial) nogil
    long random_logseries(prng_t *prng_state, double p) nogil
    long random_geometric_search(prng_t *prng_state, double p) nogil
    long random_geometric_inversion(prng_t *prng_state, double p) nogil
    long random_geometric(prng_t *prng_state, double p) nogil
    long random_zipf(prng_t *prng_state, double a) nogil
    long random_hypergeometric(prng_t *prng_state, long good, long bad,
                                    long sample) nogil
    unsigned long random_interval(prng_t *prng_state, unsigned long max) nogil
    uint64_t random_bounded_uint64(prng_t *prng_state, uint64_t off,
                                        uint64_t rng, uint64_t mask) nogil
    uint32_t random_buffered_bounded_uint32(prng_t *prng_state, uint32_t off,
                                                uint32_t rng, uint32_t mask,
                                                int *bcnt, uint32_t *buf) nogil

    uint16_t random_buffered_bounded_uint16(prng_t *prng_state, uint16_t off,
                                                uint16_t rng, uint16_t mask,
                                                int *bcnt, uint32_t *buf) nogil
    uint8_t random_buffered_bounded_uint8(prng_t *prng_state, uint8_t off,
                                                uint8_t rng, uint8_t mask,
                                                int *bcnt, uint32_t *buf) nogil
    np.npy_bool random_buffered_bounded_bool(prng_t *prng_state, np.npy_bool off,
                                                np.npy_bool rng, np.npy_bool mask,
                                                int *bcnt, uint32_t *buf) nogil
    void random_bounded_uint64_fill(prng_t *prng_state, uint64_t off,
                                        uint64_t rng, np.npy_intp cnt,
                                        uint64_t *out) nogil
    void random_bounded_uint32_fill(prng_t *prng_state, uint32_t off,
                                        uint32_t rng, np.npy_intp cnt,
                                        uint32_t *out) nogil
    void random_bounded_uint16_fill(prng_t *prng_state, uint16_t off,
                                        uint16_t rng, np.npy_intp cnt,
                                        uint16_t *out) nogil
    void random_bounded_uint8_fill(prng_t *prng_state, uint8_t off,
                                        uint8_t rng, np.npy_intp cnt, uint8_t *out) nogil
    void random_bounded_bool_fill(prng_t *prng_state, np.npy_bool off,
                                        np.npy_bool rng, np.npy_intp cnt, np.npy_bool *out) nogil
