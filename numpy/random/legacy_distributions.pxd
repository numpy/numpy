#cython: language_level=3

from libc.stdint cimport int64_t

import numpy as np
cimport numpy as np

from .distributions cimport bitgen_t, binomial_t

cdef extern from "legacy-distributions.h":

    struct aug_bitgen:
        bitgen_t *bit_generator
        int has_gauss
        double gauss

    ctypedef aug_bitgen aug_bitgen_t

    double legacy_gauss(aug_bitgen_t *aug_state) nogil
    double legacy_pareto(aug_bitgen_t *aug_state, double a) nogil
    double legacy_weibull(aug_bitgen_t *aug_state, double a) nogil
    double legacy_standard_gamma(aug_bitgen_t *aug_state, double shape) nogil
    double legacy_normal(aug_bitgen_t *aug_state, double loc, double scale) nogil
    double legacy_standard_t(aug_bitgen_t *aug_state, double df) nogil

    double legacy_standard_exponential(aug_bitgen_t *aug_state) nogil
    double legacy_power(aug_bitgen_t *aug_state, double a) nogil
    double legacy_gamma(aug_bitgen_t *aug_state, double shape, double scale) nogil
    double legacy_power(aug_bitgen_t *aug_state, double a) nogil
    double legacy_chisquare(aug_bitgen_t *aug_state, double df) nogil
    double legacy_noncentral_chisquare(aug_bitgen_t *aug_state, double df,
                                    double nonc) nogil
    double legacy_noncentral_f(aug_bitgen_t *aug_state, double dfnum, double dfden,
                            double nonc) nogil
    double legacy_wald(aug_bitgen_t *aug_state, double mean, double scale) nogil
    double legacy_lognormal(aug_bitgen_t *aug_state, double mean, double sigma) nogil
    int64_t legacy_negative_binomial(aug_bitgen_t *aug_state, double n, double p) nogil
    int64_t legacy_random_hypergeometric(bitgen_t *bitgen_state, int64_t good, int64_t bad, int64_t sample) nogil
    int64_t legacy_random_logseries(bitgen_t *bitgen_state, double p) nogil
    int64_t legacy_random_poisson(bitgen_t *bitgen_state, double lam) nogil
    int64_t legacy_random_zipf(bitgen_t *bitgen_state, double a) nogil
    int64_t legacy_random_geometric(bitgen_t *bitgen_state, double p) nogil
    void legacy_random_multinomial(bitgen_t *bitgen_state, long n, long *mnix, double *pix, np.npy_intp d, binomial_t *binomial) nogil
    double legacy_standard_cauchy(aug_bitgen_t *state) nogil
    double legacy_beta(aug_bitgen_t *aug_state, double a, double b) nogil
    double legacy_f(aug_bitgen_t *aug_state, double dfnum, double dfden) nogil
    double legacy_exponential(aug_bitgen_t *aug_state, double scale) nogil
    double legacy_power(aug_bitgen_t *state, double a) nogil
