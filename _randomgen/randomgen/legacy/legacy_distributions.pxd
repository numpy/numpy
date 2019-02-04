#cython: language_level=3

from libc.stdint cimport uint64_t

import numpy as np
cimport numpy as np 

from randomgen.distributions cimport brng_t

cdef extern from "../src/legacy/distributions-boxmuller.h":

    struct aug_brng:
        brng_t *basicrng
        int has_gauss
        double gauss 

    ctypedef aug_brng aug_brng_t

    double legacy_gauss(aug_brng_t *aug_state) nogil
    double legacy_pareto(aug_brng_t *aug_state, double a) nogil
    double legacy_weibull(aug_brng_t *aug_state, double a) nogil
    double legacy_standard_gamma(aug_brng_t *aug_state, double shape) nogil
    double legacy_normal(aug_brng_t *aug_state, double loc, double scale) nogil
    double legacy_standard_t(aug_brng_t *aug_state, double df) nogil

    double legacy_standard_exponential(aug_brng_t *aug_state) nogil
    double legacy_power(aug_brng_t *aug_state, double a) nogil
    double legacy_gamma(aug_brng_t *aug_state, double shape, double scale) nogil
    double legacy_power(aug_brng_t *aug_state, double a) nogil
    double legacy_chisquare(aug_brng_t *aug_state, double df) nogil
    double legacy_noncentral_chisquare(aug_brng_t *aug_state, double df,
                                    double nonc) nogil
    double legacy_noncentral_f(aug_brng_t *aug_state, double dfnum, double dfden,
                            double nonc) nogil
    double legacy_wald(aug_brng_t *aug_state, double mean, double scale) nogil
    double legacy_lognormal(aug_brng_t *aug_state, double mean, double sigma) nogil
    uint64_t legacy_negative_binomial(aug_brng_t *aug_state, double n, double p) nogil
    double legacy_standard_cauchy(aug_brng_t *state) nogil
    double legacy_beta(aug_brng_t *aug_state, double a, double b) nogil
    double legacy_f(aug_brng_t *aug_state, double dfnum, double dfden) nogil
    double legacy_exponential(aug_brng_t *aug_state, double scale) nogil
    double legacy_power(aug_brng_t *state, double a) nogil
