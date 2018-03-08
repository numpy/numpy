from libc.stdint cimport int32_t, int64_t, uint32_t, uint64_t
from cpython cimport PyInt_AsLong, PyFloat_AsDouble

import numpy as np
cimport numpy as np

cdef double POISSON_LAM_MAX
cdef uint64_t MAXSIZE

cdef enum ConstraintType:
    CONS_NONE
    CONS_NON_NEGATIVE
    CONS_POSITIVE
    CONS_BOUNDED_0_1
    CONS_BOUNDED_0_1_NOTNAN
    CONS_GT_1
    CONS_GTE_1
    CONS_POISSON

ctypedef ConstraintType constraint_type

cdef extern from "src/distributions/distributions.h":

    struct s_binomial_t:
        int has_binomial;
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

    ctypedef s_binomial_t binomial_t

    struct prng:
        void *state
        uint64_t (*next_uint64)(void *st)
        uint32_t (*next_uint32)(void *st)
        double (*next_double)(void *st)
        int has_gauss
        double gauss
        int has_gauss_f
        float gauss_f
        binomial_t *binomial

    ctypedef prng prng_t

    double random_sample(prng_t *prng_state) nogil
    double random_standard_exponential(prng_t *prng_state) nogil
    double random_standard_exponential_zig(prng_t *prng_state) nogil
    double random_gauss(prng_t *prng_state) nogil
    double random_gauss_zig(prng_t* prng_state) nogil
    double random_standard_gamma(prng_t *prng_state, double shape) nogil
    double random_standard_gamma_zig(prng_t *prng_state, double shape) nogil

    float random_sample_f(prng_t *prng_state) nogil
    float random_standard_exponential_f(prng_t *prng_state) nogil
    float random_standard_exponential_zig_f(prng_t *prng_state) nogil
    float random_gauss_f(prng_t *prng_state) nogil
    float random_gauss_zig_f(prng_t* prng_state) nogil
    float random_standard_gamma_f(prng_t *prng_state, float shape) nogil
    float random_standard_gamma_zig_f(prng_t *prng_state, float shape) nogil


ctypedef double (*random_double_0)(prng_t *state) nogil
ctypedef double (*random_double_1)(prng_t *state, double a) nogil
ctypedef double (*random_double_2)(prng_t *state, double a, double b) nogil
ctypedef double (*random_double_3)(prng_t *state, double a, double b, double c) nogil

ctypedef float (*random_float_0)(prng_t *state) nogil
ctypedef float (*random_float_1)(prng_t *state, float a) nogil

ctypedef long (*random_uint_0)(prng_t *state) nogil
ctypedef long (*random_uint_d)(prng_t *state, double a) nogil
ctypedef long (*random_uint_dd)(prng_t *state, double a, double b) nogil
ctypedef long (*random_uint_di)(prng_t *state, double a, uint64_t b) nogil
ctypedef long (*random_uint_i)(prng_t *state, long a) nogil
ctypedef long (*random_uint_iii)(prng_t *state, long a, long b, long c) nogil

ctypedef uint32_t (*random_uint_0_32)(prng_t *state) nogil
ctypedef uint32_t (*random_uint_1_i_32)(prng_t *state, uint32_t a) nogil

ctypedef int32_t (*random_int_2_i_32)(prng_t *state, int32_t a, int32_t b) nogil
ctypedef int64_t (*random_int_2_i)(prng_t *state, int64_t a, int64_t b) nogil


cdef inline double uint64_to_double(uint64_t rnd) nogil:
    return (rnd >> 11) * (1.0 / 9007199254740992.0)

cdef object double_fill(void *func, prng_t *state, object size, object lock, object out)

cdef object float_fill(void *func, prng_t *state, object size, object lock, object out)

cdef object float_fill_from_double(void *func, prng_t *state, object size, object lock, object out)

cdef np.ndarray int_to_array(object value, object name, object bits)

cdef object cont(void *func, prng_t *state, object size, object lock, int narg,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint,
                 object out)

cdef object disc(void *func, prng_t *state, object size, object lock,
                 int narg_double, int narg_long,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint)

cdef object cont_f(void *func, prng_t *state, object size, object lock,
                   object a, object a_name, constraint_type a_constraint,
                   object out)
