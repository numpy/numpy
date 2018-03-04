from libc.stdint cimport uint32_t, uint64_t

cdef extern from "src/distributions/distributions.h":
    ctypedef double (*random_double_0)(void *st) nogil

    ctypedef float (*random_float_0)(void *st) nogil

    cdef struct s_binomial_t:
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

    cdef struct prng:
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

cdef inline double uint64_to_double(uint64_t rnd) nogil:
    return (rnd >> 11) * (1.0 / 9007199254740992.0)

cdef object double_fill(void *func, void *state, object size, object lock)

cdef object float_fill(void *func, void *state, object size, object lock)

cdef object float_fill_from_double(void *func, void *state, object size, object lock)