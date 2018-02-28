from libc.stdint cimport uint32_t, uint64_t

cdef extern from "src/distributions/distributions.h":
    ctypedef double (*prng_double)(void *st) nogil

    ctypedef float (*prng_float)(void *st) nogil

    ctypedef uint32_t (*prng_uint32)(void *st) nogil

    ctypedef uint64_t (*prng_uint64)(void *st) nogil

    cdef struct prng:
      void *state
      void *next_uint64
      void *next_uint32
      void *next_double

    ctypedef prng prng_t

cdef inline double uint64_to_double(uint64_t rnd) nogil:
    return (rnd >> 11) * (1.0 / 9007199254740992.0)

cdef object double_fill(void *func, void *state, object size, object lock)

cdef object float_fill(void *func, void *state, object size, object lock)

cdef object float_fill_from_double(void *func, void *state, object size, object lock)