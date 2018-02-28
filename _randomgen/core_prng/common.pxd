
from libc.stdint cimport uint64_t

ctypedef uint64_t (*random_uint64_anon)(void* st) nogil

ctypedef double (*random_double_anon)(void* st) nogil

ctypedef double (*random_0)(void *st) nogil

cdef struct anon_func_state:
    void *state
    void *next_uint64
    void *next_uint32
    void *next_double

ctypedef anon_func_state anon_func_state_t

cdef inline double uint64_to_double(uint64_t rnd) nogil:
    return (rnd >> 11) * (1.0 / 9007199254740992.0)

cdef object double_fill(void *func, void *state, object size, object lock, object out)
