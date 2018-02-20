from libc.stdint cimport uint64_t

ctypedef uint64_t (*random_uint64_anon)(void* st)

cdef struct anon_func_state:
    void *state
    void *f

ctypedef anon_func_state anon_func_state_t
