import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t
from cpython.pycapsule cimport PyCapsule_New
from common cimport *

np.import_array()

cdef struct state:
    uint64_t state

ctypedef state state_t

ctypedef uint64_t (*random_uint64)(state_t *st)

cdef struct func_state:
    state st
    random_uint64 f

ctypedef func_state func_state_t


cdef uint64_t _splitmix64(state_t *st):
    cdef uint64_t z
    # TODO: Use literals -- PyCharm complains
    cdef uint64_t c1 = 11400714819323198485
    cdef uint64_t c2 = 13787848793156543929
    cdef uint64_t c3 = 10723151780598845931
    st[0].state += c1 # 0x9E3779B97F4A7C15
    z = <uint64_t>st[0].state
    z = (z ^ (z >> 30)) * c2 # 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * c3 # 0x94D049BB133111EB
    return z ^ (z >> 31)

cdef uint64_t _splitmix64_anon(void* st):
    return _splitmix64(<state *> st)

cdef class CorePRNG:
    """
    Prototype Core PRNG using directly implemented version of SplitMix64.

    Notes
    -----
    Exposes no user-facing API except `get_state` and `set_state`. Designed
    for use in a `RandomGenerator` object.
    """
    cdef state rng_state
    cdef anon_func_state anon_func_state
    cdef public object _anon_func_state

    def __init__(self):
        self.rng_state.state = 17013192669731687406

        self.anon_func_state.state = <void *>&self.rng_state
        self.anon_func_state.f = <void *>&_splitmix64_anon
        cdef const char *anon_name = "Anon CorePRNG func_state"
        self._anon_func_state = PyCapsule_New(<void *>&self.anon_func_state,
                                              anon_name, NULL)

    def __random_integer(self):
        """
        64-bit Random Integers from the PRNG

        Returns
        -------
        rv : int
            Next random value

        Notes
        -----
        Testing only
        """
        return _splitmix64(&self.rng_state)

    def get_state(self):
        """Get PRNG state"""
        return self.rng_state.state

    def set_state(self, uint64_t value):
        """Set PRNG state"""
        self.rng_state.state = value
