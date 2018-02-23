import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t
from cpython.pycapsule cimport PyCapsule_New
from common cimport *
from core_prng.entropy import random_entropy
np.import_array()

cdef extern from "src/xoroshiro128/xoroshiro128.h":

    cdef struct s_xoroshiro128_state:
        uint64_t s[2]

    ctypedef s_xoroshiro128_state xoroshiro128_state

    cdef uint64_t xoroshiro128_next(xoroshiro128_state* state)  nogil

    cdef void xoroshiro128_jump(xoroshiro128_state* state)


ctypedef uint64_t (*random_uint64)(xoroshiro128_state* state)

cdef uint64_t _xoroshiro128_anon(void* st) nogil:
    return xoroshiro128_next(<xoroshiro128_state *>st)

cdef class Xoroshiro128:
    """
    Prototype Core PRNG using xoroshiro128

    Notes
    -----
    Exposes no user-facing API except `get_state` and `set_state`. Designed
    for use in a `RandomGenerator` object.
    """
    cdef xoroshiro128_state rng_state
    cdef anon_func_state anon_func_state
    cdef public object _anon_func_state

    def __init__(self):
        try:
            state = random_entropy(4)
        except RuntimeError:
            state = random_entropy(4, 'fallback')
        state = state.view(np.uint64)
        self.rng_state.s[0] = <uint64_t>int(state[0])
        self.rng_state.s[1] = <uint64_t>int(state[1])

        self.anon_func_state.state = <void *>&self.rng_state
        self.anon_func_state.f = <void *>&_xoroshiro128_anon
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
        return xoroshiro128_next(&self.rng_state)

    def get_state(self):
        """Get PRNG state"""
        return np.array(self.rng_state.s,dtype=np.uint64)

    def set_state(self, value):
        """Set PRNG state"""
        value = np.asarray(value, dtype=np.uint64)
        self.rng_state.s[0] = <uint64_t>value[0]
        self.rng_state.s[1] = <uint64_t>value[1]

    def jump(self):
        xoroshiro128_jump(&self.rng_state)
