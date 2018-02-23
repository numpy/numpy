import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_New
from common cimport *
from core_prng.entropy import random_entropy

np.import_array()

cdef extern from "src/splitmix64/splitmix64.h":
    cdef struct s_splitmix64_state:
        uint64_t state

    ctypedef s_splitmix64_state splitmix64_state

    cdef uint64_t splitmix64_next(splitmix64_state*state)  nogil


ctypedef uint64_t (*random_uint64)(splitmix64_state*state)

cdef uint64_t _splitmix64_anon(void*st) nogil:
    return splitmix64_next(<splitmix64_state *> st)

cdef class SplitMix64:
    """
    Prototype Core PRNG using directly implemented version of SplitMix64.

    Notes
    -----
    Exposes no user-facing API except `get_state` and `set_state`. Designed
    for use in a `RandomGenerator` object.
    """
    cdef splitmix64_state rng_state
    cdef anon_func_state anon_func_state
    cdef public object _anon_func_state

    def __init__(self):
        try:
            state = random_entropy(2)
        except RuntimeError:
            state = random_entropy(2, 'fallback')
        self.rng_state.state = <uint64_t>int(state.view(np.uint64)[0])

        self.anon_func_state.state = <void *> &self.rng_state
        self.anon_func_state.f = <void *> &_splitmix64_anon
        cdef const char *anon_name = "Anon CorePRNG func_state"
        self._anon_func_state = PyCapsule_New(<void *> &self.anon_func_state,
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
        return splitmix64_next(&self.rng_state)

    def get_state(self):
        """Get PRNG state"""
        return self.rng_state.state

    def set_state(self, uint64_t value):
        """Set PRNG state"""
        self.rng_state.state = value
