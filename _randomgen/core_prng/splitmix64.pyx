import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_New
from common cimport *

from core_prng.entropy import random_entropy
from core_prng cimport entropy

np.import_array()

cdef extern from "src/splitmix64/splitmix64.h":

    cdef uint64_t splitmix64_next(uint64_t *state)  nogil


ctypedef uint64_t (*random_uint64)(uint64_t *state)

cdef uint64_t _splitmix64_anon(void *st) nogil:
    return splitmix64_next(<uint64_t *> st)

cdef class SplitMix64:
    """
    Prototype Core PRNG using directly implemented version of SplitMix64.

    Parameters
    ----------
    seed : int, array of int
        Integer or array of integers between 0 and 2**64 - 1

    Notes
    -----
    Exposes no user-facing API except `get_state` and `set_state`. Designed
    for use in a `RandomGenerator` object.
    """
    cdef uint64_t rng_state
    cdef anon_func_state anon_func_state
    cdef public object _anon_func_state

    def __init__(self, seed=None):
        if seed is None:
            try:
                state = random_entropy(2)
            except RuntimeError:
                state = random_entropy(2, 'fallback')
            state = state.view(np.uint64)
        else:
            state = entropy.seed_by_array(seed, 1)

        self.rng_state = <uint64_t>int(state[0])

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
