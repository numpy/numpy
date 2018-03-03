from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from common cimport *
from core_prng.entropy import random_entropy
import core_prng.pickle
cimport entropy

np.import_array()

cdef extern from "src/splitmix64/splitmix64.h":
    cdef struct s_splitmix64_state:
        uint64_t state
        int has_uint32
        uint32_t uinteger

    ctypedef s_splitmix64_state splitmix64_state

    cdef uint64_t splitmix64_next64(splitmix64_state *state)  nogil
    cdef uint32_t splitmix64_next32(splitmix64_state *state)  nogil

cdef uint64_t splitmix64_uint64(void *st) nogil:
    return splitmix64_next64(<splitmix64_state *> st)

cdef uint32_t splitmix64_uint32(void *st) nogil:
    return splitmix64_next32(<splitmix64_state *> st)

cdef double splitmix64_double(void *st) nogil:
    return uint64_to_double(splitmix64_uint64(<splitmix64_state *> st))

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
    cdef splitmix64_state *rng_state
    cdef prng_t *_prng
    cdef public object _prng_capsule

    def __init__(self, seed=None):
        self.rng_state = <splitmix64_state *>malloc(sizeof(splitmix64_state))
        self._prng = <prng_t *>malloc(sizeof(prng_t))
        self.seed(seed)

        self._prng.state = <void *>self.rng_state
        self._prng.next_uint64 = &splitmix64_uint64
        self._prng.next_uint32 = &splitmix64_uint32
        self._prng.next_double = &splitmix64_double

        cdef const char *name = "CorePRNG"
        self._prng_capsule = PyCapsule_New(<void *>self._prng, name, NULL)

    def __dealloc__(self):
        free(self.rng_state)
        free(self._prng)

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        return (core_prng.pickle.__prng_ctor,
                (self.state['prng'],),
                self.state)


    def _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

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
        return splitmix64_next64(self.rng_state)

    def seed(self, seed=None):
        """
        seed(seed=None, stream=None)

        Seed the generator.

        This method is called when ``RandomState`` is initialized. It can be
        called again to re-seed the generator. For details, see
        ``RandomState``.

        Parameters
        ----------
        seed : int, optional
            Seed for ``RandomState``.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.

        """
        ub =  2 ** 64
        if seed is None:
            try:
                state = random_entropy(2)
            except RuntimeError:
                state = random_entropy(2, 'fallback')
            state = state.view(np.uint64)
        else:
            state = entropy.seed_by_array(seed, 1)
        self.rng_state.state = <uint64_t> int(state[0])
        self._reset_state_variables()


    @property
    def state(self):
        """Get or set the PRNG state"""
        return {'prng': self.__class__.__name__,
                's': self.rng_state.state,
                'has_uint32': self.rng_state.has_uint32,
                'uinteger': self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        prng = value.get('prng', '')
        if prng != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        self.rng_state.state = value['s']
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']
