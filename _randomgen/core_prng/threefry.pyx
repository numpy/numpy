from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from common cimport *
from core_prng.entropy import random_entropy
cimport entropy

np.import_array()

cdef extern from "src/threefry/threefry.h":

    cdef struct s_threefry:
        uint64_t c0, c1

    ctypedef s_threefry threefry_t

    cdef struct s_threefry_state:
        threefry_t *c
        threefry_t *k
        int has_uint32
        uint32_t uinteger

    ctypedef s_threefry_state threefry_state

    cdef uint64_t threefry_next64(threefry_state *state)  nogil
    cdef uint64_t threefry_next32(threefry_state *state)  nogil


cdef uint64_t threefry_uint64(void* st):# nogil:
    return threefry_next64(<threefry_state *>st)

cdef uint32_t threefry_uint32(void *st) nogil:
    return threefry_next32(<threefry_state *> st)

cdef double threefry_double(void* st) nogil:
    return uint64_to_double(threefry_next64(<threefry_state *>st))

cdef class ThreeFry:
    """
    Prototype Core PRNG using threefry

    Parameters
    ----------
    seed : int, array of int
        Integer or array of integers between 0 and 2**64 - 1

    Notes
    -----
    Exposes no user-facing API except `state`. Designed for use in
    a `RandomGenerator` object.
    """
    cdef threefry_state  *rng_state
    cdef prng_t *_prng
    cdef public object _prng_capsule

    def __init__(self, seed=None):
        self.rng_state = <threefry_state *>malloc(sizeof(threefry_state))
        self.rng_state.c = <threefry_t *>malloc(sizeof(threefry_t))
        self.rng_state.k = <threefry_t *>malloc(sizeof(threefry_t))
        self._prng = <prng_t *>malloc(sizeof(prng_t))
        self.seed(seed)

        self._prng.state = <void *>self.rng_state
        self._prng.next_uint64 = &threefry_uint64
        self._prng.next_uint32 = &threefry_uint32
        self._prng.next_double = &threefry_double

        cdef const char *name = "CorePRNG"
        self._prng_capsule = PyCapsule_New(<void *>self._prng, name, NULL)

    def __dealloc__(self):
        free(self.rng_state.c)
        free(self.rng_state.k)
        free(self.rng_state)
        free(self._prng)

    def _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def __random_integer(self, bits=64):
        """
        64-bit Random Integers from the PRNG

        Parameters
        ----------
        bits : {32, 64}
            Number of random bits to return

        Returns
        -------
        rv : int
            Next random value

        Notes
        -----
        Testing only
        """
        if bits == 64:
            return threefry_next64(self.rng_state)
        elif bits == 32:
            return threefry_next32(self.rng_state)
        else:
            raise ValueError('bits must be 32 or 64')

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
                state = random_entropy(4)
            except RuntimeError:
                state = random_entropy(4, 'fallback')
            state = state.view(np.uint64)
        else:
            state = entropy.seed_by_array(seed, 2)
        # TODO: Need to be able to set the key and counter directly
        self.rng_state.c.c0 = 0
        self.rng_state.c.c1 = 0
        self.rng_state.k.c0 = <uint64_t>int(state[0])
        self.rng_state.k.c1 = <uint64_t>int(state[1])
        self._reset_state_variables()

    @property
    def state(self):
        """Get or set the PRNG state"""
        c = np.empty(2, dtype=np.uint64)
        k = np.empty(2, dtype=np.uint64)
        c[0] = self.rng_state.c.c0
        c[1] = self.rng_state.c.c1
        k[0] = self.rng_state.k.c0
        k[1] = self.rng_state.k.c1
        state = {'c':c,'k':k}
        return {'prng': self.__class__.__name__,
                'state': state,
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
        self.rng_state.c.c0 = <uint64_t>value['state']['c'][0]
        self.rng_state.c.c1 = <uint64_t>value['state']['c'][1]
        self.rng_state.k.c0 = <uint64_t>value['state']['k'][0]
        self.rng_state.k.c1 = <uint64_t>value['state']['k'][1]
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']
