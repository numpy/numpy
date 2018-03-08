from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from common cimport *
from core_prng.entropy import random_entropy
import core_prng.pickle
cimport entropy

np.import_array()

cdef extern from "src/xorshift1024/xorshift1024.h":

    struct s_xorshift1024_state:
      uint64_t s[16]
      int p
      int has_uint32
      uint32_t uinteger

    ctypedef s_xorshift1024_state xorshift1024_state

    uint64_t xorshift1024_next64(xorshift1024_state *state)  nogil
    uint64_t xorshift1024_next32(xorshift1024_state *state)  nogil
    void xorshift1024_jump(xorshift1024_state  *state)

cdef uint64_t xorshift1024_uint64(void* st) nogil:
    return xorshift1024_next64(<xorshift1024_state *>st)

cdef uint32_t xorshift1024_uint32(void *st) nogil:
    return xorshift1024_next32(<xorshift1024_state *> st)

cdef double xorshift1024_double(void* st) nogil:
    return uint64_to_double(xorshift1024_next64(<xorshift1024_state *>st))

cdef class Xorshift1024:
    """
    Prototype Core PRNG using xorshift1024

    Parameters
    ----------
    seed : int, array of int
        Integer or array of integers between 0 and 2**64 - 1

    Notes
    -----
    Exposes no user-facing API except `get_state` and `set_state`. Designed
    for use in a `RandomGenerator` object.
    """
    cdef xorshift1024_state  *rng_state
    cdef prng_t *_prng
    cdef public object capsule

    def __init__(self, seed=None):
        self.rng_state = <xorshift1024_state *>malloc(sizeof(xorshift1024_state))
        self._prng = <prng_t *>malloc(sizeof(prng_t))
        self._prng.binomial = <binomial_t *>malloc(sizeof(binomial_t))
        self.seed(seed)

        self._prng.state = <void *>self.rng_state
        self._prng.next_uint64 = &xorshift1024_uint64
        self._prng.next_uint32 = &xorshift1024_uint32
        self._prng.next_double = &xorshift1024_double

        cdef const char *name = "CorePRNG"
        self.capsule = PyCapsule_New(<void *>self._prng, name, NULL)

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        return (core_prng.pickle.__prng_ctor,
                (self.state['prng'],),
                self.state)

    def __dealloc__(self):
        free(self.rng_state)
        free(self._prng.binomial)
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
            return self._prng.next_uint64(self._prng.state)
        elif bits == 32:
            return self._prng.next_uint32(self._prng.state)
        else:
            raise ValueError('bits must be 32 or 64')

    def _benchmark(self, Py_ssize_t cnt, method=u'uint64'):
        cdef Py_ssize_t i
        if method==u'uint64':
            for i in range(cnt):
                self._prng.next_uint64(self._prng.state)
        elif method==u'double':
            for i in range(cnt):
                self._prng.next_double(self._prng.state)
        else:
            raise ValueError('Unknown method')

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
                state = random_entropy(32)
            except RuntimeError:
                state = random_entropy(4, 'fallback')
            state = state.view(np.uint64)
        else:
            state = entropy.seed_by_array(seed, 16)
        for i in range(16):
            self.rng_state.s[i] = <uint64_t>int(state[i])
        self.rng_state.p = 0
        self._reset_state_variables()

    def jump(self):
        xorshift1024_jump(self.rng_state)
        return self

    @property
    def state(self):
        """Get or set the PRNG state"""
        s = np.empty(16, dtype=np.uint64)
        for i in range(16):
            s[i] = self.rng_state.s[i]
        return {'prng': self.__class__.__name__,
                'state': {'s':s,'p':self.rng_state.p},
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
        for i in range(16):
            self.rng_state.s[i] = <uint64_t>value['state']['s'][i]
        self.rng_state.p = value['state']['p']
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']
