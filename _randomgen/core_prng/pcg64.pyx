from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from common cimport *
from core_prng.entropy import random_entropy
import core_prng.pickle
cimport entropy

np.import_array()

IF PCG_EMULATED_MATH==1:
    cdef extern from "src/pcg64/pcg64.h":

        ctypedef struct pcg128_t:
            uint64_t high
            uint64_t low
ELSE:
    cdef extern from "inttypes.h":
        ctypedef unsigned long long __uint128_t

    cdef extern from "src/pcg64/pcg64.h":
        ctypedef __uint128_t pcg128_t

cdef extern from "src/pcg64/pcg64.h":

    ctypedef struct pcg128_t:
        uint64_t high
        uint64_t low        

cdef extern from "src/pcg64/pcg64.h":

    cdef struct pcg_state_setseq_128:
        pcg128_t state
        pcg128_t inc

    ctypedef pcg_state_setseq_128 pcg64_random_t

    struct s_pcg64_state:
      pcg64_random_t *pcg_state
      int has_uint32
      uint32_t uinteger

    ctypedef s_pcg64_state pcg64_state

    uint64_t pcg64_next64(pcg64_state *state)  nogil
    uint64_t pcg64_next32(pcg64_state *state)  nogil
    void pcg64_jump(pcg64_state  *state)
    void pcg64_advance(pcg64_state *state, uint64_t *step)


cdef uint64_t pcg64_uint64(void* st):# nogil:
    return pcg64_next64(<pcg64_state *>st)

cdef uint32_t pcg64_uint32(void *st) nogil:
    return pcg64_next32(<pcg64_state *> st)

cdef double pcg64_double(void* st) nogil:
    return uint64_to_double(pcg64_next64(<pcg64_state *>st))

cdef class PCG64:
    """
    Prototype Core PRNG using pcg64

    Parameters
    ----------
    seed : int, array of int
        Integer or array of integers between 0 and 2**64 - 1

    Notes
    -----
    Exposes no user-facing API except `get_state` and `set_state`. Designed
    for use in a `RandomGenerator` object.
    """
    cdef pcg64_state *rng_state
    cdef prng_t *_prng
    cdef public object _prng_capsule

    def __init__(self, seed=None, inc=1):
        self.rng_state = <pcg64_state *>malloc(sizeof(pcg64_state))
        self.rng_state.pcg_state = <pcg64_random_t *>malloc(sizeof(pcg64_random_t))
        self._prng = <prng_t *>malloc(sizeof(prng_t))
        self._prng.binomial = <binomial_t *>malloc(sizeof(binomial_t))
        self.seed(seed, inc)

        self._prng.state = <void *>self.rng_state
        self._prng.next_uint64 = &pcg64_uint64
        self._prng.next_uint32 = &pcg64_uint32
        self._prng.next_double = &pcg64_double

        cdef const char *name = "CorePRNG"
        self._prng_capsule = PyCapsule_New(<void *>self._prng, name, NULL)

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

    def _benchmark(self, Py_ssize_t cnt):
        cdef Py_ssize_t i
        for i in range(cnt):
            self._prng.next_uint64(self._prng.state)


    def seed(self, seed=None, inc=1):
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
        inc : int, optional
            Increment to use for PCG stream

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
        IF PCG_EMULATED_MATH==1:
            self.rng_state.pcg_state.state.high = <uint64_t>int(state[0])
            self.rng_state.pcg_state.state.low = <uint64_t>int(state[1])
            self.rng_state.pcg_state.inc.high = inc // 2**64
            self.rng_state.pcg_state.inc.low = inc % 2**64
        ELSE:
            self.rng_state.pcg_state.state = state[0] * 2**64 + state[1]
            self.rng_state.pcg_state.inc = inc
        self._reset_state_variables()

    @property
    def state(self):
        """Get or set the PRNG state"""
        IF PCG_EMULATED_MATH==1:
            state = 2 **64 * self.rng_state.pcg_state.state.high
            state += self.rng_state.pcg_state.state.low
            inc = 2 **64 * self.rng_state.pcg_state.inc.high
            inc += self.rng_state.pcg_state.inc.low
        ELSE:
            state = self.rng_state.pcg_state.state
            inc = self.rng_state.pcg_state.inc

        return {'prng': self.__class__.__name__,
                'state': {'state': state, 'inc':inc},
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
        IF PCG_EMULATED_MATH==1:
            self.rng_state.pcg_state.state.high = value['state']['state'] // 2 ** 64
            self.rng_state.pcg_state.state.low = value['state']['state'] % 2 ** 64
            self.rng_state.pcg_state.inc.high = value['state']['inc'] // 2 ** 64
            self.rng_state.pcg_state.inc.low = value['state']['inc'] % 2 ** 64
        ELSE:
            self.rng_state.pcg_state.state  = value['state']['state']
            self.rng_state.pcg_state.inc = value['state']['inc']

        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']

    def advance(self, step):
        cdef np.ndarray delta = np.empty(2,dtype=np.uint64)
        delta[0] = step // 2**64
        delta[1] = step % 2**64
        pcg64_advance(self.rng_state, <uint64_t *>delta.data)
        return self

    def jump(self):
        return self.advance(2**64)
