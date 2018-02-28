import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc
from cpython.pycapsule cimport PyCapsule_New
from common cimport *
from core_prng.entropy import random_entropy
cimport entropy

np.import_array()

cdef extern from "src/xoroshiro128/xoroshiro128.h":

    cdef struct s_xoroshiro128_state:
      uint64_t s[2]
      int has_uint32
      uint32_t uinteger

    ctypedef s_xoroshiro128_state xoroshiro128_state

    cdef uint64_t xoroshiro128_next(uint64_t *s)  nogil

    cdef uint64_t xoroshiro128_next64(xoroshiro128_state *state)  nogil
    cdef uint64_t xoroshiro128_next32(xoroshiro128_state *state)  nogil
    cdef void xoroshiro128_jump(xoroshiro128_state  *state)

cdef uint64_t xoroshiro128_uint64(void* st) nogil:
    return xoroshiro128_next64(<xoroshiro128_state *>st)

cdef uint64_t xoroshiro128_uint32(void *st) nogil:
    return xoroshiro128_next32(<xoroshiro128_state *> st)

cdef double xoroshiro128_double(void* st) nogil:
    return uint64_to_double(xoroshiro128_next64(<xoroshiro128_state *>st))

cdef class Xoroshiro128:
    """
    Prototype Core PRNG using xoroshiro128

    Parameters
    ----------
    seed : int, array of int
        Integer or array of integers between 0 and 2**64 - 1

    Notes
    -----
    Exposes no user-facing API except `get_state` and `set_state`. Designed
    for use in a `RandomGenerator` object.
    """
    cdef xoroshiro128_state rng_state
    cdef anon_func_state anon_func_state
    cdef public object _anon_func_state


    def __init__(self, seed=None):
        if seed is None:
            try:
                state = random_entropy(4)
            except RuntimeError:
                state = random_entropy(4, 'fallback')
            state = state.view(np.uint64)
        else:
            state = entropy.seed_by_array(seed, 2)
        self.rng_state.s[0] = <uint64_t>int(state[0])
        self.rng_state.s[1] = <uint64_t>int(state[1])

        self.anon_func_state.state = <void *>&self.rng_state
        self.anon_func_state.next_uint64 = <void *>&xoroshiro128_uint64
        self.anon_func_state.next_uint32 = <void *>&xoroshiro128_uint32
        self.anon_func_state.next_double = <void *>&xoroshiro128_double
        cdef const char *anon_name = "Anon CorePRNG func_state"
        self._anon_func_state = PyCapsule_New(<void *>&self.anon_func_state,
                                              anon_name, NULL)

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
            return xoroshiro128_next64(&self.rng_state)
        elif bits == 32:
            return xoroshiro128_next32(&self.rng_state)
        else:
            raise ValueError('bits must be 32 or 64')

    def jump(self):
        xoroshiro128_jump(&self.rng_state)

    @property
    def state(self):
        """Get or set the PRNG state"""
        state = np.empty(2, dtype=np.uint64)
        state[0] = self.rng_state.s[0]
        state[1] = self.rng_state.s[1]
        return {'prng': self.__class__.__name__,
                's': state,
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
        self.rng_state.s[0] = <uint64_t>value['s'][0]
        self.rng_state.s[1] = <uint64_t>value['s'][1]
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']
