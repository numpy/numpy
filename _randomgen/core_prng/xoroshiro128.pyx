from libc.stdint cimport uint32_t, uint64_t
from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

from collections import namedtuple
interface = namedtuple('interface', ['state_address','state','next_uint64','next_uint32','next_double','prng'])

import numpy as np
cimport numpy as np

from common cimport *
from core_prng.entropy import random_entropy
import core_prng.pickle
cimport entropy

np.import_array()

cdef extern from "src/xoroshiro128/xoroshiro128.h":

    struct s_xoroshiro128_state:
      uint64_t s[2]
      int has_uint32
      uint32_t uinteger

    ctypedef s_xoroshiro128_state xoroshiro128_state

    uint64_t xoroshiro128_next64(xoroshiro128_state *state)  nogil
    uint64_t xoroshiro128_next32(xoroshiro128_state *state)  nogil
    void xoroshiro128_jump(xoroshiro128_state  *state)

cdef uint64_t xoroshiro128_uint64(void* st) nogil:
    return xoroshiro128_next64(<xoroshiro128_state *>st)

cdef uint32_t xoroshiro128_uint32(void *st) nogil:
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
    cdef xoroshiro128_state  *rng_state
    cdef prng_t *_prng
    cdef public object capsule
    cdef object ctypes
    cdef object cffi

    def __init__(self, seed=None):
        self.rng_state = <xoroshiro128_state *>malloc(sizeof(xoroshiro128_state))
        self._prng = <prng_t *>malloc(sizeof(prng_t))
        self._prng.binomial = <binomial_t *>malloc(sizeof(binomial_t))
        self.seed(seed)

        self._prng.state = <void *>self.rng_state
        self._prng.next_uint64 = &xoroshiro128_uint64
        self._prng.next_uint32 = &xoroshiro128_uint32
        self._prng.next_double = &xoroshiro128_double

        self.ctypes = None
        self.cffi = None
        
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
        # TODO: These should be done everywhere for safety
        self._prng.has_gauss = 0
        self._prng.has_gauss_f = 0

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
            return xoroshiro128_next64(self.rng_state)
        elif bits == 32:
            return xoroshiro128_next32(self.rng_state)
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
                state = random_entropy(4)
            except RuntimeError:
                state = random_entropy(4, 'fallback')
            state = state.view(np.uint64)
        else:
            state = entropy.seed_by_array(seed, 2)
        self.rng_state.s[0] = <uint64_t>int(state[0])
        self.rng_state.s[1] = <uint64_t>int(state[1])
        self._reset_state_variables()

    def jump(self):
        xoroshiro128_jump(self.rng_state)
        return self

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

    @property
    def ctypes(self):
        if self.ctypes is not None:
            return self.ctypes

        import ctypes
        
        self.ctypes = interface(<Py_ssize_t>self.rng_state,
                         ctypes.c_void_p(<Py_ssize_t>self.rng_state),
                         ctypes.cast(<Py_ssize_t>&xoroshiro128_uint64, 
                                     ctypes.CFUNCTYPE(ctypes.c_uint64, 
                                     ctypes.c_void_p)),
                         ctypes.cast(<Py_ssize_t>&xoroshiro128_uint32, 
                                     ctypes.CFUNCTYPE(ctypes.c_uint32, 
                                     ctypes.c_void_p)),
                         ctypes.cast(<Py_ssize_t>&xoroshiro128_double, 
                                     ctypes.CFUNCTYPE(ctypes.c_double, 
                                     ctypes.c_void_p)),
                         ctypes.c_void_p(<Py_ssize_t>self._prng))
        return self.ctypes

    @property
    def cffi(self):
        if self.cffi is not None:
            return self.cffi
        import cffi 

        ffi = cffi.FFI()
        self.cffi = interface(<Py_ssize_t>self.rng_state,
                         ffi.cast('void *',<Py_ssize_t>self.rng_state),
                         ffi.cast('uint64_t (*)(void *)',<uint64_t>self._prng.next_uint64),
                         ffi.cast('uint32_t (*)(void *)',<uint64_t>self._prng.next_uint32),
                         ffi.cast('double (*)(void *)',<uint64_t>self._prng.next_double),
                         ffi.cast('void *',<Py_ssize_t>self._prng))
        return self.cffi
