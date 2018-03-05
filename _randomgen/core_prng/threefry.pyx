from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np

from common cimport *
from core_prng.entropy import random_entropy
import core_prng.pickle
cimport entropy

np.import_array()

cdef extern from 'src/threefry/threefry.h':
    struct s_r123array4x64:
        uint64_t v[4]

    ctypedef s_r123array4x64 r123array4x64

    ctypedef r123array4x64 threefry4x64_key_t
    ctypedef r123array4x64 threefry4x64_ctr_t

    struct s_threefry_state:
        threefry4x64_ctr_t *ctr;
        threefry4x64_key_t *key;
        int buffer_pos;
        uint64_t buffer[4];
        int has_uint32
        uint32_t uinteger

    ctypedef s_threefry_state threefry_state

    uint64_t threefry_next64(threefry_state *state)  nogil
    uint64_t threefry_next32(threefry_state *state)  nogil
    void threefry_jump(threefry_state *state)
    void threefry_advance(uint64_t *step, threefry_state *state)


cdef uint64_t threefry_uint64(void* st) nogil:
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

    def __init__(self, seed=None, counter=None, key=None):
        self.rng_state = <threefry_state *>malloc(sizeof(threefry_state))
        self.rng_state.ctr = <threefry4x64_ctr_t *>malloc(sizeof(threefry4x64_ctr_t))
        self.rng_state.key = <threefry4x64_key_t *>malloc(sizeof(threefry4x64_key_t))
        self._prng = <prng_t *>malloc(sizeof(prng_t))
        self._prng.binomial = <binomial_t *>malloc(sizeof(binomial_t))
        self.seed(seed, counter, key)

        self._prng.state = <void *>self.rng_state
        self._prng.next_uint64 = &threefry_uint64
        self._prng.next_uint32 = &threefry_uint32
        self._prng.next_double = &threefry_double

        cdef const char *name = 'CorePRNG'
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
        free(self.rng_state.ctr)
        free(self.rng_state.key)
        free(self.rng_state)
        free(self._prng.binomial)
        free(self._prng)

    def _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0
        self.rng_state.buffer_pos = 4
        for i in range(4):
            self.rng_state.buffer[i] = 0

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

    def seed(self, seed=None, counter=None, key=None):
        """
        seed(seed=None, counter=None, key=None)

        Seed the generator.

        This method is called when ``RandomState`` is initialized. It can be
        called again to re-seed the generator. For details, see
        ``RandomState``.

        Parameters
        ----------
        seed : int, optional
            Seed for ``RandomState``.
        counter : {int array}, optional
            Positive integer less than 2**256 containing the counter position
            or a 4 element array of uint64 containing the counter
        key : {int, array}, options
            Positive integer less than 2**256 containing the key
            or a 4 element array of uint64 containing the key

        Raises
        ------
        ValueError
            If values are out of range for the PRNG.

        Notes
        -----
        The two representation of the counter and key are related through
        array[i] = (value // 2**(64*i)) % 2**64.
        """
        if seed is not None and key is not None:
            raise ValueError('seed and key cannot be both used')
        ub =  2 ** 64
        if key is None:
            if seed is None:
                try:
                    state = random_entropy(8)
                except RuntimeError:
                    state = random_entropy(8, 'fallback')
                state = state.view(np.uint64)
            else:
                state = entropy.seed_by_array(seed, 4)
            for i in range(4):
                self.rng_state.key.v[i] = state[i]
        else:
            key = int_to_array(key, 'key', 256)
            for i in range(4):
                self.rng_state.key.v[i] = key[i]

        counter = 0 if counter is None else counter
        counter = int_to_array(counter, 'counter', 256)
        for i in range(4):
            self.rng_state.ctr.v[i] = counter[i]

        self._reset_state_variables()

    @property
    def state(self):
        """Get or set the PRNG state"""
        ctr = np.empty(4, dtype=np.uint64)
        key = np.empty(4, dtype=np.uint64)
        buffer = np.empty(4, dtype=np.uint64)
        for i in range(4):
            ctr[i] = self.rng_state.ctr.v[i]
            key[i] = self.rng_state.key.v[i]
            buffer[i] = self.rng_state.buffer[i]
        state = {'counter':ctr,'key':key}
        return {'prng': self.__class__.__name__,
                'state': state,
                'buffer': buffer,
                'buffer_pos': self.rng_state.buffer_pos,
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
        for i in range(4):
            self.rng_state.ctr.v[i] = <uint64_t>value['state']['counter'][i]
            self.rng_state.key.v[i] = <uint64_t>value['state']['key'][i]
            self.rng_state.buffer[i] = <uint64_t>value['buffer'][i]
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']
        self.rng_state.buffer_pos = value['buffer_pos']

    def jump(self):
        """Jump the state as-if 2**128 draws have been made"""
        return self.advance(2**128)

    def advance(self, step):
        """Advance the state as-if a specific number of draws have been made"""
        cdef np.ndarray step_a
        step_a = int_to_array(step, 'step', 256)
        loc = 0
        threefry_advance(<uint64_t *>step_a.data, self.rng_state)
        return self
