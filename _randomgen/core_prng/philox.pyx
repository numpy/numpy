from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from common cimport *
from core_prng.entropy import random_entropy
import core_prng.pickle
cimport entropy

np.import_array()

cdef extern from "src/philox/philox.h":
    struct s_r123array2x64:
        uint64_t v[2]

    struct s_r123array4x64:
        uint64_t v[4]

    ctypedef s_r123array4x64 r123array4x64
    ctypedef s_r123array2x64 r123array2x64

    ctypedef r123array4x64 philox4x64_ctr_t;
    ctypedef r123array2x64 philox4x64_key_t;

    struct s_philox_state:
        philox4x64_ctr_t *ctr;
        philox4x64_key_t *key;
        int buffer_pos;
        uint64_t buffer[4];
        int has_uint32
        uint32_t uinteger

    ctypedef s_philox_state philox_state

    uint64_t philox_next64(philox_state *state)  nogil
    uint64_t philox_next32(philox_state *state)  nogil
    void philox_jump(philox_state *state)
    void philox_advance(uint64_t *step, philox_state *state)


cdef uint64_t philox_uint64(void*st) nogil:
    return philox_next64(<philox_state *> st)

cdef uint32_t philox_uint32(void *st) nogil:
    return philox_next32(<philox_state *> st)

cdef double philox_double(void*st) nogil:
    return uint64_to_double(philox_next64(<philox_state *> st))

cdef class Philox:
    """
    Prototype Core PRNG using philox

    Parameters
    ----------
    seed : int, array of int
        Integer or array of integers between 0 and 2**64 - 1

    Notes
    -----
    Exposes no user-facing API except `state`. Designed for use in
    a `RandomGenerator` object.
    """
    cdef philox_state  *rng_state
    cdef prng_t *_prng
    cdef public object _prng_capsule

    def __init__(self, seed=None):
        self.rng_state = <philox_state *> malloc(sizeof(philox_state))
        self.rng_state.ctr = <philox4x64_ctr_t *> malloc(
            sizeof(philox4x64_ctr_t))
        self.rng_state.key = <philox4x64_key_t *> malloc(
            sizeof(philox4x64_key_t))
        self._prng = <prng_t *> malloc(sizeof(prng_t))
        self._prng.binomial = <binomial_t *> malloc(sizeof(binomial_t))
        self.seed(seed)

        self._prng.state = <void *> self.rng_state
        self._prng.next_uint64 = &philox_uint64
        self._prng.next_uint32 = &philox_uint32
        self._prng.next_double = &philox_double

        cdef const char *name = "CorePRNG"
        self._prng_capsule = PyCapsule_New(<void *> self._prng, name, NULL)

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
        ub = 2 ** 64
        if seed is None:
            try:
                state = random_entropy(4)
            except RuntimeError:
                state = random_entropy(4, 'fallback')
            state = state.view(np.uint64)
        else:
            state = entropy.seed_by_array(seed, 2)
        # TODO: Need to be able to set the key and counter directly
        for i in range(4):
            self.rng_state.ctr.v[i] = 0
        self.rng_state.key.v[0] = state[0]
        self.rng_state.key.v[1] = state[1]
        self._reset_state_variables()

    @property
    def state(self):
        """Get or set the PRNG state"""
        ctr = np.empty(4, dtype=np.uint64)
        key = np.empty(2, dtype=np.uint64)
        buffer = np.empty(4, dtype=np.uint64)
        for i in range(4):
            ctr[i] = self.rng_state.ctr.v[i]
            buffer[i] = self.rng_state.buffer[i]
            if i < 2:
                key[i] = self.rng_state.key.v[i]

        state = {'ctr': ctr, 'key': key}
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
            self.rng_state.ctr.v[i] = <uint64_t> value['state']['ctr'][i]
            self.rng_state.buffer[i] = <uint64_t> value['buffer'][i]
            if i < 2:
                self.rng_state.key.v[i] = <uint64_t> value['state']['key'][i]
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']
        self.rng_state.buffer_pos = value['buffer_pos']

    def jump(self):
        """Jump the state as-if 2**128 draws have been made"""
        philox_jump(self.rng_state)
        return self

    def advance(self, step):
        """Advance the state as-if a specific number of draws have been made"""
        cdef np.ndarray step_a = np.zeros(4, dtype=np.uint64)
        if step >= 2 ** 256 or step < 0:
            raise ValueError('step must be between 0 and 2**256-1')
        loc = 0
        while step > 0:
            step_a[loc] = step % 2 ** 64
            step >>= 64
            loc += 1
        philox_advance(<uint64_t *> step_a.data, self.rng_state)
