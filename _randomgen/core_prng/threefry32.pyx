from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np

from common cimport *
from distributions cimport prng_t, binomial_t
from core_prng.entropy import random_entropy
import core_prng.pickle
cimport entropy

np.import_array()

DEF THREEFRY_BUFFER_SIZE=4

cdef extern from 'src/threefry32/threefry32.h':
    struct s_r123array4x32:
        uint32_t v[4]

    ctypedef s_r123array4x32 r123array4x32

    ctypedef r123array4x32 threefry4x32_key_t
    ctypedef r123array4x32 threefry4x32_ctr_t

    struct s_threefry32_state:
        threefry4x32_ctr_t *ctr;
        threefry4x32_key_t *key;
        int buffer_pos;
        uint32_t buffer[THREEFRY_BUFFER_SIZE];

    ctypedef s_threefry32_state threefry32_state

    uint64_t threefry32_next64(threefry32_state *state)  nogil
    uint32_t threefry32_next32(threefry32_state *state)  nogil
    double threefry32_next_double(threefry32_state *state)  nogil
    void threefry32_jump(threefry32_state *state)
    void threefry32_advance(uint32_t *step, threefry32_state *state)


cdef uint64_t threefry32_uint64(void* st) nogil:
    return threefry32_next64(<threefry32_state *>st)

cdef uint32_t threefry32_uint32(void *st) nogil:
    return threefry32_next32(<threefry32_state *> st)

cdef double threefry32_double(void* st) nogil:
    return threefry32_next_double(<threefry32_state *>st)

cdef uint64_t threefry32_raw(void *st) nogil:
    return <uint64_t>threefry32_next32(<threefry32_state *> st)


cdef class ThreeFry32:
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
    cdef threefry32_state  *rng_state
    cdef prng_t *_prng
    cdef public object capsule

    def __init__(self, seed=None, counter=None, key=None):
        self.rng_state = <threefry32_state *>malloc(sizeof(threefry32_state))
        self.rng_state.ctr = <threefry4x32_ctr_t *>malloc(sizeof(threefry4x32_ctr_t))
        self.rng_state.key = <threefry4x32_key_t *>malloc(sizeof(threefry4x32_key_t))
        self._prng = <prng_t *>malloc(sizeof(prng_t))
        self._prng.binomial = <binomial_t *>malloc(sizeof(binomial_t))
        self.seed(seed, counter, key)

        self._prng.state = <void *>self.rng_state
        self._prng.next_uint64 = &threefry32_uint64
        self._prng.next_uint32 = &threefry32_uint32
        self._prng.next_double = &threefry32_double
        self._prng.next_raw = &threefry32_raw

        cdef const char *name = 'CorePRNG'
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
        free(self.rng_state.ctr)
        free(self.rng_state.key)
        free(self.rng_state)
        free(self._prng.binomial)
        free(self._prng)

    cdef _reset_state_variables(self):
        self.rng_state.buffer_pos = THREEFRY_BUFFER_SIZE
        for i in range(THREEFRY_BUFFER_SIZE):
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
            Positive integer less than 2**128 containing the counter position
            or a 4 element array of uint32 containing the counter
        key : {int, array}, options
            Positive integer less than 2**128 containing the key
            or a 4 element array of uint32 containing the key

        Raises
        ------
        ValueError
            If values are out of range for the PRNG.

        Notes
        -----
        The two representation of the counter and key are related through
        array[i] = (value // 2**(32*i)) % 2**32.
        """
        if seed is not None and key is not None:
            raise ValueError('seed and key cannot be both used')
        if key is None:
            if seed is None:
                try:
                    state = random_entropy(4)
                except RuntimeError:
                    state = random_entropy(4, 'fallback')
            else:
                state = entropy.seed_by_array(seed, 2)
                state = state.view(np.uint32)
            for i in range(4):
                self.rng_state.key.v[i] = state[i]
        else:
            key = int_to_array(key, 'key', 128, 32)
            for i in range(4):
                self.rng_state.key.v[i] = key[i]

        counter = 0 if counter is None else counter
        counter = int_to_array(counter, 'counter', 128, 32)
        for i in range(4):
            self.rng_state.ctr.v[i] = counter[i]

        self._reset_state_variables()

    @property
    def state(self):
        """Get or set the PRNG state"""
        ctr = np.empty(4, dtype=np.uint32)
        key = np.empty(4, dtype=np.uint32)
        buffer = np.empty(THREEFRY_BUFFER_SIZE, dtype=np.uint32)
        for i in range(4):
            ctr[i] = self.rng_state.ctr.v[i]
            key[i] = self.rng_state.key.v[i]
        for i in range(THREEFRY_BUFFER_SIZE):
            buffer[i] = self.rng_state.buffer[i]
        state = {'counter':ctr,'key':key}
        return {'prng': self.__class__.__name__,
                'state': state,
                'buffer': buffer,
                'buffer_pos': self.rng_state.buffer_pos}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        prng = value.get('prng', '')
        if prng != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        for i in range(4):
            self.rng_state.ctr.v[i] = <uint32_t>value['state']['counter'][i]
            self.rng_state.key.v[i] = <uint32_t>value['state']['key'][i]
        for i in range(THREEFRY_BUFFER_SIZE):
            self.rng_state.buffer[i] = <uint32_t>value['buffer'][i]
        self.rng_state.buffer_pos = value['buffer_pos']

    def jump(self):
        """Jump the state as-if 2**64draws have been made"""
        return self.advance(2**64)

    def advance(self, step):
        """Advance the state as-if a specific number of draws have been made"""
        cdef np.ndarray step_a
        step_a = int_to_array(step, 'step', 128, 32)
        loc = 0
        threefry32_advance(<uint32_t *>step_a.data, self.rng_state)
        return self
