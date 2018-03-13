import operator

from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from common cimport *
from distributions cimport prng_t, binomial_t
import core_prng.pickle
from core_prng.entropy import random_entropy

np.import_array()

cdef extern from "src/mt19937/mt19937.h":

    struct s_mt19937_state:
        uint32_t key[624]
        int pos

    ctypedef s_mt19937_state mt19937_state

    uint64_t mt19937_next64(mt19937_state *state)  nogil
    uint32_t mt19937_next32(mt19937_state *state)  nogil
    double mt19937_next_double(mt19937_state *state)  nogil
    void mt19937_init_by_array(mt19937_state *state, uint32_t *init_key, int key_length)
    void mt19937_seed(mt19937_state *state, uint32_t seed)
    void mt19937_jump(mt19937_state *state)

cdef uint64_t mt19937_uint64(void *st) nogil:
    return mt19937_next64(<mt19937_state *> st)

cdef uint32_t mt19937_uint32(void *st) nogil:
    return mt19937_next32(<mt19937_state *> st)

cdef double mt19937_double(void *st) nogil:
    return mt19937_next_double(<mt19937_state *> st)

cdef uint64_t mt19937_raw(void *st) nogil:
    return <uint64_t>mt19937_next32(<mt19937_state *> st)

cdef class MT19937:
    """
    Prototype Core PRNG using MT19937

    Parameters
    ----------
    seed : int, array of int
        Integer or array of integers between 0 and 2**64 - 1

    Notes
    -----
    Exposes no user-facing API except `state`. Designed for use in a
    `RandomGenerator` object.
    """
    cdef mt19937_state *rng_state
    cdef prng_t *_prng
    cdef public object capsule

    def __init__(self, seed=None):
        self.rng_state = <mt19937_state *>malloc(sizeof(mt19937_state))
        self._prng = <prng_t *>malloc(sizeof(prng_t))
        self._prng.binomial = <binomial_t *>malloc(sizeof(binomial_t))
        self.seed(seed)

        self._prng.state = <void *>self.rng_state
        self._prng.next_uint64 = &mt19937_uint64
        self._prng.next_uint32 = &mt19937_uint32
        self._prng.next_double = &mt19937_double
        self._prng.next_raw = &mt19937_raw

        cdef const char *name = "CorePRNG"
        self.capsule = PyCapsule_New(<void *>self._prng, name, NULL)

    def __dealloc__(self):
        free(self.rng_state)
        free(self._prng.binomial)
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
        cdef np.ndarray obj
        try:
            if seed is None:
                try:
                    seed = random_entropy(1)
                except RuntimeError:
                    seed = random_entropy(1, 'fallback')
                mt19937_seed(self.rng_state, seed[0])
            else:
                if hasattr(seed, 'squeeze'):
                    seed = seed.squeeze()
                idx = operator.index(seed)
                if idx > int(2**32 - 1) or idx < 0:
                    raise ValueError("Seed must be between 0 and 2**32 - 1")
                mt19937_seed(self.rng_state, seed)
        except TypeError:
            obj = np.asarray(seed).astype(np.int64, casting='safe')
            if ((obj > int(2**32 - 1)) | (obj < 0)).any():
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            obj = obj.astype(np.uint32, casting='unsafe', order='C')
            mt19937_init_by_array(self.rng_state, <uint32_t*> obj.data, np.PyArray_DIM(obj, 0))

    def jump(self):
        mt19937_jump(self.rng_state)
        return self

    @property
    def state(self):
        """Get or set the PRNG state"""
        key = np.zeros(624, dtype=np.uint32)
        for i in range(624):
            key[i] = self.rng_state.key[i]

        return {'prng': self.__class__.__name__,
                'state': {'key':key, 'pos': self.rng_state.pos}}

    @state.setter
    def state(self, value):
        if isinstance(value, tuple):
            if value[0] != 'MT19937' or len(value) not in (3,5):
                    raise ValueError('state is not a legacy MT19937 state')
            value ={'prng': 'MT19937',
                    'state':{'key': value[1], 'pos': value[2]}}


        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        prng = value.get('prng', '')
        if prng != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        key = value['state']['key']
        for i in range(624):
            self.rng_state.key[i] = key[i]
        self.rng_state.pos = value['state']['pos']
