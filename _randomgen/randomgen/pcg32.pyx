from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from common cimport *
from distributions cimport brng_t
from randomgen.entropy import random_entropy
import randomgen.pickle

np.import_array()


cdef extern from "src/pcg32/pcg32.h":

    cdef struct pcg_state_setseq_64:
        uint64_t state
        uint64_t inc

    ctypedef pcg_state_setseq_64 pcg32_random_t

    struct s_pcg32_state:
      pcg32_random_t *pcg_state

    ctypedef s_pcg32_state pcg32_state

    uint64_t pcg32_next64(pcg32_state *state)  nogil
    uint32_t pcg32_next32(pcg32_state *state)  nogil
    double pcg32_next_double(pcg32_state *state)  nogil
    void pcg32_jump(pcg32_state  *state)
    void pcg32_advance_state(pcg32_state *state, uint64_t step)
    void pcg32_set_seed(pcg32_state *state, uint64_t seed, uint64_t inc)

cdef uint64_t pcg32_uint64(void* st) nogil:
    return pcg32_next64(<pcg32_state *>st)

cdef uint32_t pcg32_uint32(void *st) nogil:
    return pcg32_next32(<pcg32_state *> st)

cdef double pcg32_double(void* st) nogil:
    return pcg32_next_double(<pcg32_state *>st)

cdef uint64_t pcg32_raw(void* st) nogil:
    return <uint64_t>pcg32_next32(<pcg32_state *> st)


cdef class PCG32:
    """
    Prototype Basic RNG using pcg64

    Parameters
    ----------
    seed : int, array of int
        Integer or array of integers between 0 and 2**64 - 1

    Notes
    -----
    Exposes no user-facing API except `get_state` and `set_state`. Designed
    for use in a `RandomGenerator` object.
    """
    cdef pcg32_state *rng_state
    cdef brng_t *_brng
    cdef public object capsule

    def __init__(self, seed=None, inc=0):
        self.rng_state = <pcg32_state *>malloc(sizeof(pcg32_state))
        self.rng_state.pcg_state = <pcg32_random_t *>malloc(sizeof(pcg32_random_t))
        self._brng = <brng_t *>malloc(sizeof(brng_t))
        self.seed(seed, inc)

        self._brng.state = <void *>self.rng_state
        self._brng.next_uint64 = &pcg32_uint64
        self._brng.next_uint32 = &pcg32_uint32
        self._brng.next_double = &pcg32_double
        self._brng.next_raw = &pcg32_raw

        cdef const char *name = "BasicRNG"
        self.capsule = PyCapsule_New(<void *>self._brng, name, NULL)

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        return (randomgen.pickle.__brng_ctor,
                (self.state['brng'],),
                self.state)

    def __dealloc__(self):
        free(self.rng_state)
        free(self._brng)

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
            return self._brng.next_uint64(self._brng.state)
        elif bits == 32:
            return self._brng.next_uint32(self._brng.state)
        else:
            raise ValueError('bits must be 32 or 64')

    def _benchmark(self, Py_ssize_t cnt, method=u'uint64'):
        cdef Py_ssize_t i
        if method==u'uint64':
            for i in range(cnt):
                self._brng.next_uint64(self._brng.state)
        elif method==u'double':
            for i in range(cnt):
                self._brng.next_double(self._brng.state)
        else:
            raise ValueError('Unknown method')


    def seed(self, seed=None, inc=0):
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
                seed = <np.ndarray>random_entropy(2)
            except RuntimeError:
                seed = <np.ndarray>random_entropy(2, 'fallback')
            seed = seed.view(np.uint64).squeeze()
        else:
            err_msg = 'seed must be a scalar integer between 0 and ' \
                      '{ub}'.format(ub=ub)
            if not np.isscalar(seed):
                raise TypeError(err_msg)
            if int(seed) != seed:
                raise TypeError(err_msg)
            if seed < 0 or seed > ub:
                raise ValueError(err_msg)

        if not np.isscalar(inc):
            raise TypeError('inc must be a scalar integer between 0 '
                            'and {ub}'.format(ub=ub))
        if inc < 0 or inc > ub or int(inc) != inc:
            raise ValueError('inc must be a scalar integer between 0 '
                             'and {ub}'.format(ub=ub))

        pcg32_set_seed(self.rng_state, <uint64_t>seed, <uint64_t>inc)

    @property
    def state(self):
        """Get or set the PRNG state"""
        return {'brng': self.__class__.__name__,
                'state': {'state': self.rng_state.pcg_state.state,
                          'inc':self.rng_state.pcg_state.inc}}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        brng = value.get('brng', '')
        if brng != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        self.rng_state.pcg_state.state  = value['state']['state']
        self.rng_state.pcg_state.inc = value['state']['inc']

    def advance(self, step):
        pcg32_advance_state(self.rng_state, <uint64_t>step)
        return self

    def jump(self):
        return self.advance(2**32)
