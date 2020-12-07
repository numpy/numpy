import numpy as np
cimport numpy as np

from libc.stdint cimport uint32_t, uint64_t
from ._common cimport uint64_to_double
from numpy.random cimport BitGenerator

__all__ = ['SFC64']

cdef extern from "src/sfc64/sfc64.h":
    struct s_sfc64_state:
        uint64_t s[4]
        int has_uint32
        uint32_t uinteger

    ctypedef s_sfc64_state sfc64_state
    uint64_t sfc64_next64(sfc64_state *state)  nogil
    uint32_t sfc64_next32(sfc64_state *state)  nogil
    void sfc64_set_seed(sfc64_state *state, uint64_t *seed)
    void sfc64_get_state(sfc64_state *state, uint64_t *state_arr, int *has_uint32, uint32_t *uinteger)
    void sfc64_set_state(sfc64_state *state, uint64_t *state_arr, int has_uint32, uint32_t uinteger)


cdef uint64_t sfc64_uint64(void* st) nogil:
    return sfc64_next64(<sfc64_state *>st)

cdef uint32_t sfc64_uint32(void *st) nogil:
    return sfc64_next32(<sfc64_state *> st)

cdef double sfc64_double(void* st) nogil:
    return uint64_to_double(sfc64_next64(<sfc64_state *>st))


cdef class SFC64(BitGenerator):
    """
    SFC64(seed=None)

    BitGenerator for Chris Doty-Humphrey's Small Fast Chaotic PRNG.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.

    Notes
    -----
    ``SFC64`` is a 256-bit implementation of Chris Doty-Humphrey's Small Fast
    Chaotic PRNG ([1]_). ``SFC64`` has a few different cycles that one might be
    on, depending on the seed; the expected period will be about
    :math:`2^{255}` ([2]_). ``SFC64`` incorporates a 64-bit counter which means
    that the absolute minimum cycle length is :math:`2^{64}` and that distinct
    seeds will not run into each other for at least :math:`2^{64}` iterations.

    ``SFC64`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    **State and Seeding**

    The ``SFC64`` state vector consists of 4 unsigned 64-bit values. The last
    is a 64-bit counter that increments by 1 each iteration.

    The input seed is processed by `SeedSequence` to generate the first
    3 values, then the ``SFC64`` algorithm is iterated a small number of times
    to mix.

    **Compatibility Guarantee**

    ``SFC64`` makes a guarantee that a fixed seed will always produce the same
    random integer stream.

    References
    ----------
    .. [1] `"PractRand"
            <http://pracrand.sourceforge.net/RNG_engines.txt>`_
    .. [2] `"Random Invertible Mapping Statistics"
            <http://www.pcg-random.org/posts/random-invertible-mapping-statistics.html>`_
    """

    cdef sfc64_state rng_state

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed)
        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &sfc64_uint64
        self._bitgen.next_uint32 = &sfc64_uint32
        self._bitgen.next_double = &sfc64_double
        self._bitgen.next_raw = &sfc64_uint64
        # Seed the _bitgen
        val = self._seed_seq.generate_state(3, np.uint64)
        sfc64_set_seed(&self.rng_state, <uint64_t*>np.PyArray_DATA(val))
        self._reset_state_variables()

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    @property
    def state(self):
        """
        Get or set the PRNG state

        Returns
        -------
        state : dict
            Dictionary containing the information required to describe the
            state of the PRNG
        """
        cdef np.ndarray state_vec
        cdef int has_uint32
        cdef uint32_t uinteger

        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        sfc64_get_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        &has_uint32, &uinteger)
        return {'bit_generator': self.__class__.__name__,
                'state': {'state': state_vec},
                'has_uint32': has_uint32,
                'uinteger': uinteger}

    @state.setter
    def state(self, value):
        cdef np.ndarray state_vec
        cdef int has_uint32
        cdef uint32_t uinteger
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'RNG'.format(self.__class__.__name__))
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        state_vec[:] = value['state']['state']
        has_uint32 = value['has_uint32']
        uinteger = value['uinteger']
        sfc64_set_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        has_uint32, uinteger)
