#cython: binding=True

import numpy as np
cimport numpy as np

from libc.stdint cimport uint32_t, uint64_t
from ._common cimport uint64_to_double, wrap_int
from numpy.random cimport BitGenerator

__all__ = ['PCG64']

cdef extern from "src/pcg64/pcg64.h":
    # Use int as generic type, actual type read from pcg64.h and is platform dependent
    ctypedef int pcg64_random_t

    struct s_pcg64_state:
        pcg64_random_t *pcg_state
        int has_uint32
        uint32_t uinteger

    ctypedef s_pcg64_state pcg64_state

    uint64_t pcg64_next64(pcg64_state *state)  nogil
    uint32_t pcg64_next32(pcg64_state *state)  nogil
    void pcg64_jump(pcg64_state *state)
    void pcg64_advance(pcg64_state *state, uint64_t *step)
    void pcg64_set_seed(pcg64_state *state, uint64_t *seed, uint64_t *inc)
    void pcg64_get_state(pcg64_state *state, uint64_t *state_arr, int *has_uint32, uint32_t *uinteger)
    void pcg64_set_state(pcg64_state *state, uint64_t *state_arr, int has_uint32, uint32_t uinteger)

    uint64_t pcg64_cm_next64(pcg64_state *state)  noexcept nogil
    uint32_t pcg64_cm_next32(pcg64_state *state)  noexcept nogil
    void pcg64_cm_advance(pcg64_state *state, uint64_t *step)

cdef uint64_t pcg64_uint64(void* st) noexcept nogil:
    return pcg64_next64(<pcg64_state *>st)

cdef uint32_t pcg64_uint32(void *st) noexcept nogil:
    return pcg64_next32(<pcg64_state *> st)

cdef double pcg64_double(void* st) noexcept nogil:
    return uint64_to_double(pcg64_next64(<pcg64_state *>st))

cdef uint64_t pcg64_cm_uint64(void* st) noexcept nogil:
    return pcg64_cm_next64(<pcg64_state *>st)

cdef uint32_t pcg64_cm_uint32(void *st) noexcept nogil:
    return pcg64_cm_next32(<pcg64_state *> st)

cdef double pcg64_cm_double(void* st) noexcept nogil:
    return uint64_to_double(pcg64_cm_next64(<pcg64_state *>st))

cdef class PCG64(BitGenerator):
    # the first line is used to populate `__text_signature__`
    """PCG64(seed=None)\n--

    BitGenerator for the PCG-64 pseudo-random number generator.

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
    PCG-64 is a 128-bit implementation of O'Neill's permutation congruential
    generator ([1]_, [2]_). PCG-64 has a period of :math:`2^{128}` and supports
    advancing an arbitrary number of steps as well as :math:`2^{127}` streams.
    The specific member of the PCG family that we use is PCG XSL RR 128/64
    as described in the paper ([2]_).

    `PCG64` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a `Generator`
    or similar object that supports low-level access.

    Supports the method :meth:`advance` to advance the RNG an arbitrary number of
    steps. The state of the PCG-64 RNG is represented by 2 128-bit unsigned
    integers.

    **State and Seeding**

    The `PCG64` state vector consists of 2 unsigned 128-bit values,
    which are represented externally as Python ints. One is the state of the
    PRNG, which is advanced by a linear congruential generator (LCG). The
    second is a fixed odd increment used in the LCG.

    The input seed is processed by `SeedSequence` to generate both values. The
    increment is not independently settable.

    **Parallel Features**

    The preferred way to use a BitGenerator in parallel applications is to use
    the `SeedSequence.spawn` method to obtain entropy values, and to use these
    to generate new BitGenerators:

    >>> from numpy.random import Generator, PCG64, SeedSequence
    >>> sg = SeedSequence(1234)
    >>> rg = [Generator(PCG64(s)) for s in sg.spawn(10)]

    **Compatibility Guarantee**

    `PCG64` makes a guarantee that a fixed seed will always produce
    the same random integer stream.

    References
    ----------
    .. [1] `"PCG, A Family of Better Random Number Generators"
           <https://www.pcg-random.org/>`_
    .. [2] O'Neill, Melissa E. `"PCG: A Family of Simple Fast Space-Efficient
           Statistically Good Algorithms for Random Number Generation"
           <https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf>`_
    """

    cdef pcg64_state rng_state
    cdef pcg64_random_t pcg64_random_state

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed)
        self.rng_state.pcg_state = &self.pcg64_random_state

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &pcg64_uint64
        self._bitgen.next_uint32 = &pcg64_uint32
        self._bitgen.next_double = &pcg64_double
        self._bitgen.next_raw = &pcg64_uint64
        # Seed the _bitgen
        val = self._seed_seq.generate_state(4, np.uint64)
        pcg64_set_seed(&self.rng_state,
                       <uint64_t *>np.PyArray_DATA(val),
                       (<uint64_t *>np.PyArray_DATA(val) + 2))
        self._reset_state_variables()

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    cdef jump_inplace(self, jumps):
        """
        Jump state in-place
        Not part of public API

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the rng.

        Notes
        -----
        The step size is phi-1 when multiplied by 2**128 where phi is the
        golden ratio.
        """
        step = 0x9e3779b97f4a7c15f39cc0605cedc835
        self.advance(step * int(jumps))

    def jumped(self, jumps=1):
        """
        jumped(jumps=1)

        Returns a new bit generator with the state jumped.

        Jumps the state as-if jumps * 210306068529402873165736369884012333109
        random numbers have been generated.

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : PCG64
            New instance of generator jumped iter times

        Notes
        -----
        The step size is phi-1 when multiplied by 2**128 where phi is the
        golden ratio.
        """
        cdef PCG64 bit_generator

        bit_generator = self.__class__()
        bit_generator.state = self.state
        bit_generator.jump_inplace(jumps)

        return bit_generator

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

        # state_vec is state.high, state.low, inc.high, inc.low
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        pcg64_get_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        &has_uint32, &uinteger)
        state = int(state_vec[0]) * 2**64 + int(state_vec[1])
        inc = int(state_vec[2]) * 2**64 + int(state_vec[3])
        return {'bit_generator': self.__class__.__name__,
                'state': {'state': state, 'inc': inc},
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
            raise ValueError(f'state must be for a {self.__class__.__name__} RNG')
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        state_vec[0] = value['state']['state'] // 2 ** 64
        state_vec[1] = value['state']['state'] % 2 ** 64
        state_vec[2] = value['state']['inc'] // 2 ** 64
        state_vec[3] = value['state']['inc'] % 2 ** 64
        has_uint32 = value['has_uint32']
        uinteger = value['uinteger']
        pcg64_set_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        has_uint32, uinteger)

    def advance(self, delta):
        """
        advance(delta)

        Advance the underlying RNG as-if delta draws have occurred.

        Parameters
        ----------
        delta : integer, positive
            Number of draws to advance the RNG. Must be less than the
            size state variable in the underlying RNG.

        Returns
        -------
        self : PCG64
            RNG advanced delta steps

        Notes
        -----
        Advancing a RNG updates the underlying RNG state as-if a given
        number of calls to the underlying RNG have been made. In general
        there is not a one-to-one relationship between the number output
        random values from a particular distribution and the number of
        draws from the core RNG.  This occurs for two reasons:

        * The random values are simulated using a rejection-based method
          and so, on average, more than one value from the underlying
          RNG is required to generate an single draw.
        * The number of bits required to generate a simulated value
          differs from the number of bits generated by the underlying
          RNG.  For example, two 16-bit integer values can be simulated
          from a single draw of a 32-bit RNG.

        Advancing the RNG state resets any pre-computed random numbers.
        This is required to ensure exact reproducibility.
        """
        delta = wrap_int(delta, 128)

        cdef np.ndarray d = np.empty(2, dtype=np.uint64)
        d[0] = delta // 2**64
        d[1] = delta % 2**64
        pcg64_advance(&self.rng_state, <uint64_t *>np.PyArray_DATA(d))
        self._reset_state_variables()
        return self


cdef class PCG64DXSM(BitGenerator):
    # the first line is used to populate `__text_signature__`
    """PCG64DXSM(seed=None)\n--

    BitGenerator for the PCG-64 DXSM pseudo-random number generator.

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
    PCG-64 DXSM is a 128-bit implementation of O'Neill's permutation congruential
    generator ([1]_, [2]_). PCG-64 DXSM has a period of :math:`2^{128}` and supports
    advancing an arbitrary number of steps as well as :math:`2^{127}` streams.
    The specific member of the PCG family that we use is PCG CM DXSM 128/64. It
    differs from `PCG64` in that it uses the stronger DXSM output function,
    a 64-bit "cheap multiplier" in the LCG, and outputs from the state before
    advancing it rather than advance-then-output.

    `PCG64DXSM` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a `Generator`
    or similar object that supports low-level access.

    Supports the method :meth:`advance` to advance the RNG an arbitrary number of
    steps. The state of the PCG-64 DXSM RNG is represented by 2 128-bit unsigned
    integers.

    **State and Seeding**

    The `PCG64DXSM` state vector consists of 2 unsigned 128-bit values,
    which are represented externally as Python ints. One is the state of the
    PRNG, which is advanced by a linear congruential generator (LCG). The
    second is a fixed odd increment used in the LCG.

    The input seed is processed by `SeedSequence` to generate both values. The
    increment is not independently settable.

    **Parallel Features**

    The preferred way to use a BitGenerator in parallel applications is to use
    the `SeedSequence.spawn` method to obtain entropy values, and to use these
    to generate new BitGenerators:

    >>> from numpy.random import Generator, PCG64DXSM, SeedSequence
    >>> sg = SeedSequence(1234)
    >>> rg = [Generator(PCG64DXSM(s)) for s in sg.spawn(10)]

    **Compatibility Guarantee**

    `PCG64DXSM` makes a guarantee that a fixed seed will always produce
    the same random integer stream.

    References
    ----------
    .. [1] `"PCG, A Family of Better Random Number Generators"
           <http://www.pcg-random.org/>`_
    .. [2] O'Neill, Melissa E. `"PCG: A Family of Simple Fast Space-Efficient
           Statistically Good Algorithms for Random Number Generation"
           <https://www.cs.hmc.edu/tr/hmc-cs-2014-0905.pdf>`_
    """
    cdef pcg64_state rng_state
    cdef pcg64_random_t pcg64_random_state

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed)
        self.rng_state.pcg_state = &self.pcg64_random_state

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &pcg64_cm_uint64
        self._bitgen.next_uint32 = &pcg64_cm_uint32
        self._bitgen.next_double = &pcg64_cm_double
        self._bitgen.next_raw = &pcg64_cm_uint64
        # Seed the _bitgen
        val = self._seed_seq.generate_state(4, np.uint64)
        pcg64_set_seed(&self.rng_state,
                       <uint64_t *>np.PyArray_DATA(val),
                       (<uint64_t *>np.PyArray_DATA(val) + 2))
        self._reset_state_variables()

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    cdef jump_inplace(self, jumps):
        """
        Jump state in-place
        Not part of public API

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the rng.

        Notes
        -----
        The step size is phi-1 when multiplied by 2**128 where phi is the
        golden ratio.
        """
        step = 0x9e3779b97f4a7c15f39cc0605cedc835
        self.advance(step * int(jumps))

    def jumped(self, jumps=1):
        """
        jumped(jumps=1)

        Returns a new bit generator with the state jumped.

        Jumps the state as-if jumps * 210306068529402873165736369884012333109
        random numbers have been generated.

        Parameters
        ----------
        jumps : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : PCG64DXSM
            New instance of generator jumped iter times

        Notes
        -----
        The step size is phi-1 when multiplied by 2**128 where phi is the
        golden ratio.
        """
        cdef PCG64DXSM bit_generator

        bit_generator = self.__class__()
        bit_generator.state = self.state
        bit_generator.jump_inplace(jumps)

        return bit_generator

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

        # state_vec is state.high, state.low, inc.high, inc.low
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        pcg64_get_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        &has_uint32, &uinteger)
        state = int(state_vec[0]) * 2**64 + int(state_vec[1])
        inc = int(state_vec[2]) * 2**64 + int(state_vec[3])
        return {'bit_generator': self.__class__.__name__,
                'state': {'state': state, 'inc': inc},
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
            raise ValueError(f'state must be for a {self.__class__.__name__} RNG')
        state_vec = <np.ndarray>np.empty(4, dtype=np.uint64)
        state_vec[0] = value['state']['state'] // 2 ** 64
        state_vec[1] = value['state']['state'] % 2 ** 64
        state_vec[2] = value['state']['inc'] // 2 ** 64
        state_vec[3] = value['state']['inc'] % 2 ** 64
        has_uint32 = value['has_uint32']
        uinteger = value['uinteger']
        pcg64_set_state(&self.rng_state,
                        <uint64_t *>np.PyArray_DATA(state_vec),
                        has_uint32, uinteger)

    def advance(self, delta):
        """
        advance(delta)

        Advance the underlying RNG as-if delta draws have occurred.

        Parameters
        ----------
        delta : integer, positive
            Number of draws to advance the RNG. Must be less than the
            size state variable in the underlying RNG.

        Returns
        -------
        self : PCG64
            RNG advanced delta steps

        Notes
        -----
        Advancing a RNG updates the underlying RNG state as-if a given
        number of calls to the underlying RNG have been made. In general
        there is not a one-to-one relationship between the number output
        random values from a particular distribution and the number of
        draws from the core RNG.  This occurs for two reasons:

        * The random values are simulated using a rejection-based method
          and so, on average, more than one value from the underlying
          RNG is required to generate an single draw.
        * The number of bits required to generate a simulated value
          differs from the number of bits generated by the underlying
          RNG.  For example, two 16-bit integer values can be simulated
          from a single draw of a 32-bit RNG.

        Advancing the RNG state resets any pre-computed random numbers.
        This is required to ensure exact reproducibility.
        """
        delta = wrap_int(delta, 128)

        cdef np.ndarray d = np.empty(2, dtype=np.uint64)
        d[0] = delta // 2**64
        d[1] = delta % 2**64
        pcg64_cm_advance(&self.rng_state, <uint64_t *>np.PyArray_DATA(d))
        self._reset_state_variables()
        return self
