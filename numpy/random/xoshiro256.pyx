try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock

from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from .common cimport *
from .distributions cimport bitgen_t
from .entropy import random_entropy, seed_by_array

__all__ = ['Xoshiro256']

np.import_array()

cdef extern from "src/xoshiro256/xoshiro256.h":

    struct s_xoshiro256_state:
        uint64_t s[4]
        int has_uint32
        uint32_t uinteger

    ctypedef s_xoshiro256_state xoshiro256_state

    uint64_t xoshiro256_next64(xoshiro256_state *state)  nogil
    uint32_t xoshiro256_next32(xoshiro256_state *state)  nogil
    void xoshiro256_jump(xoshiro256_state *state)

cdef uint64_t xoshiro256_uint64(void* st) nogil:
    return xoshiro256_next64(<xoshiro256_state *>st)

cdef uint32_t xoshiro256_uint32(void *st) nogil:
    return xoshiro256_next32(<xoshiro256_state *> st)

cdef double xoshiro256_double(void* st) nogil:
    return uint64_to_double(xoshiro256_next64(<xoshiro256_state *>st))

cdef class Xoshiro256:
    """
    Xoshiro256(seed=None)

    Container for the xoshiro256** pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64-1], array of integers in [0, 2**64-1]
        or ``None`` (the default). If `seed` is ``None``, then  data is read
        from ``/dev/urandom`` (or the Windows analog) if available.  If
        unavailable, a hash of the time and process ID is used.

    Attributes
    ----------
    lock: threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.

    Notes
    -----
    xoshiro256** is written by David Blackman and Sebastiano Vigna.
    It is a 64-bit PRNG that uses a carefully constructed linear transformation.
    This produces a fast PRNG with excellent statistical quality
    [1]_. xoshiro256** has a period of :math:`2^{256} - 1`
    and supports jumping the sequence in increments of :math:`2^{128}`,
    which allows multiple non-overlapping subsequences to be generated.

    ``Xoshiro256`` provides a capsule containing function pointers that
    produce doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    See ``Xoshiro512`` for a related PRNG with a different period
    (:math:`2^{1024} - 1`) and jumped size (:math:`2^{512} - 1`).

    **State and Seeding**

    The ``Xoshiro256`` state vector consists of a 4-element array
    of 64-bit unsigned integers.

    ``Xoshiro256`` is seeded using either a single 64-bit unsigned
    integer or a vector of 64-bit unsigned integers.  In either case, the seed
    is used as an input for another simple random number generator, SplitMix64,
    and the output of this PRNG function is used as the initial state. Using
    a single 64-bit value for the seed can only initialize a small range of
    the possible initial state values.

    **Parallel Features**

    ``Xoshiro256`` can be used in parallel applications by calling the
    method ``jumped`` which advances the state as-if :math:`2^{128}` random
    numbers have been generated. This allows the original sequence to be split
    so that distinct segments can be used in each worker process. All
    generators should be chained to ensure that the segments come from the same
    sequence.

    >>> from numpy.random import Generator, Xoshiro256
    >>> bit_generator = Xoshiro256(1234)
    >>> rg = []
    >>> for _ in range(10):
    ...    rg.append(Generator(bit_generator))
    ...    bit_generator = bit_generator.jumped()

    **Compatibility Guarantee**

    ``Xoshiro256`` makes a guarantee that a fixed seed will always
    produce the same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator, Xoshiro256
    >>> rg = Generator(Xoshiro256(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] "xoroshiro+ / xorshift* / xorshift+ generators and the PRNG shootout",
           http://xorshift.di.unimi.it/
    """
    cdef xoshiro256_state rng_state
    cdef bitgen_t _bitgen
    cdef public object capsule
    cdef object _ctypes
    cdef object _cffi
    cdef public object lock

    def __init__(self, seed=None):
        self.seed(seed)
        self.lock = Lock()

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &xoshiro256_uint64
        self._bitgen.next_uint32 = &xoshiro256_uint32
        self._bitgen.next_double = &xoshiro256_double
        self._bitgen.next_raw = &xoshiro256_uint64

        self._ctypes = None
        self._cffi = None

        cdef const char *name = "BitGenerator"
        self.capsule = PyCapsule_New(<void *>&self._bitgen, name, NULL)

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        from ._pickle import __bit_generator_ctor
        return __bit_generator_ctor, (self.state['bit_generator'],), self.state

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def random_raw(self, size=None, output=True):
        """
        random_raw(self, size=None)

        Return randoms as generated by the underlying BitGenerator

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        output : bool, optional
            Output values.  Used for performance testing since the generated
            values are not returned.

        Returns
        -------
        out : uint or ndarray
            Drawn samples.

        Notes
        -----
        This method directly exposes the the raw underlying pseudo-random
        number generator. All values are returned as unsigned 64-bit
        values irrespective of the number of bits produced by the PRNG.

        See the class docstring for the number of bits returned.
        """
        return random_raw(&self._bitgen, self.lock, size, output)

    def _benchmark(self, Py_ssize_t cnt, method=u'uint64'):
        return benchmark(&self._bitgen, self.lock, cnt, method)

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        This method is called at initialization. It can be called again to
        re-seed the generator.

        Parameters
        ----------
        seed : {int, ndarray}, optional
            Seed for PRNG. Can be a single 64 bit unsigned integer or an array
            of 64 bit unsigned integers.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        ub = 2 ** 64
        if seed is None:
            try:
                state = random_entropy(8)
            except RuntimeError:
                state = random_entropy(8, 'fallback')
            state = state.view(np.uint64)
        else:
            state = seed_by_array(seed, 4)
        self.rng_state.s[0] = <uint64_t>int(state[0])
        self.rng_state.s[1] = <uint64_t>int(state[1])
        self.rng_state.s[2] = <uint64_t>int(state[2])
        self.rng_state.s[3] = <uint64_t>int(state[3])
        self._reset_state_variables()

    cdef jump_inplace(self, np.npy_intp iter):
        """
        Jump state in-place

        Not part of public API

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.
        """
        cdef np.npy_intp i
        for i in range(iter):
            xoshiro256_jump(&self.rng_state)
        self._reset_state_variables()

    def jumped(self, np.npy_intp iter=1):
        """
        jumped(iter=1)

        Returns a new bit generator with the state jumped

        The state of the returned big generator is jumped as-if
        2**(128 * iter) random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : Xoshiro256
            New instance of generator jumped iter times
        """
        cdef Xoshiro256 bit_generator

        bit_generator = self.__class__()
        bit_generator.state = self.state
        bit_generator.jump_inplace(iter)

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
        state = np.empty(4, dtype=np.uint64)
        state[0] = self.rng_state.s[0]
        state[1] = self.rng_state.s[1]
        state[2] = self.rng_state.s[2]
        state[3] = self.rng_state.s[3]
        return {'bit_generator': self.__class__.__name__,
                's': state,
                'has_uint32': self.rng_state.has_uint32,
                'uinteger': self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        self.rng_state.s[0] = <uint64_t>value['s'][0]
        self.rng_state.s[1] = <uint64_t>value['s'][1]
        self.rng_state.s[2] = <uint64_t>value['s'][2]
        self.rng_state.s[3] = <uint64_t>value['s'][3]
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']

    @property
    def ctypes(self):
        """
        ctypes interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing ctypes wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * bitgen - pointer to the bit generator struct
        """
        if self._ctypes is None:
            self._ctypes = prepare_ctypes(&self._bitgen)

        return self._ctypes

    @property
    def cffi(self):
        """
        CFFI interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing CFFI wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * bitgen - pointer to the bit generator struct
        """
        if self._cffi is not None:
            return self._cffi
        self._cffi = prepare_cffi(&self._bitgen)
        return self._cffi
