from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from common import interface
from common cimport *
from distributions cimport brng_t
from randomgen.entropy import random_entropy, seed_by_array
import randomgen.pickle

np.import_array()

cdef extern from "src/xorshift1024/xorshift1024.h":

    struct s_xorshift1024_state:
      uint64_t s[16]
      int p
      int has_uint32
      uint32_t uinteger

    ctypedef s_xorshift1024_state xorshift1024_state

    uint64_t xorshift1024_next64(xorshift1024_state *state)  nogil
    uint32_t xorshift1024_next32(xorshift1024_state *state)  nogil
    void xorshift1024_jump(xorshift1024_state  *state)

cdef uint64_t xorshift1024_uint64(void* st) nogil:
    return xorshift1024_next64(<xorshift1024_state *>st)

cdef uint32_t xorshift1024_uint32(void *st) nogil:
    return xorshift1024_next32(<xorshift1024_state *> st)

cdef double xorshift1024_double(void* st) nogil:
    return uint64_to_double(xorshift1024_next64(<xorshift1024_state *>st))

cdef class Xorshift1024:
    u"""
    Xorshift1024(seed=None)

    Container for the xorshift1024*φ pseudo-random number generator.

    xorshift1024*φ is a 64-bit implementation of Saito and Matsumoto's XSadd
    generator [1]_ (see also [2]_, [3]_, [4]_). xorshift1024*φ has a period of
    :math:`2^{1024} - 1` and supports jumping the sequence in increments of
    :math:`2^{512}`, which allows multiple non-overlapping sequences to be
    generated.

    ``Xorshift1024`` exposes no user-facing API except ``generator``,
    ``state``, ``cffi`` and ``ctypes``. Designed for use in a
    ``RandomGenerator`` object.

    **Compatibility Guarantee**

    ``Xorshift1024`` guarantees that a fixed seed will always produce the
    same results.

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64-1], array of integers in
        [0, 2**64-1] or ``None`` (the default). If `seed` is ``None``,
        then ``Xorshift1024`` will try to read data from
        ``/dev/urandom`` (or the Windows analog) if available.  If
        unavailable, a 64-bit hash of the time and process ID is used.

    Notes
    -----
    See ``Xoroshiro128`` for a faster implementation that has a smaller
    period.

    **Parallel Features**

    ``Xorshift1024`` can be used in parallel applications by
    calling the method ``jump`` which advances the state as-if
    :math:`2^{512}` random numbers have been generated. This
    allows the original sequence to be split so that distinct segments can be used
    in each worker process. All generators should be initialized with the same
    seed to ensure that the segments come from the same sequence.

    >>> from randomgen import RandomGenerator, Xorshift1024
    >>> rg = [RandomGenerator(Xorshift1024(1234)) for _ in range(10)]
    # Advance rg[i] by i jumps
    >>> for i in range(10):
    ...     rg[i].jump(i)

    **State and Seeding**

    The ``Xorshift1024`` state vector consists of a 16 element array
    of 64-bit unsigned integers.

    ``Xorshift1024`` is seeded using either a single 64-bit unsigned integer
    or a vector of 64-bit unsigned integers.  In either case, the input seed is
    used as an input (or inputs) for another simple random number generator,
    Splitmix64, and the output of this PRNG function is used as the initial state.
    Using a single 64-bit value for the seed can only initialize a small range of
    the possible initial state values.  When using an array, the SplitMix64 state
    for producing the ith component of the initial state is XORd with the ith
    value of the seed array until the seed array is exhausted. When using an array
    the initial state for the SplitMix64 state is 0 so that using a single element
    array and using the same value as a scalar will produce the same initial state.

    Examples
    --------
    >>> from randomgen import RandomGenerator, Xorshift1024
    >>> rg = RandomGenerator(Xorshift1024(1234))
    >>> rg.standard_normal()

    Identical method using only Xoroshiro128

    >>> rg = Xorshift10241234).generator
    >>> rg.standard_normal()

    References
    ----------
    .. [1] "xorshift*/xorshift+ generators and the PRNG shootout",
           http://xorshift.di.unimi.it/
    .. [2] Marsaglia, George. "Xorshift RNGs." Journal of Statistical Software
           [Online], 8.14, pp. 1 - 6, .2003.
    .. [3] Sebastiano Vigna. "An experimental exploration of Marsaglia's xorshift
           generators, scrambled." CoRR, abs/1402.6246, 2014.
    .. [4] Sebastiano Vigna. "Further scramblings of Marsaglia's xorshift
           generators." CoRR, abs/1403.0930, 2014.
    """

    cdef xorshift1024_state  *rng_state
    cdef brng_t *_brng
    cdef public object capsule
    cdef object _ctypes
    cdef object _cffi
    cdef object _generator

    def __init__(self, seed=None):
        self.rng_state = <xorshift1024_state *>malloc(sizeof(xorshift1024_state))
        self._brng = <brng_t *>malloc(sizeof(brng_t))
        self.seed(seed)

        self._brng.state = <void *>self.rng_state
        self._brng.next_uint64 = &xorshift1024_uint64
        self._brng.next_uint32 = &xorshift1024_uint32
        self._brng.next_double = &xorshift1024_double
        self._brng.next_raw = &xorshift1024_uint64

        self._ctypes = None
        self._cffi = None
        self._generator = None

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

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

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

    def seed(self, seed=None):
        """
        seed(seed=None, stream=None)

        Seed the generator.

        This method is called when ``Xorshift1024`` is initialized. It can be
        called again to re-seed the generator. For details, see
        ``Xorshift1024``.

        Parameters
        ----------
        seed : int, optional
            Seed for ``Xorshift1024``.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.

        """
        ub =  2 ** 64
        if seed is None:
            try:
                state = random_entropy(32)
            except RuntimeError:
                state = random_entropy(4, 'fallback')
            state = state.view(np.uint64)
        else:
            state = seed_by_array(seed, 16)
        for i in range(16):
            self.rng_state.s[i] = <uint64_t>int(state[i])
        self.rng_state.p = 0
        self._reset_state_variables()

    def jump(self, np.npy_intp iter=1):
        """
        jump(iter=1)

        Jumps the state as-if 2**512 random numbers have been generated

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : Xorshift1024
            PRNG jumped iter times

        Notes
        -----
        Jumping the rng state resets any pre-computed random numbers. This is required
        to ensure exact reproducibility.
        """
        cdef np.npy_intp i
        for i in range(iter):
            xorshift1024_jump(self.rng_state)
        self._reset_state_variables()
        return self

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
        s = np.empty(16, dtype=np.uint64)
        for i in range(16):
            s[i] = self.rng_state.s[i]
        return {'brng': self.__class__.__name__,
                'state': {'s':s,'p':self.rng_state.p},
                'has_uint32': self.rng_state.has_uint32,
                'uinteger': self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        brng = value.get('brng', '')
        if brng != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        for i in range(16):
            self.rng_state.s[i] = <uint64_t>value['state']['s'][i]
        self.rng_state.p = value['state']['p']
        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']

    @property
    def ctypes(self):
        """
        Cytpes interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing CFFI wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * brng - pointer to the Basic RNG struct
        """

        if self._ctypes is not None:
            return self._ctypes

        import ctypes
        
        self._ctypes = interface(<uintptr_t>self.rng_state,
                         ctypes.c_void_p(<uintptr_t>self.rng_state),
                         ctypes.cast(<uintptr_t>&xorshift1024_uint64, 
                                     ctypes.CFUNCTYPE(ctypes.c_uint64, 
                                     ctypes.c_void_p)),
                         ctypes.cast(<uintptr_t>&xorshift1024_uint32, 
                                     ctypes.CFUNCTYPE(ctypes.c_uint32, 
                                     ctypes.c_void_p)),
                         ctypes.cast(<uintptr_t>&xorshift1024_double, 
                                     ctypes.CFUNCTYPE(ctypes.c_double, 
                                     ctypes.c_void_p)),
                         ctypes.c_void_p(<uintptr_t>self._brng))
        return self.ctypes

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
            * brng - pointer to the Basic RNG struct
        """
        if self._cffi is not None:
            return self._cffi
        try:
            import cffi 
        except ImportError:
            raise ImportError('cffi is cannot be imported.')

        ffi = cffi.FFI()
        self._cffi = interface(<uintptr_t>self.rng_state,
                         ffi.cast('void *',<uintptr_t>self.rng_state),
                         ffi.cast('uint64_t (*)(void *)',<uintptr_t>self._brng.next_uint64),
                         ffi.cast('uint32_t (*)(void *)',<uintptr_t>self._brng.next_uint32),
                         ffi.cast('double (*)(void *)',<uintptr_t>self._brng.next_double),
                         ffi.cast('void *',<uintptr_t>self._brng))
        return self.cffi

    @property
    def generator(self):
        """
        Return a RandomGenerator object

        Returns
        -------
        gen : randomgen.generator.RandomGenerator
            Random generator used this instance as the core PRNG
        """
        if self._generator is None:
            from .generator import RandomGenerator
            self._generator = RandomGenerator(self)
        return self._generator