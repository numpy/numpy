from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np

from common import interface
from common cimport *
from distributions cimport brng_t
from randomgen.entropy import random_entropy, seed_by_array
import randomgen.pickle

np.import_array()

DEF PHILOX_BUFFER_SIZE=4

cdef extern from 'src/philox/philox.h':
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
        uint64_t buffer[PHILOX_BUFFER_SIZE];
        int has_uint32
        uint32_t uinteger

    ctypedef s_philox_state philox_state

    uint64_t philox_next64(philox_state *state)  nogil
    uint32_t philox_next32(philox_state *state)  nogil
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
    Philox(seed=None, counter=None, key=None)

    Container for the Philox (4x64) pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64-1], array of integers in
        [0, 2**64-1] or ``None`` (the default). If `seed` is ``None``,
        data will be read from ``/dev/urandom`` (or the Windows analog)
        if available.  If unavailable, a hash of the time and process ID is
        used.
    counter : {None, int, array_like}, optional
        Counter to use in the Philox state. Can be either
        a Python int (long in 2.x) in [0, 2**256) or a 4-element uint64 array.
        If not provided, the RNG is initialized at 0.
    key : {None, int, array_like}, optional
        Key to use in the Philox state.  Unlike seed, which is run through
        another RNG before use, the value in key is directly set. Can be either
        a Python int (long in 2.x) in [0, 2**128) or a 2-element uint64 array.
        key and seed cannot both be used.

    Notes
    -----
    Philox is a 64-bit PRNG that uses a counter-based design based on weaker
    (and faster) versions of cryptographic functions [1]_. Instances using
    different values of the key produce independent sequences.  Philox has a
    period of :math:`2^{256} - 1` and supports arbitrary advancing and jumping
    the sequence in increments of :math:`2^{128}`. These features allow
    multiple non-overlapping sequences to be generated.

    ``Philox`` exposes no user-facing API except ``generator``,
    ``state``, ``cffi`` and ``ctypes``. Designed for use in a
    ``RandomGenerator`` object.

    **Compatibility Guarantee**

    ``Philox`` guarantees that a fixed seed will always produce the
    same results.

    See ``Philox`` for a closely related PRNG implementation.

    **Parallel Features**

    ``Philox`` can be used in parallel applications by
    calling the method ``jump`` which advances the state as-if
    :math:`2^{128}` random numbers have been generated. Alternatively,
    ``advance`` can be used to advance the counter for an abritrary number of
    positive steps in [0, 2**256). When using ``jump``, all generators should
    be initialized with the same seed to ensure that the segments come from
    the same sequence. Alternatively, ``Philox`` can be used
    in parallel applications by using a sequence of distinct keys where each
    instance uses different key.

    >>> from randomgen import RandomGenerator, Philox
    >>> rg = [RandomGenerator(Philox(1234)) for _ in range(10)]
    # Advance rs[i] by i jumps
    >>> for i in range(10):
    ...     rg[i].jump(i)

    Using distinct keys produces independent streams

    >>> key = 2**196 + 2**132 + 2**65 + 2**33 + 2**17 + 2**9
    >>> rg = [RandomGenerator(Philox(key=key+i)) for i in range(10)]

    **State and Seeding**

    The ``Philox`` state vector consists of a 256-bit counter encoded as a
    4-element uint64 array and a 128-bit key encoded as a 2-element uint64
    array. The counter is incremented by 1 for every 4 64-bit randoms
    produced.  The key determines the sequence produced.  Using different
    keys produces independent sequences.

    ``Philox`` is seeded using either a single 64-bit unsigned integer
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
    >>> from randomgen import RandomGenerator, Philox
    >>> rg = RandomGenerator(Philox(1234))
    >>> rg.standard_normal()

    Identical method using only Philox

    >>> rg = Philox(1234).generator
    >>> rg.standard_normal()

    References
    ----------
    .. [1] John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw,
           "Parallel Random Numbers: As Easy as 1, 2, 3," Proceedings of
           the International Conference for High Performance Computing,
           Networking, Storage and Analysis (SC11), New York, NY: ACM, 2011.
    """
    cdef philox_state  *rng_state
    cdef brng_t *_brng
    cdef public object capsule
    cdef object _ctypes
    cdef object _cffi
    cdef object _generator

    def __init__(self, seed=None, counter=None, key=None):
        self.rng_state = <philox_state *> malloc(sizeof(philox_state))
        self.rng_state.ctr = <philox4x64_ctr_t *> malloc(
            sizeof(philox4x64_ctr_t))
        self.rng_state.key = <philox4x64_key_t *> malloc(
            sizeof(philox4x64_key_t))
        self._brng = <brng_t *> malloc(sizeof(brng_t))
        self.seed(seed, counter, key)

        self._brng.state = <void *> self.rng_state
        self._brng.next_uint64 = &philox_uint64
        self._brng.next_uint32 = &philox_uint32
        self._brng.next_double = &philox_double
        self._brng.next_raw = &philox_uint64

        self._ctypes = None
        self._cffi = None
        self._generator = None

        cdef const char *name = 'BasicRNG'
        self.capsule = PyCapsule_New(<void *> self._brng, name, NULL)

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
        free(self.rng_state.ctr)
        free(self.rng_state.key)
        free(self.rng_state)
        free(self._brng)

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0
        self.rng_state.buffer_pos = PHILOX_BUFFER_SIZE
        for i in range(PHILOX_BUFFER_SIZE):
            self.rng_state.buffer[i] = 0

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

    def seed(self, seed=None, counter=None, key=None):
        """
        seed(seed=None, counter=None, key=None)

        Seed the generator.

        This method is called when ``Philox`` is initialized. It can be
        called again to re-seed the generator. For details, see
        ``Philox``.

        Parameters
        ----------
        seed : int, optional
            Seed for ``Philox``.
        counter : {int array}, optional
            Positive integer less than 2**256 containing the counter position
            or a 4 element array of uint64 containing the counter
        key : {int, array}, options
            Positive integer less than 2**128 containing the key
            or a 2 element array of uint64 containing the key

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
                    state = random_entropy(4)
                except RuntimeError:
                    state = random_entropy(4, 'fallback')
                state = state.view(np.uint64)
            else:
                state = seed_by_array(seed, 2)
            for i in range(2):
                self.rng_state.key.v[i] = state[i]
        else:
            key = int_to_array(key, 'key', 128, 64)
            for i in range(2):
                self.rng_state.key.v[i] = key[i]
        counter = 0 if counter is None else counter
        counter = int_to_array(counter, 'counter', 256, 64)
        for i in range(4):
            self.rng_state.ctr.v[i] = counter[i]

        self._reset_state_variables()

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
        ctr = np.empty(4, dtype=np.uint64)
        key = np.empty(2, dtype=np.uint64)
        buffer = np.empty(PHILOX_BUFFER_SIZE, dtype=np.uint64)
        for i in range(4):
            ctr[i] = self.rng_state.ctr.v[i]
            if i < 2:
                key[i] = self.rng_state.key.v[i]
        for i in range(PHILOX_BUFFER_SIZE):
            buffer[i] = self.rng_state.buffer[i]

        state = {'counter': ctr, 'key': key}
        return {'brng': self.__class__.__name__,
                'state': state,
                'buffer': buffer,
                'buffer_pos': self.rng_state.buffer_pos,
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
        for i in range(4):
            self.rng_state.ctr.v[i] = <uint64_t> value['state']['counter'][i]
            if i < 2:
                self.rng_state.key.v[i] = <uint64_t> value['state']['key'][i]
        for i in range(PHILOX_BUFFER_SIZE):
            self.rng_state.buffer[i] = <uint64_t> value['buffer'][i]

        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']
        self.rng_state.buffer_pos = value['buffer_pos']

    def jump(self, np.npy_intp iter):
        """
        jump(iter=1)

        Jumps the state as-if 2**128 random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the rng.

        Returns
        -------
        self : Philox
            PRNG jumped iter times

        Notes
        -----
        Jumping the rng state resets any pre-computed random numbers. This is
        required to ensure exact reproducibility.
        """
        return self.advance(iter * 2 ** 128)

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
        self : Philox
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
        cdef np.ndarray delta_a
        delta_a = int_to_array(delta, 'step', 256, 64)
        philox_advance(<uint64_t *> delta_a.data, self.rng_state)
        self._reset_state_variables()
        return self

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
                         ctypes.cast(<uintptr_t>&philox_uint64,
                                     ctypes.CFUNCTYPE(ctypes.c_uint64,
                                     ctypes.c_void_p)),
                         ctypes.cast(<uintptr_t>&philox_uint32,
                                     ctypes.CFUNCTYPE(ctypes.c_uint32,
                                     ctypes.c_void_p)),
                         ctypes.cast(<uintptr_t>&philox_double,
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