import operator
from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from common import interface
from common cimport *
from distributions cimport brng_t
from randomgen.entropy import random_entropy
import randomgen.pickle

np.import_array()

DEF DSFMT_MEXP = 19937
DEF DSFMT_N = 191 # ((DSFMT_MEXP - 128) / 104 + 1)
DEF DSFMT_N_PLUS_1 = 192 # DSFMT_N + 1
DEF DSFMT_N64 = DSFMT_N * 2

cdef extern from "src/dsfmt/dSFMT.h":

    union W128_T:
        uint64_t u[2];
        uint32_t u32[4];
        double d[2];

    ctypedef W128_T w128_t;

    struct DSFMT_T:
        w128_t status[DSFMT_N_PLUS_1];
        int idx;

    ctypedef DSFMT_T dsfmt_t;

    struct s_dsfmt_state:
        dsfmt_t *state
        int has_uint32
        uint32_t uinteger

        double *buffered_uniforms
        int buffer_loc

    ctypedef s_dsfmt_state dsfmt_state

    double dsfmt_next_double(dsfmt_state *state)  nogil
    uint64_t dsfmt_next64(dsfmt_state *state)  nogil
    uint32_t dsfmt_next32(dsfmt_state *state)  nogil
    uint64_t dsfmt_next_raw(dsfmt_state *state)  nogil

    void dsfmt_init_gen_rand(dsfmt_t *dsfmt, uint32_t seed)
    void dsfmt_init_by_array(dsfmt_t *dsfmt, uint32_t init_key[], int key_length)
    void dsfmt_jump(dsfmt_state  *state);

cdef uint64_t dsfmt_uint64(void* st) nogil:
    return dsfmt_next64(<dsfmt_state *>st)

cdef uint32_t dsfmt_uint32(void *st) nogil:
    return dsfmt_next32(<dsfmt_state *> st)

cdef double dsfmt_double(void* st) nogil:
    return dsfmt_next_double(<dsfmt_state *>st)

cdef uint64_t dsfmt_raw(void *st) nogil:
    return dsfmt_next_raw(<dsfmt_state *>st)

cdef class DSFMT:
    u"""
    DSFMT(seed=None)

    Container for the SIMD-based Mersenne Twister pseudo RNG.

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**32-1], array of integers in
        [0, 2**32-1] or ``None`` (the default). If `seed` is ``None``,
        then ``DSFMT`` will try to read entropy from ``/dev/urandom``
        (or the Windows analog) if available to produce a 64-bit
        seed. If unavailable, a 64-bit hash of the time and process
        ID is used.

    Notes
    -----
    ``DSFMT`` directly provides generators for doubles, and unsigned 32 and 64-
    bit integers [1]_ . These are not firectly available and must be consumed
    via a ``RandomGenerator`` object.

    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator.

    **Parallel Features**

    ``DSFMT`` can be used in parallel applications by calling the method
    ``jump`` which advances the state as-if :math:`2^{128}` random numbers
    have been generated [2]_. This allows the original sequence to be split
    so that distinct segments can be used in each worker process.  All
    generators should be initialized with the same seed to ensure that the
    segments come from the same sequence.

    >>> from randomgen.entropy import random_entropy
    >>> from randomgen import RandomGenerator, DSFMT
    >>> seed = random_entropy()
    >>> rs = [RandomGenerator(DSFMT(seed)) for _ in range(10)]
    # Advance rs[i] by i jumps
    >>> for i in range(10):
    ...     rs[i].jump()

    **State and Seeding**

    The ``DSFMT`` state vector consists of a 384 element array of
    64-bit unsigned integers plus a single integer value between 0 and 382
    indicating  the current position within the main array. The implementation
    used here augments this with a 384 element array of doubles which are used
    to efficiently access the random numbers produced by the dSFMT generator.

    ``DSFMT`` is seeded using either a single 32-bit unsigned integer
    or a vector of 32-bit unsigned integers.  In either case, the input seed is
    used as an input (or inputs) for a hashing function, and the output of the
    hashing function is used as the initial state. Using a single 32-bit value
    for the seed can only initialize a small range of the possible initial
    state values.

    **Compatibility Guarantee**

    ``DSFMT`` does makes a guarantee that a fixed seed and will always
    produce the same results.

    References
    ----------
    .. [1] Mutsuo Saito and Makoto Matsumoto, "SIMD-oriented Fast Mersenne
           Twister: a 128-bit Pseudorandom Number Generator." Monte Carlo
           and Quasi-Monte Carlo Methods 2006, Springer, pp. 607--622, 2008.
    .. [2] Hiroshi Haramoto, Makoto Matsumoto, and Pierre L\'Ecuyer, "A Fast
           Jump Ahead Algorithm for Linear Recurrences in a Polynomial Space",
           Sequences and Their Applications - SETA, 290--298, 2008.
    """
    cdef dsfmt_state  *rng_state
    cdef brng_t *_brng
    cdef public object capsule
    cdef public object _cffi
    cdef public object _ctypes
    cdef public object _generator

    def __init__(self, seed=None):
        self.rng_state = <dsfmt_state *>malloc(sizeof(dsfmt_state))
        self.rng_state.state = <dsfmt_t *>PyArray_malloc_aligned(sizeof(dsfmt_t))
        self.rng_state.buffered_uniforms = <double *>PyArray_calloc_aligned(DSFMT_N64, sizeof(double))
        self.rng_state.buffer_loc = DSFMT_N64
        self._brng = <brng_t *>malloc(sizeof(brng_t))
        self.seed(seed)
        self._brng.state = <void *>self.rng_state
        self._brng.next_uint64 = &dsfmt_uint64
        self._brng.next_uint32 = &dsfmt_uint32
        self._brng.next_double = &dsfmt_double
        self._brng.next_raw = &dsfmt_raw
        cdef const char *name = "BasicRNG"
        self.capsule = PyCapsule_New(<void *>self._brng, name, NULL)

        self._cffi = None
        self._ctypes = None
        self._generator = None

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
        PyArray_free_aligned(self.rng_state.state)
        PyArray_free_aligned(self.rng_state.buffered_uniforms)
        free(self.rng_state)
        free(self._brng)

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
        seed(seed=None)

        Seed the generator.

        Parameters
        ----------
        seed : {None, int, array_like}, optional
            Random seed initializing the pseudo-random number generator.
            Can be an integer in [0, 2**32-1], array of integers in
            [0, 2**32-1] or ``None`` (the default). If `seed` is ``None``,
            then ``DSFMT`` will try to read entropy from ``/dev/urandom``
            (or the Windows analog) if available to produce a 64-bit
            seed. If unavailable, a 64-bit hash of the time and process
            ID is used.

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
                dsfmt_init_gen_rand(self.rng_state.state, seed)
            else:
                if hasattr(seed, 'squeeze'):
                    seed = seed.squeeze()
                idx = operator.index(seed)
                if idx > int(2**32 - 1) or idx < 0:
                    raise ValueError("Seed must be between 0 and 2**32 - 1")
                dsfmt_init_gen_rand(self.rng_state.state, seed)
        except TypeError:
            obj = np.asarray(seed).astype(np.int64, casting='safe').ravel()
            if ((obj > int(2**32 - 1)) | (obj < 0)).any():
                raise ValueError("Seed must be between 0 and 2**32 - 1")
            obj = obj.astype(np.uint32, casting='unsafe', order='C')
            dsfmt_init_by_array(self.rng_state.state,
                                <uint32_t *>obj.data,
                                np.PyArray_DIM(obj, 0))

    def jump(self, np.npy_intp iter=1):
        """
        jump(iter=1)

        Jumps the state as-if 2**128 random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the brng.

        Returns
        -------
        self : DSFMT
            PRNG jumped iter times
        """
        cdef np.npy_intp i
        for i in range(iter):
            dsfmt_jump(self.rng_state)
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

        cdef Py_ssize_t i, j, loc = 0
        cdef uint64_t[::1] state
        cdef double[::1] buffered_uniforms

        state = np.empty(2 *DSFMT_N_PLUS_1, dtype=np.uint64)
        for i in range(DSFMT_N_PLUS_1):
            for j in range(2):
                state[loc] = self.rng_state.state.status[i].u[j]
                loc += 1
        buffered_uniforms = np.empty(DSFMT_N64,dtype=np.double)
        for i in range(DSFMT_N64):
            buffered_uniforms[i] = self.rng_state.buffered_uniforms[i]
        return {'brng': self.__class__.__name__,
                'state': {'state':np.asarray(state),
                          'idx':self.rng_state.state.idx},
                'buffer_loc': self.rng_state.buffer_loc,
                'buffered_uniforms':np.asarray(buffered_uniforms)}

    @state.setter
    def state(self, value):
        cdef Py_ssize_t i, j, loc = 0
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        brng = value.get('brng', '')
        if brng != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        state = value['state']['state']
        for i in range(DSFMT_N_PLUS_1):
            for j in range(2):
                self.rng_state.state.status[i].u[j] = state[loc]
                loc += 1
        self.rng_state.state.idx = value['state']['idx']
        buffered_uniforms = value['buffered_uniforms']
        for i in range(DSFMT_N64):
            self.rng_state.buffered_uniforms[i] = buffered_uniforms[i]
        self.rng_state.buffer_loc = value['buffer_loc']

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
                         ctypes.cast(<uintptr_t>&dsfmt_uint64,
                                     ctypes.CFUNCTYPE(ctypes.c_uint64,
                                     ctypes.c_void_p)),
                         ctypes.cast(<uintptr_t>&dsfmt_uint32,
                                     ctypes.CFUNCTYPE(ctypes.c_uint32,
                                     ctypes.c_void_p)),
                         ctypes.cast(<uintptr_t>&dsfmt_double,
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
            Random generator used this instance as the basic RNG
        """
        if self._generator is None:
            from .generator import RandomGenerator
            self._generator = RandomGenerator(self)
        return self._generator