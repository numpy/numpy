import operator

from libc.stdlib cimport malloc, free
from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from common import interface
from common cimport *
from distributions cimport brng_t
import randomgen.pickle
from randomgen.entropy import random_entropy

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
    MT19937(seed=None)

    Container for the Mersenne Twister pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed used to initialize the pseudo-random number generator.  Can
        be any integer between 0 and 2**32 - 1 inclusive, an array (or other
        sequence) of such integers, or ``None`` (the default).  If `seed` is
        ``None``, then will attempt to read data from ``/dev/urandom`` 
        (or the Windows analog) if available or seed from the clock otherwise.

    Notes
    -----
    ``MT19937`` directly provides generators for doubles, and unsigned 32 and 64-
    bit integers [1]_ . These are not firectly available and must be consumed
    via a ``RandomGenerator`` object.

    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator.

    **State and Seeding**

    The ``MT19937`` state vector consists of a 768 element array of
    32-bit unsigned integers plus a single integer value between 0 and 768
    indicating  the current position within the main array.

    ``MT19937`` is seeded using either a single 32-bit unsigned integer
    or a vector of 32-bit unsigned integers.  In either case, the input seed is
    used as an input (or inputs) for a hashing function, and the output of the
    hashing function is used as the initial state. Using a single 32-bit value
    for the seed can only initialize a small range of the possible initial
    state values.

    **Compatibility Guarantee**

    ``MT19937`` make a compatibility guarantee. A fixed seed and a fixed 
    series of calls to ``MT19937`` methods will always produce the same 
    results up to roundoff error except when the values were incorrect. 
    Incorrect values will be fixed and the version in which the fix was 
    made will be noted in the relevant docstring.

    **Parallel Features**

    ``MT19937`` can be used in parallel applications by
    calling the method ``jump`` which advances the state as-if :math:`2^{128}`
    random numbers have been generated ([1]_, [2]_). This allows the original sequence to
    be split so that distinct segments can be used in each worker process.  All
    generators should be initialized with the same seed to ensure that the
    segments come from the same sequence.

    >>> from randomgen.entropy import random_entropy
    >>> from randomgen import RandomGenerator, MT19937
    >>> seed = random_entropy()
    >>> rs = [RandomGenerator(MT19937(seed) for _ in range(10)]
    # Advance rs[i] by i jumps
    >>> for i in range(10):
            rs[i].jump(i)

    References
    ----------
    .. [1] Hiroshi Haramoto, Makoto Matsumoto, and Pierre L\'Ecuyer, "A Fast
        Jump Ahead Algorithm for Linear Recurrences in a Polynomial Space",
        Sequences and Their Applications - SETA, 290--298, 2008.        
    .. [2] Hiroshi Haramoto, Makoto Matsumoto, Takuji Nishimura, François 
        Panneton, Pierre L\'Ecuyer, "Efficient Jump Ahead for F2-Linear
        Random Number Generators", INFORMS JOURNAL ON COMPUTING, Vol. 20, 
        No. 3, Summer 2008, pp. 385-390.

    """
    cdef mt19937_state *rng_state
    cdef brng_t *_brng
    cdef public object capsule
    cdef object _ctypes
    cdef object _cffi
    cdef object _generator

    def __init__(self, seed=None):
        self.rng_state = <mt19937_state *>malloc(sizeof(mt19937_state))
        self._brng = <brng_t *>malloc(sizeof(brng_t))
        self.seed(seed)

        self._brng.state = <void *>self.rng_state
        self._brng.next_uint64 = &mt19937_uint64
        self._brng.next_uint32 = &mt19937_uint32
        self._brng.next_double = &mt19937_double
        self._brng.next_raw = &mt19937_raw

        self._ctypes = None
        self._cffi = None
        self._generator = None

        cdef const char *name = "BasicRNG"
        self.capsule = PyCapsule_New(<void *>self._brng, name, NULL)

    def __dealloc__(self):
        free(self.rng_state)
        free(self._brng)

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        return (randomgen.pickle.__brng_ctor,
                (self.state['brng'],),
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
        seed(seed=None)

        Seed the generator.

        Parameters
        ----------
        seed : {None, int, array_like}, optional
            Random seed initializing the pseudo-random number generator.
            Can be an integer in [0, 2**32-1], array of integers in
            [0, 2**32-1] or ``None`` (the default). If `seed` is ``None``,
            then ``MT19937`` will try to read entropy from ``/dev/urandom``
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
            mt19937_jump(self.rng_state)
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
        key = np.zeros(624, dtype=np.uint32)
        for i in range(624):
            key[i] = self.rng_state.key[i]

        return {'brng': self.__class__.__name__,
                'state': {'key':key, 'pos': self.rng_state.pos}}

    @state.setter
    def state(self, value):
        if isinstance(value, tuple):
            if value[0] != 'MT19937' or len(value) not in (3,5):
                    raise ValueError('state is not a legacy MT19937 state')
            value ={'brng': 'MT19937',
                    'state':{'key': value[1], 'pos': value[2]}}


        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        brng = value.get('brng', '')
        if brng != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        key = value['state']['key']
        for i in range(624):
            self.rng_state.key[i] = key[i]
        self.rng_state.pos = value['state']['pos']

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
                         ctypes.cast(<uintptr_t>&mt19937_uint64,
                                     ctypes.CFUNCTYPE(ctypes.c_uint64,
                                     ctypes.c_void_p)),
                         ctypes.cast(<uintptr_t>&mt19937_uint32,
                                     ctypes.CFUNCTYPE(ctypes.c_uint32,
                                     ctypes.c_void_p)),
                         ctypes.cast(<uintptr_t>&mt19937_double,
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