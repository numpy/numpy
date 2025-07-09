#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, binding=True
import operator
import warnings
from collections.abc import Sequence

from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from cpython cimport (Py_INCREF, PyFloat_AsDouble)
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cimport cython
import numpy as np
cimport numpy as np
from numpy.lib.array_utils import normalize_axis_index

from .c_distributions cimport *
from libc cimport string
from libc.math cimport sqrt
from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int32_t, int64_t, INT64_MAX, SIZE_MAX)
from ._bounded_integers cimport (_rand_bool, _rand_int32, _rand_int64,
         _rand_int16, _rand_int8, _rand_uint64, _rand_uint32, _rand_uint16,
         _rand_uint8, _gen_mask)
from ._pcg64 import PCG64
from ._mt19937 import MT19937
from numpy.random cimport bitgen_t
from ._common cimport (POISSON_LAM_MAX, CONS_POSITIVE, CONS_NONE,
            CONS_NON_NEGATIVE, CONS_BOUNDED_0_1, CONS_BOUNDED_GT_0_1,
            CONS_BOUNDED_LT_0_1, CONS_GT_1, CONS_POSITIVE_NOT_NAN, CONS_POISSON,
            double_fill, cont, kahan_sum, cont_broadcast_3, float_fill, cont_f,
            check_array_constraint, check_constraint, disc, discrete_broadcast_iii,
            validate_output_shape
        )

cdef extern from "numpy/arrayobject.h":
    int PyArray_ResolveWritebackIfCopy(np.ndarray)
    int PyArray_FailUnlessWriteable(np.PyArrayObject *obj,
                                    const char *name) except -1
    object PyArray_FromArray(np.PyArrayObject *, np.PyArray_Descr *, int)

    enum:
        NPY_ARRAY_WRITEBACKIFCOPY

np.import_array()

cdef int64_t _safe_sum_nonneg_int64(size_t num_colors, int64_t *colors):
    """
    Sum the values in the array `colors`.

    Return -1 if an overflow occurs.
    The values in *colors are assumed to be nonnegative.
    """
    cdef size_t i
    cdef int64_t sum

    sum = 0
    for i in range(num_colors):
        if colors[i] > INT64_MAX - sum:
            return -1
        sum += colors[i]
    return sum


cdef inline void _shuffle_raw_wrap(bitgen_t *bitgen, np.npy_intp n,
                                   np.npy_intp first, np.npy_intp itemsize,
                                   np.npy_intp stride,
                                   char* data, char* buf) noexcept nogil:
    # We trick gcc into providing a specialized implementation for
    # the most common case, yielding a ~33% performance improvement.
    # Note that apparently, only one branch can ever be specialized.
    if itemsize == sizeof(np.npy_intp):
        _shuffle_raw(bitgen, n, first, sizeof(np.npy_intp), stride, data, buf)
    else:
        _shuffle_raw(bitgen, n, first, itemsize, stride, data, buf)


cdef inline void _shuffle_raw(bitgen_t *bitgen, np.npy_intp n,
                              np.npy_intp first, np.npy_intp itemsize,
                              np.npy_intp stride,
                              char* data, char* buf) noexcept nogil:
    """
    Parameters
    ----------
    bitgen
        Pointer to a bitgen_t instance.
    n
        Number of elements in data
    first
        First observation to shuffle.  Shuffles n-1,
        n-2, ..., first, so that when first=1 the entire
        array is shuffled
    itemsize
        Size in bytes of item
    stride
        Array stride
    data
        Location of data
    buf
        Location of buffer (itemsize)
    """
    cdef np.npy_intp i, j

    for i in reversed(range(first, n)):
        j = random_interval(bitgen, i)
        string.memcpy(buf, data + j * stride, itemsize)
        string.memcpy(data + j * stride, data + i * stride, itemsize)
        string.memcpy(data + i * stride, buf, itemsize)


cdef inline void _shuffle_int(bitgen_t *bitgen, np.npy_intp n,
                              np.npy_intp first, int64_t* data) noexcept nogil:
    """
    Parameters
    ----------
    bitgen
        Pointer to a bitgen_t instance.
    n
        Number of elements in data
    first
        First observation to shuffle.  Shuffles n-1,
        n-2, ..., first, so that when first=1 the entire
        array is shuffled
    data
        Location of data
    """
    cdef np.npy_intp i, j
    cdef int64_t temp
    for i in reversed(range(first, n)):
        j = random_bounded_uint64(bitgen, 0, i, 0, 0)
        temp = data[j]
        data[j] = data[i]
        data[i] = temp


cdef bint _check_bit_generator(object bitgen):
    """Check if an object satisfies the BitGenerator interface.
    """
    if not hasattr(bitgen, "capsule"):
        return False
    cdef const char *name = "BitGenerator"
    return PyCapsule_IsValid(bitgen.capsule, name)


cdef class Generator:
    """
    Generator(bit_generator)

    Container for the BitGenerators.

    `Generator` exposes a number of methods for generating random
    numbers drawn from a variety of probability distributions. In addition to
    the distribution-specific arguments, each method takes a keyword argument
    `size` that defaults to ``None``. If `size` is ``None``, then a single
    value is generated and returned. If `size` is an integer, then a 1-D
    array filled with generated values is returned. If `size` is a tuple,
    then an array with that shape is filled and returned.

    The function :func:`numpy.random.default_rng` will instantiate
    a `Generator` with numpy's default `BitGenerator`.

    **No Compatibility Guarantee**

    `Generator` does not provide a version compatibility guarantee. In
    particular, as better algorithms evolve the bit stream may change.

    Parameters
    ----------
    bit_generator : BitGenerator
        BitGenerator to use as the core generator.

    Notes
    -----
    The Python stdlib module :external+python:mod:`random` contains
    pseudo-random number generator with a number of methods that are similar
    to the ones available in `Generator`.
    It uses Mersenne Twister, and this bit generator can
    be accessed using `MT19937`. `Generator`, besides being
    NumPy-aware, has the advantage that it provides a much larger number
    of probability distributions to choose from.

    Examples
    --------
    >>> from numpy.random import Generator, PCG64
    >>> rng = Generator(PCG64())
    >>> rng.standard_normal()
    -0.203  # random

    See Also
    --------
    default_rng : Recommended constructor for `Generator`.
    """
    cdef public object _bit_generator
    cdef bitgen_t _bitgen
    cdef binomial_t _binomial
    cdef object lock
    _poisson_lam_max = POISSON_LAM_MAX

    def __init__(self, bit_generator):
        self._bit_generator = bit_generator

        capsule = bit_generator.capsule
        cdef const char *name = "BitGenerator"
        if not PyCapsule_IsValid(capsule, name):
            raise ValueError("Invalid bit generator. The bit generator must "
                             "be instantiated.")
        self._bitgen = (<bitgen_t *> PyCapsule_GetPointer(capsule, name))[0]
        self.lock = bit_generator.lock

    def __repr__(self):
        return f'{self} at 0x{id(self):X}'

    def __str__(self):
        return f'{self.__class__.__name__}({self.bit_generator.__class__.__name__})'

    # Pickling support:
    def __getstate__(self):
        return None

    def __setstate__(self, bit_gen):
        if isinstance(bit_gen, dict):
            # Legacy path
            # Prior to 2.0.x only the state of the underlying bit generator
            # was preserved and any seed sequence information was lost
            self.bit_generator.state = bit_gen

    def __reduce__(self):
        from ._pickle import __generator_ctor
        # Requirements of __generator_ctor are (bit_generator, )
        return __generator_ctor, (self._bit_generator, ), None

    @property
    def bit_generator(self):
        """
        Gets the bit generator instance used by the generator

        Returns
        -------
        bit_generator : BitGenerator
            The bit generator instance used by the generator
        """
        return self._bit_generator

    def spawn(self, int n_children):
        """
        spawn(n_children)

        Create new independent child generators.

        See :ref:`seedsequence-spawn` for additional notes on spawning
        children.

        .. versionadded:: 1.25.0

        Parameters
        ----------
        n_children : int

        Returns
        -------
        child_generators : list of Generators

        Raises
        ------
        TypeError
            When the underlying SeedSequence does not implement spawning.

        See Also
        --------
        random.BitGenerator.spawn, random.SeedSequence.spawn :
            Equivalent method on the bit generator and seed sequence.
        bit_generator :
            The bit generator instance used by the generator.

        Examples
        --------
        Starting from a seeded default generator:

        >>> # High quality entropy created with: f"0x{secrets.randbits(128):x}"
        >>> entropy = 0x3034c61a9ae04ff8cb62ab8ec2c4b501
        >>> rng = np.random.default_rng(entropy)

        Create two new generators for example for parallel execution:

        >>> child_rng1, child_rng2 = rng.spawn(2)

        Drawn numbers from each are independent but derived from the initial
        seeding entropy:

        >>> rng.uniform(), child_rng1.uniform(), child_rng2.uniform()
        (0.19029263503854454, 0.9475673279178444, 0.4702687338396767)

        It is safe to spawn additional children from the original ``rng`` or
        the children:

        >>> more_child_rngs = rng.spawn(20)
        >>> nested_spawn = child_rng1.spawn(20)

        """
        return [type(self)(g) for g in self._bit_generator.spawn(n_children)]

    def random(self, size=None, dtype=np.float64, out=None):
        """
        random(size=None, dtype=np.float64, out=None)

        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` use `uniform`
        or multiply the output of `random` by ``(b - a)`` and add ``a``::

            (b - a) * random() + a

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result, only `float64` and `float32` are supported.
            Byteorder must be native. The default value is np.float64.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is not None,
            it must have the same shape as the provided size and must match the type of
            the output values.

        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned).

        See Also
        --------
        uniform : Draw samples from the parameterized uniform distribution.

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.random()
        0.47108547995356098 # random
        >>> type(rng.random())
        <class 'float'>
        >>> rng.random((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428]) # random

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * rng.random((3, 2)) - 5
        array([[-3.99149989, -0.52338984], # random
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])

        """
        cdef double temp
        _dtype = np.dtype(dtype)
        if _dtype == np.float64:
            return double_fill(&random_standard_uniform_fill, &self._bitgen, size, self.lock, out)
        elif _dtype == np.float32:
            return float_fill(&random_standard_uniform_fill_f, &self._bitgen, size, self.lock, out)
        else:
            raise TypeError('Unsupported dtype %r for random' % _dtype)

    def beta(self, a, b, size=None):
        """
        beta(a, b, size=None)

        Draw samples from a Beta distribution.

        The Beta distribution is a special case of the Dirichlet distribution,
        and is related to the Gamma distribution. It has the probability
        distribution function

        .. math:: f(x; a,b) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}
                                                         (1 - x)^{\\beta - 1},

        where the normalization, B, is the beta function,

        .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}
                                     (1 - t)^{\\beta - 1} dt.

        It is often seen in Bayesian inference and order statistics.

        Parameters
        ----------
        a : float or array_like of floats
            Alpha, positive (>0).
        b : float or array_like of floats
            Beta, positive (>0).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``a`` and ``b`` are both scalars.
            Otherwise, ``np.broadcast(a, b).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized beta distribution.

        Examples
        -------- 
        The beta distribution has mean a/(a+b). If ``a == b`` and both 
        are > 1, the distribution is symmetric with mean 0.5.

        >>> rng = np.random.default_rng()
        >>> a, b, size = 2.0, 2.0, 10000
        >>> sample = rng.beta(a=a, b=b, size=size)
        >>> np.mean(sample)
        0.5047328775385895  # may vary
        
        Otherwise the distribution is skewed left or right according to
        whether ``a`` or ``b`` is greater. The distribution is mirror
        symmetric. See for example:
        
        >>> a, b, size = 2, 7, 10000
        >>> sample_left = rng.beta(a=a, b=b, size=size)
        >>> sample_right = rng.beta(a=b, b=a, size=size)
        >>> m_left, m_right = np.mean(sample_left), np.mean(sample_right)
        >>> print(m_left, m_right)
        0.2238596793678923 0.7774613834041182  # may vary
        >>> print(m_left - a/(a+b))
        0.001637457145670096  # may vary
        >>> print(m_right - b/(a+b))
        -0.0003163943736596009  # may vary

        Display the histogram of the two samples:
        
        >>> import matplotlib.pyplot as plt
        >>> plt.hist([sample_left, sample_right], 
        ...          50, density=True, histtype='bar')
        >>> plt.show()
        
        References
        ----------
        .. [1] Wikipedia, "Beta distribution",
               https://en.wikipedia.org/wiki/Beta_distribution

        """
        return cont(&random_beta, &self._bitgen, size, self.lock, 2,
                    a, 'a', CONS_POSITIVE,
                    b, 'b', CONS_POSITIVE,
                    0.0, '', CONS_NONE, None)

    def exponential(self, scale=1.0, size=None):
        """
        exponential(scale=1.0, size=None)

        Draw samples from an exponential distribution.

        Its probability density function is

        .. math:: f(x; \\frac{1}{\\beta}) = \\frac{1}{\\beta} \\exp(-\\frac{x}{\\beta}),

        for ``x > 0`` and 0 elsewhere. :math:`\\beta` is the scale parameter,
        which is the inverse of the rate parameter :math:`\\lambda = 1/\\beta`.
        The rate parameter is an alternative, widely used parameterization
        of the exponential distribution [3]_.

        The exponential distribution is a continuous analogue of the
        geometric distribution.  It describes many common situations, such as
        the size of raindrops measured over many rainstorms [1]_, or the time
        between page requests to Wikipedia [2]_.

        Parameters
        ----------
        scale : float or array_like of floats
            The scale parameter, :math:`\\beta = 1/\\lambda`. Must be
            non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``scale`` is a scalar.  Otherwise,
            ``np.array(scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized exponential distribution.

        Examples
        --------
        Assume a company has 10000 customer support agents and the time 
        between customer calls is exponentially distributed and that the 
        average time between customer calls is 4 minutes.

        >>> scale, size = 4, 10000
        >>> rng = np.random.default_rng()
        >>> time_between_calls = rng.exponential(scale=scale, size=size)

        What is the probability that a customer will call in the next 
        4 to 5 minutes? 
        
        >>> x = ((time_between_calls < 5).sum())/size
        >>> y = ((time_between_calls < 4).sum())/size
        >>> x - y
        0.08  # may vary

        The corresponding distribution can be visualized as follows:

        >>> import matplotlib.pyplot as plt
        >>> scale, size = 4, 10000
        >>> rng = np.random.default_rng()
        >>> sample = rng.exponential(scale=scale, size=size)
        >>> count, bins, _ = plt.hist(sample, 30, density=True)
        >>> plt.plot(bins, scale**(-1)*np.exp(-scale**-1*bins), linewidth=2, color='r')
        >>> plt.show()

        References
        ----------
        .. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
               Random Signal Principles", 4th ed, 2001, p. 57.
        .. [2] Wikipedia, "Poisson process",
               https://en.wikipedia.org/wiki/Poisson_process
        .. [3] Wikipedia, "Exponential distribution",
               https://en.wikipedia.org/wiki/Exponential_distribution

        """
        return cont(&random_exponential, &self._bitgen, size, self.lock, 1,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE,
                    None)

    def standard_exponential(self, size=None, dtype=np.float64, method='zig', out=None):
        """
        standard_exponential(size=None, dtype=np.float64, method='zig', out=None)

        Draw samples from the standard exponential distribution.

        `standard_exponential` is identical to the exponential distribution
        with a scale parameter of 1.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result, only `float64` and `float32` are supported.
            Byteorder must be native. The default value is np.float64.
        method : str, optional
            Either 'inv' or 'zig'. 'inv' uses the default inverse CDF method.
            'zig' uses the much faster Ziggurat method of Marsaglia and Tsang.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is not None,
            it must have the same shape as the provided size and must match the type of
            the output values.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        Examples
        --------
        Output a 3x8000 array:

        >>> rng = np.random.default_rng()
        >>> n = rng.standard_exponential((3, 8000))

        """
        _dtype = np.dtype(dtype)
        if _dtype == np.float64:
            if method == 'zig':
                return double_fill(&random_standard_exponential_fill, &self._bitgen, size, self.lock, out)
            else:
                return double_fill(&random_standard_exponential_inv_fill, &self._bitgen, size, self.lock, out)
        elif _dtype == np.float32:
            if method == 'zig':
                return float_fill(&random_standard_exponential_fill_f, &self._bitgen, size, self.lock, out)
            else:
                return float_fill(&random_standard_exponential_inv_fill_f, &self._bitgen, size, self.lock, out)
        else:
            raise TypeError('Unsupported dtype %r for standard_exponential'
                            % _dtype)

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        """
        integers(low, high=None, size=None, dtype=np.int64, endpoint=False)

        Return random integers from `low` (inclusive) to `high` (exclusive), or
        if endpoint=True, `low` (inclusive) to `high` (inclusive). Replaces
        `RandomState.randint` (with endpoint=False) and
        `RandomState.random_integers` (with endpoint=True)

        Return random integers from the "discrete uniform" distribution of
        the specified dtype. If `high` is None (the default), then results are
        from 0 to `low`.

        Parameters
        ----------
        low : int or array-like of ints
            Lowest (signed) integers to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is 0 and this value is
            used for `high`).
        high : int or array-like of ints, optional
            If provided, one above the largest (signed) integer to be drawn
            from the distribution (see above for behavior if ``high=None``).
            If array-like, must contain integer values
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result. Byteorder must be native.
            The default value is np.int64.
        endpoint : bool, optional
            If true, sample from the interval [low, high] instead of the
            default [low, high)
            Defaults to False

        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        Notes
        -----
        When using broadcasting with uint64 dtypes, the maximum value (2**64)
        cannot be represented as a standard integer type. The high array (or
        low if high is None) must have object dtype, e.g., array([2**64]).

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.integers(2, size=10)
        array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])  # random
        >>> rng.integers(1, size=10)
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        Generate a 2 x 4 array of ints between 0 and 4, inclusive:

        >>> rng.integers(5, size=(2, 4))
        array([[4, 0, 2, 1],
               [3, 2, 2, 0]])  # random

        Generate a 1 x 3 array with 3 different upper bounds

        >>> rng.integers(1, [3, 5, 10])
        array([2, 2, 9])  # random

        Generate a 1 by 3 array with 3 different lower bounds

        >>> rng.integers([1, 5, 7], 10)
        array([9, 8, 7])  # random

        Generate a 2 by 4 array using broadcasting with dtype of uint8

        >>> rng.integers([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)
        array([[ 8,  6,  9,  7],
               [ 1, 16,  9, 12]], dtype=uint8)  # random

        References
        ----------
        .. [1] Daniel Lemire., "Fast Random Integer Generation in an Interval",
               ACM Transactions on Modeling and Computer Simulation 29 (1), 2019,
               https://arxiv.org/abs/1805.10941.

        """
        if high is None:
            high = low
            low = 0

        _dtype = np.dtype(dtype)

        # Implementation detail: the old API used a masked method to generate
        # bounded uniform integers. Lemire's method is preferable since it is
        # faster. randomgen allows a choice, we will always use the faster one.
        cdef bint _masked = False

        if _dtype == np.int32:
            ret = _rand_int32(low, high, size, _masked, endpoint, &self._bitgen, self.lock)
        elif _dtype == np.int64:
            ret = _rand_int64(low, high, size, _masked, endpoint, &self._bitgen, self.lock)
        elif _dtype == np.int16:
            ret = _rand_int16(low, high, size, _masked, endpoint, &self._bitgen, self.lock)
        elif _dtype == np.int8:
            ret = _rand_int8(low, high, size, _masked, endpoint, &self._bitgen, self.lock)
        elif _dtype == np.uint64:
            ret = _rand_uint64(low, high, size, _masked, endpoint, &self._bitgen, self.lock)
        elif _dtype == np.uint32:
            ret = _rand_uint32(low, high, size, _masked, endpoint, &self._bitgen, self.lock)
        elif _dtype == np.uint16:
            ret = _rand_uint16(low, high, size, _masked, endpoint, &self._bitgen, self.lock)
        elif _dtype == np.uint8:
            ret = _rand_uint8(low, high, size, _masked, endpoint, &self._bitgen, self.lock)
        elif _dtype == np.bool:
            ret = _rand_bool(low, high, size, _masked, endpoint, &self._bitgen, self.lock)
        elif not _dtype.isnative:
            raise ValueError('Providing a dtype with a non-native byteorder '
                             'is not supported. If you require '
                             'platform-independent byteorder, call byteswap '
                             'when required.')
        else:
            raise TypeError('Unsupported dtype %r for integers' % _dtype)


        if size is None and (dtype is bool or dtype is int):
            if np.array(ret).shape == ():
                return dtype(ret)
        return ret

    def bytes(self, np.npy_intp length):
        """
        bytes(length)

        Return random bytes.

        Parameters
        ----------
        length : int
            Number of random bytes.

        Returns
        -------
        out : bytes
            String of length `length`.

        Notes
        -----
        This function generates random bytes from a discrete uniform 
        distribution. The generated bytes are independent from the CPU's 
        native endianness.
        
        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.bytes(10)
        b'\\xfeC\\x9b\\x86\\x17\\xf2\\xa1\\xafcp'  # random

        """
        cdef Py_ssize_t n_uint32 = ((length - 1) // 4 + 1)
        # Interpret the uint32s as little-endian to convert them to bytes
        # consistently.
        return self.integers(0, 4294967296, size=n_uint32,
                             dtype=np.uint32).astype('<u4').tobytes()[:length]

    @cython.wraparound(True)
    def choice(self, a, size=None, replace=True, p=None, axis=0, bint shuffle=True):
        """
        choice(a, size=None, replace=True, p=None, axis=0, shuffle=True)

        Generates a random sample from a given array

        Parameters
        ----------
        a : {array_like, int}
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated from np.arange(a).
        size : {int, tuple[int]}, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn from the 1-d `a`. If `a` has more
            than one dimension, the `size` shape will be inserted into the
            `axis` dimension, so the output ``ndim`` will be ``a.ndim - 1 +
            len(size)``. Default is None, in which case a single value is
            returned.
        replace : bool, optional
            Whether the sample is with or without replacement. Default is True,
            meaning that a value of ``a`` can be selected multiple times.
        p : 1-D array_like, optional
            The probabilities associated with each entry in a.
            If not given, the sample assumes a uniform distribution over all
            entries in ``a``.
        axis : int, optional
            The axis along which the selection is performed. The default, 0,
            selects by row.
        shuffle : bool, optional
            Whether the sample is shuffled when sampling without replacement.
            Default is True, False provides a speedup.

        Returns
        -------
        samples : single item or ndarray
            The generated random samples

        Raises
        ------
        ValueError
            If a is an int and less than zero, if p is not 1-dimensional, if
            a is array-like with a size 0, if p is not a vector of
            probabilities, if a and p have different lengths, or if
            replace=False and the sample size is greater than the population
            size.

        See Also
        --------
        integers, shuffle, permutation

        Notes
        -----
        Setting user-specified probabilities through ``p`` uses a more general but less
        efficient sampler than the default. The general sampler produces a different sample
        than the optimized sampler even if each element of ``p`` is 1 / len(a).

        ``p`` must sum to 1 when cast to ``float64``. To ensure this, you may wish
        to normalize using ``p = p / np.sum(p, dtype=float)``.

        When passing ``a`` as an integer type and ``size`` is not specified, the return
        type is a native Python ``int``.

        Examples
        --------
        Generate a uniform random sample from np.arange(5) of size 3:

        >>> rng = np.random.default_rng()
        >>> rng.choice(5, 3)
        array([0, 3, 4]) # random
        >>> #This is equivalent to rng.integers(0,5,3)

        Generate a non-uniform random sample from np.arange(5) of size 3:

        >>> rng.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        array([3, 3, 0]) # random

        Generate a uniform random sample from np.arange(5) of size 3 without
        replacement:

        >>> rng.choice(5, 3, replace=False)
        array([3,1,0]) # random
        >>> #This is equivalent to rng.permutation(np.arange(5))[:3]

        Generate a uniform random sample from a 2-D array along the first
        axis (the default), without replacement:

        >>> rng.choice([[0, 1, 2], [3, 4, 5], [6, 7, 8]], 2, replace=False)
        array([[3, 4, 5], # random
               [0, 1, 2]])

        Generate a non-uniform random sample from np.arange(5) of size
        3 without replacement:

        >>> rng.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
        array([2, 3, 0]) # random

        Any of the above can be repeated with an arbitrary array-like
        instead of just integers. For instance:

        >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
        >>> rng.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
        array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'], # random
              dtype='<U11')

        """

        cdef int64_t val, t, loc, size_i, pop_size_i
        cdef int64_t *idx_data
        cdef np.npy_intp j
        cdef uint64_t set_size, mask
        cdef uint64_t[::1] hash_set
        # Format and Verify input
        a_original = a
        a = np.asarray(a)
        if a.ndim == 0:
            try:
                # __index__ must return an integer by python rules.
                pop_size = operator.index(a.item())
            except TypeError as exc:
                raise ValueError("a must be a sequence or an integer, "
                                 f"not {type(a_original)}") from exc
            if pop_size <= 0 and np.prod(size) != 0:
                raise ValueError("a must be a positive integer unless no "
                                 "samples are taken")
        else:
            pop_size = a.shape[axis]
            if pop_size == 0 and np.prod(size) != 0:
                raise ValueError("a cannot be empty unless no samples are "
                                 "taken")

        if p is not None:
            d = len(p)

            atol = np.sqrt(np.finfo(np.float64).eps)
            if isinstance(p, np.ndarray):
                if np.issubdtype(p.dtype, np.floating):
                    atol = max(atol, np.sqrt(np.finfo(p.dtype).eps))

            p = <np.ndarray>np.PyArray_FROM_OTF(
                p, np.NPY_DOUBLE, np.NPY_ARRAY_ALIGNED | np.NPY_ARRAY_C_CONTIGUOUS)
            pix = <double*>np.PyArray_DATA(p)

            if p.ndim != 1:
                raise ValueError("p must be 1-dimensional")
            if p.size != pop_size:
                raise ValueError("a and p must have same size")
            p_sum = kahan_sum(pix, d)
            if np.isnan(p_sum):
                raise ValueError("Probabilities contain NaN")
            if np.logical_or.reduce(p < 0):
                raise ValueError("Probabilities are not non-negative")
            if abs(p_sum - 1.) > atol:
                raise ValueError("Probabilities do not sum to 1. See Notes "
                                 "section of docstring for more information.")

        # `shape == None` means `shape == ()`, but with scalar unpacking at the
        # end
        is_scalar = size is None
        if not is_scalar:
            shape = size
            size = np.prod(shape, dtype=np.intp)
        else:
            shape = ()
            size = 1

        # Actual sampling
        if replace:
            if p is not None:
                cdf = p.cumsum()
                cdf /= cdf[-1]
                uniform_samples = self.random(shape)
                idx = cdf.searchsorted(uniform_samples, side='right')
                # searchsorted returns a scalar
                idx = np.asarray(idx, dtype=np.int64)
            else:
                idx = self.integers(0, pop_size, size=shape, dtype=np.int64)
        else:
            if size > pop_size:
                raise ValueError("Cannot take a larger sample than "
                                 "population when replace is False")
            elif size < 0:
                raise ValueError("negative dimensions are not allowed")

            if p is not None:
                if np.count_nonzero(p > 0) < size:
                    raise ValueError("Fewer non-zero entries in p than size")
                n_uniq = 0
                p = p.copy()
                found = np.zeros(shape, dtype=np.int64)
                flat_found = found.ravel()
                while n_uniq < size:
                    x = self.random((size - n_uniq,))
                    if n_uniq > 0:
                        p[flat_found[0:n_uniq]] = 0
                    cdf = np.cumsum(p)
                    cdf /= cdf[-1]
                    new = cdf.searchsorted(x, side='right')
                    _, unique_indices = np.unique(new, return_index=True)
                    unique_indices.sort()
                    new = new.take(unique_indices)
                    flat_found[n_uniq:n_uniq + new.size] = new
                    n_uniq += new.size
                idx = found
            else:
                size_i = size
                pop_size_i = pop_size
                # This is a heuristic tuning. should be improvable
                if shuffle:
                    cutoff = 50
                else:
                    cutoff = 20
                if pop_size_i > 10000 and (size_i > (pop_size_i // cutoff)):
                    # Tail shuffle size elements
                    idx = np.PyArray_Arange(0, pop_size_i, 1, np.NPY_INT64)
                    idx_data = <int64_t*>(<np.ndarray>idx).data
                    with self.lock, nogil:
                        _shuffle_int(&self._bitgen, pop_size_i,
                                     max(pop_size_i - size_i, 1), idx_data)
                    # Copy to allow potentially large array backing idx to be gc
                    idx = idx[(pop_size - size):].copy()
                else:
                    # Floyd's algorithm
                    idx = np.empty(size, dtype=np.int64)
                    idx_data = <int64_t*>np.PyArray_DATA(<np.ndarray>idx)
                    # smallest power of 2 larger than 1.2 * size
                    set_size = <uint64_t>(1.2 * size_i)
                    mask = _gen_mask(set_size)
                    set_size = 1 + mask
                    hash_set = np.full(set_size, <uint64_t>-1, np.uint64)
                    with self.lock, cython.wraparound(False), nogil:
                        for j in range(pop_size_i - size_i, pop_size_i):
                            val = random_bounded_uint64(&self._bitgen, 0, j, 0, 0)
                            loc = val & mask
                            while hash_set[loc] != <uint64_t>-1 and hash_set[loc] != <uint64_t>val:
                                loc = (loc + 1) & mask
                            if hash_set[loc] == <uint64_t>-1: # then val not in hash_set
                                hash_set[loc] = val
                                idx_data[j - pop_size_i + size_i] = val
                            else: # we need to insert j instead
                                loc = j & mask
                                while hash_set[loc] != <uint64_t>-1:
                                    loc = (loc + 1) & mask
                                hash_set[loc] = j
                                idx_data[j - pop_size_i + size_i] = j
                        if shuffle:
                            _shuffle_int(&self._bitgen, size_i, 1, idx_data)
                idx.shape = shape

        if is_scalar and isinstance(idx, np.ndarray):
            # In most cases a scalar will have been made an array
            idx = idx.item(0)

        # Use samples as indices for a if a is array-like
        if a.ndim == 0:
            return idx

        if not is_scalar and idx.ndim == 0 and a.ndim == 1:
            # If size == () then the user requested a 0-d array as opposed to
            # a scalar object when size is None. However a[idx] is always a
            # scalar and not an array. So this makes sure the result is an
            # array, taking into account that np.array(item) may not work
            # for object arrays.
            res = np.empty((), dtype=a.dtype)
            res[()] = a[idx]
            return res

        # asarray downcasts on 32-bit platforms, always safe
        # no-op on 64-bit platforms
        return a.take(np.asarray(idx, dtype=np.intp), axis=axis)

    def uniform(self, low=0.0, high=1.0, size=None):
        """
        uniform(low=0.0, high=1.0, size=None)

        Draw samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval
        ``[low, high)`` (includes low, but excludes high).  In other words,
        any value within the given interval is equally likely to be drawn
        by `uniform`.

        Parameters
        ----------
        low : float or array_like of floats, optional
            Lower boundary of the output interval.  All values generated will be
            greater than or equal to low.  The default value is 0.
        high : float or array_like of floats
            Upper boundary of the output interval.  All values generated will be
            less than high.  The high limit may be included in the returned array of 
            floats due to floating-point rounding in the equation 
            ``low + (high-low) * random_sample()``.  high - low must be 
            non-negative.  The default value is 1.0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``low`` and ``high`` are both scalars.
            Otherwise, ``np.broadcast(low, high).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized uniform distribution.

        See Also
        --------
        integers : Discrete uniform distribution, yielding integers.
        random : Floats uniformly distributed over ``[0, 1)``.

        Notes
        -----
        The probability density function of the uniform distribution is

        .. math:: p(x) = \\frac{1}{b - a}

        anywhere within the interval ``[a, b)``, and zero elsewhere.

        When ``high`` == ``low``, values of ``low`` will be returned.

        Examples
        --------
        Draw samples from the distribution:

        >>> rng = np.random.default_rng()
        >>> s = rng.uniform(-1,0,1000)

        All values are within the given interval:

        >>> np.all(s >= -1)
        True
        >>> np.all(s < 0)
        True

        Display the histogram of the samples, along with the
        probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, 15, density=True)
        >>> plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        >>> plt.show()

        """
        cdef bint is_scalar = True
        cdef np.ndarray alow, ahigh, arange
        cdef double _low, _high, rng
        cdef object temp

        alow = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_DOUBLE, np.NPY_ARRAY_ALIGNED)
        ahigh = <np.ndarray>np.PyArray_FROM_OTF(high, np.NPY_DOUBLE, np.NPY_ARRAY_ALIGNED)

        if np.PyArray_NDIM(alow) == np.PyArray_NDIM(ahigh) == 0:
            _low = PyFloat_AsDouble(low)
            _high = PyFloat_AsDouble(high)
            rng = _high - _low
            if not np.isfinite(rng):
                raise OverflowError('high - low range exceeds valid bounds')

            return cont(&random_uniform, &self._bitgen, size, self.lock, 2,
                        _low, '', CONS_NONE,
                        rng, 'high - low', CONS_NON_NEGATIVE,
                        0.0, '', CONS_NONE,
                        None)

        temp = np.subtract(ahigh, alow)
        # needed to get around Pyrex's automatic reference-counting
        # rules because EnsureArray steals a reference
        Py_INCREF(temp)

        arange = <np.ndarray>np.PyArray_EnsureArray(temp)
        if not np.all(np.isfinite(arange)):
            raise OverflowError('Range exceeds valid bounds')
        return cont(&random_uniform, &self._bitgen, size, self.lock, 2,
                    alow, '', CONS_NONE,
                    arange, 'high - low', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    None)

    # Complicated, continuous distributions:
    def standard_normal(self, size=None, dtype=np.float64, out=None):
        """
        standard_normal(size=None, dtype=np.float64, out=None)

        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : dtype, optional
            Desired dtype of the result, only `float64` and `float32` are supported.
            Byteorder must be native. The default value is np.float64.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is not None,
            it must have the same shape as the provided size and must match the type of
            the output values.

        Returns
        -------
        out : float or ndarray
            A floating-point array of shape ``size`` of drawn samples, or a
            single sample if ``size`` was not specified.

        See Also
        --------
        normal :
            Equivalent function with additional ``loc`` and ``scale`` arguments
            for setting the mean and standard deviation.

        Notes
        -----
        For random samples from the normal distribution with mean ``mu`` and
        standard deviation ``sigma``, use one of::

            mu + sigma * rng.standard_normal(size=...)
            rng.normal(mu, sigma, size=...)

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.standard_normal()
        2.1923875335537315 # random

        >>> s = rng.standard_normal(8000)
        >>> s
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311,  # random
               -0.38672696, -0.4685006 ])                                # random
        >>> s.shape
        (8000,)
        >>> s = rng.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        Two-by-four array of samples from the normal distribution with
        mean 3 and standard deviation 2.5:

        >>> 3 + 2.5 * rng.standard_normal(size=(2, 4))
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random

        """
        _dtype = np.dtype(dtype)
        if _dtype == np.float64:
            return double_fill(&random_standard_normal_fill, &self._bitgen, size, self.lock, out)
        elif _dtype == np.float32:
            return float_fill(&random_standard_normal_fill_f, &self._bitgen, size, self.lock, out)
        else:
            raise TypeError('Unsupported dtype %r for standard_normal' % _dtype)

    def normal(self, loc=0.0, scale=1.0, size=None):
        """
        normal(loc=0.0, scale=1.0, size=None)

        Draw random samples from a normal (Gaussian) distribution.

        The probability density function of the normal distribution, first
        derived by De Moivre and 200 years later by both Gauss and Laplace
        independently [2]_, is often called the bell curve because of
        its characteristic shape (see the example below).

        The normal distributions occurs often in nature.  For example, it
        describes the commonly occurring distribution of samples influenced
        by a large number of tiny, random disturbances, each with its own
        unique distribution [2]_.

        Parameters
        ----------
        loc : float or array_like of floats
            Mean ("centre") of the distribution.
        scale : float or array_like of floats
            Standard deviation (spread or "width") of the distribution. Must be
            non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``loc`` and ``scale`` are both scalars.
            Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized normal distribution.

        See Also
        --------
        scipy.stats.norm : probability density function, distribution or
            cumulative density function, etc.

        Notes
        -----
        The probability density for the Gaussian distribution is

        .. math:: p(x) = \\frac{1}{\\sqrt{ 2 \\pi \\sigma^2 }}
                         e^{ - \\frac{ (x - \\mu)^2 } {2 \\sigma^2} },

        where :math:`\\mu` is the mean and :math:`\\sigma` the standard
        deviation. The square of the standard deviation, :math:`\\sigma^2`,
        is called the variance.

        The function has its peak at the mean, and its "spread" increases with
        the standard deviation (the function reaches 0.607 times its maximum at
        :math:`x + \\sigma` and :math:`x - \\sigma` [2]_).  This implies that
        :meth:`normal` is more likely to return samples lying close to the
        mean, rather than those far away.

        References
        ----------
        .. [1] Wikipedia, "Normal distribution",
               https://en.wikipedia.org/wiki/Normal_distribution
        .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
               Random Variables and Random Signal Principles", 4th ed., 2001,
               pp. 51, 51, 125.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, sigma = 0, 0.1 # mean and standard deviation
        >>> rng = np.random.default_rng()
        >>> s = rng.normal(mu, sigma, 1000)

        Verify the mean and the standard deviation:

        >>> abs(mu - np.mean(s))
        0.0  # may vary

        >>> abs(sigma - np.std(s, ddof=1))
        0.0  # may vary

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, 30, density=True)
        >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        Two-by-four array of samples from the normal distribution with
        mean 3 and standard deviation 2.5:

        >>> rng = np.random.default_rng()
        >>> rng.normal(3, 2.5, size=(2, 4))
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random

        """
        return cont(&random_normal, &self._bitgen, size, self.lock, 2,
                    loc, '', CONS_NONE,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    None)

    def standard_gamma(self, shape, size=None, dtype=np.float64, out=None):
        """
        standard_gamma(shape, size=None, dtype=np.float64, out=None)

        Draw samples from a standard Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,
        shape (sometimes designated "k") and scale=1.

        Parameters
        ----------
        shape : float or array_like of floats
            Parameter, must be non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``shape`` is a scalar.  Otherwise,
            ``np.array(shape).size`` samples are drawn.
        dtype : dtype, optional
            Desired dtype of the result, only `float64` and `float32` are supported.
            Byteorder must be native. The default value is np.float64.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is
            not None, it must have the same shape as the provided size and
            must match the type of the output values.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized standard gamma distribution.

        See Also
        --------
        scipy.stats.gamma : probability density function, distribution or
            cumulative density function, etc.

        Notes
        -----
        The probability density for the Gamma distribution is

        .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

        where :math:`k` is the shape and :math:`\\theta` the scale,
        and :math:`\\Gamma` is the Gamma function.

        The Gamma distribution is often used to model the times to failure of
        electronic components, and arises naturally in processes for which the
        waiting times between Poisson distributed events are relevant.

        References
        ----------
        .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
               Wolfram Web Resource.
               https://mathworld.wolfram.com/GammaDistribution.html
        .. [2] Wikipedia, "Gamma distribution",
               https://en.wikipedia.org/wiki/Gamma_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> shape, scale = 2., 1. # mean and width
        >>> rng = np.random.default_rng()
        >>> s = rng.standard_gamma(shape, 1000000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps  # doctest: +SKIP
        >>> count, bins, _ = plt.hist(s, 50, density=True)
        >>> y = bins**(shape-1) * ((np.exp(-bins/scale))/  # doctest: +SKIP
        ...                       (sps.gamma(shape) * scale**shape))
        >>> plt.plot(bins, y, linewidth=2, color='r')  # doctest: +SKIP
        >>> plt.show()

        """
        cdef void *func
        _dtype = np.dtype(dtype)
        if _dtype == np.float64:
            return cont(&random_standard_gamma, &self._bitgen, size, self.lock, 1,
                        shape, 'shape', CONS_NON_NEGATIVE,
                        0.0, '', CONS_NONE,
                        0.0, '', CONS_NONE,
                        out)
        if _dtype == np.float32:
            return cont_f(&random_standard_gamma_f, &self._bitgen, size, self.lock,
                          shape, 'shape', CONS_NON_NEGATIVE,
                          out)
        else:
            raise TypeError('Unsupported dtype %r for standard_gamma' % _dtype)

    def gamma(self, shape, scale=1.0, size=None):
        """
        gamma(shape, scale=1.0, size=None)

        Draw samples from a Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,
        `shape` (sometimes designated "k") and `scale` (sometimes designated
        "theta"), where both parameters are > 0.

        Parameters
        ----------
        shape : float or array_like of floats
            The shape of the gamma distribution. Must be non-negative.
        scale : float or array_like of floats, optional
            The scale of the gamma distribution. Must be non-negative.
            Default is equal to 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``shape`` and ``scale`` are both scalars.
            Otherwise, ``np.broadcast(shape, scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized gamma distribution.

        See Also
        --------
        scipy.stats.gamma : probability density function, distribution or
            cumulative density function, etc.

        Notes
        -----
        The probability density for the Gamma distribution is

        .. math:: p(x) = x^{k-1}\\frac{e^{-x/\\theta}}{\\theta^k\\Gamma(k)},

        where :math:`k` is the shape and :math:`\\theta` the scale,
        and :math:`\\Gamma` is the Gamma function.

        The Gamma distribution is often used to model the times to failure of
        electronic components, and arises naturally in processes for which the
        waiting times between Poisson distributed events are relevant.

        References
        ----------
        .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
               Wolfram Web Resource.
               https://mathworld.wolfram.com/GammaDistribution.html
        .. [2] Wikipedia, "Gamma distribution",
               https://en.wikipedia.org/wiki/Gamma_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
        >>> rng = np.random.default_rng()
        >>> s = rng.gamma(shape, scale, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps  # doctest: +SKIP
        >>> count, bins, _ = plt.hist(s, 50, density=True)
        >>> y = bins**(shape-1)*(np.exp(-bins/scale) /  # doctest: +SKIP
        ...                      (sps.gamma(shape)*scale**shape))
        >>> plt.plot(bins, y, linewidth=2, color='r')  # doctest: +SKIP
        >>> plt.show()

        """
        return cont(&random_gamma, &self._bitgen, size, self.lock, 2,
                    shape, 'shape', CONS_NON_NEGATIVE,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE, None)

    def f(self, dfnum, dfden, size=None):
        """
        f(dfnum, dfden, size=None)

        Draw samples from an F distribution.

        Samples are drawn from an F distribution with specified parameters,
        `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
        freedom in denominator), where both parameters must be greater than
        zero.

        The random variate of the F distribution (also known as the
        Fisher distribution) is a continuous probability distribution
        that arises in ANOVA tests, and is the ratio of two chi-square
        variates.

        Parameters
        ----------
        dfnum : float or array_like of floats
            Degrees of freedom in numerator, must be > 0.
        dfden : float or array_like of float
            Degrees of freedom in denominator, must be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``dfnum`` and ``dfden`` are both scalars.
            Otherwise, ``np.broadcast(dfnum, dfden).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Fisher distribution.

        See Also
        --------
        scipy.stats.f : probability density function, distribution or
            cumulative density function, etc.

        Notes
        -----
        The F statistic is used to compare in-group variances to between-group
        variances. Calculating the distribution depends on the sampling, and
        so it is a function of the respective degrees of freedom in the
        problem.  The variable `dfnum` is the number of samples minus one, the
        between-groups degrees of freedom, while `dfden` is the within-groups
        degrees of freedom, the sum of the number of samples in each group
        minus the number of groups.

        References
        ----------
        .. [1] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
               Fifth Edition, 2002.
        .. [2] Wikipedia, "F-distribution",
               https://en.wikipedia.org/wiki/F-distribution

        Examples
        --------
        An example from Glantz[1], pp 47-40:

        Two groups, children of diabetics (25 people) and children from people
        without diabetes (25 controls). Fasting blood glucose was measured,
        case group had a mean value of 86.1, controls had a mean value of
        82.2. Standard deviations were 2.09 and 2.49 respectively. Are these
        data consistent with the null hypothesis that the parents diabetic
        status does not affect their children's blood glucose levels?
        Calculating the F statistic from the data gives a value of 36.01.

        Draw samples from the distribution:

        >>> dfnum = 1. # between group degrees of freedom
        >>> dfden = 48. # within groups degrees of freedom
        >>> rng = np.random.default_rng()
        >>> s = rng.f(dfnum, dfden, 1000)

        The lower bound for the top 1% of the samples is :

        >>> np.sort(s)[-10]
        7.61988120985 # random

        So there is about a 1% chance that the F statistic will exceed 7.62,
        the measured value is 36, so the null hypothesis is rejected at the 1%
        level.
        
        The corresponding probability density function for ``n = 20`` 
        and ``m = 20`` is:
        
        >>> import matplotlib.pyplot as plt
        >>> from scipy import stats
        >>> dfnum, dfden, size = 20, 20, 10000
        >>> s = rng.f(dfnum=dfnum, dfden=dfden, size=size)
        >>> bins, density, _ = plt.hist(s, 30, density=True)
        >>> x = np.linspace(0, 5, 1000)
        >>> plt.plot(x, stats.f.pdf(x, dfnum, dfden))
        >>> plt.xlim([0, 5])
        >>> plt.show()
        
        """
        return cont(&random_f, &self._bitgen, size, self.lock, 2,
                    dfnum, 'dfnum', CONS_POSITIVE,
                    dfden, 'dfden', CONS_POSITIVE,
                    0.0, '', CONS_NONE, None)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        """
        noncentral_f(dfnum, dfden, nonc, size=None)

        Draw samples from the noncentral F distribution.

        Samples are drawn from an F distribution with specified parameters,
        `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
        freedom in denominator), where both parameters > 1.
        `nonc` is the non-centrality parameter.

        Parameters
        ----------
        dfnum : float or array_like of floats
            Numerator degrees of freedom, must be > 0.
        dfden : float or array_like of floats
            Denominator degrees of freedom, must be > 0.
        nonc : float or array_like of floats
            Non-centrality parameter, the sum of the squares of the numerator
            means, must be >= 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``dfnum``, ``dfden``, and ``nonc``
            are all scalars.  Otherwise, ``np.broadcast(dfnum, dfden, nonc).size``
            samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized noncentral Fisher distribution.

        Notes
        -----
        When calculating the power of an experiment (power = probability of
        rejecting the null hypothesis when a specific alternative is true) the
        non-central F statistic becomes important.  When the null hypothesis is
        true, the F statistic follows a central F distribution. When the null
        hypothesis is not true, then it follows a non-central F statistic.

        References
        ----------
        .. [1] Weisstein, Eric W. "Noncentral F-Distribution."
               From MathWorld--A Wolfram Web Resource.
               https://mathworld.wolfram.com/NoncentralF-Distribution.html
        .. [2] Wikipedia, "Noncentral F-distribution",
               https://en.wikipedia.org/wiki/Noncentral_F-distribution

        Examples
        --------
        In a study, testing for a specific alternative to the null hypothesis
        requires use of the Noncentral F distribution. We need to calculate the
        area in the tail of the distribution that exceeds the value of the F
        distribution for the null hypothesis.  We'll plot the two probability
        distributions for comparison.

        >>> rng = np.random.default_rng()
        >>> dfnum = 3 # between group deg of freedom
        >>> dfden = 20 # within groups degrees of freedom
        >>> nonc = 3.0
        >>> nc_vals = rng.noncentral_f(dfnum, dfden, nonc, 1000000)
        >>> NF = np.histogram(nc_vals, bins=50, density=True)
        >>> c_vals = rng.f(dfnum, dfden, 1000000)
        >>> F = np.histogram(c_vals, bins=50, density=True)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(F[1][1:], F[0])
        >>> plt.plot(NF[1][1:], NF[0])
        >>> plt.show()

        """
        return cont(&random_noncentral_f, &self._bitgen, size, self.lock, 3,
                    dfnum, 'dfnum', CONS_POSITIVE,
                    dfden, 'dfden', CONS_POSITIVE,
                    nonc, 'nonc', CONS_NON_NEGATIVE, None)

    def chisquare(self, df, size=None):
        """
        chisquare(df, size=None)

        Draw samples from a chi-square distribution.

        When `df` independent random variables, each with standard normal
        distributions (mean 0, variance 1), are squared and summed, the
        resulting distribution is chi-square (see Notes).  This distribution
        is often used in hypothesis testing.

        Parameters
        ----------
        df : float or array_like of floats
             Number of degrees of freedom, must be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``df`` is a scalar.  Otherwise,
            ``np.array(df).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized chi-square distribution.

        Raises
        ------
        ValueError
            When `df` <= 0 or when an inappropriate `size` (e.g. ``size=-1``)
            is given.

        Notes
        -----
        The variable obtained by summing the squares of `df` independent,
        standard normally distributed random variables:

        .. math:: Q = \\sum_{i=1}^{\\mathtt{df}} X^2_i

        is chi-square distributed, denoted

        .. math:: Q \\sim \\chi^2_k.

        The probability density function of the chi-squared distribution is

        .. math:: p(x) = \\frac{(1/2)^{k/2}}{\\Gamma(k/2)}
                         x^{k/2 - 1} e^{-x/2},

        where :math:`\\Gamma` is the gamma function,

        .. math:: \\Gamma(x) = \\int_0^{-\\infty} t^{x - 1} e^{-t} dt.

        References
        ----------
        .. [1] NIST "Engineering Statistics Handbook"
               https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.chisquare(2,4)
        array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272]) # random

        The distribution of a chi-square random variable
        with 20 degrees of freedom looks as follows:
        
        >>> import matplotlib.pyplot as plt
        >>> import scipy.stats as stats
        >>> s = rng.chisquare(20, 10000)
        >>> count, bins, _ = plt.hist(s, 30, density=True)
        >>> x = np.linspace(0, 60, 1000)
        >>> plt.plot(x, stats.chi2.pdf(x, df=20))
        >>> plt.xlim([0, 60])
        >>> plt.show()

        """
        return cont(&random_chisquare, &self._bitgen, size, self.lock, 1,
                    df, 'df', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)

    def noncentral_chisquare(self, df, nonc, size=None):
        """
        noncentral_chisquare(df, nonc, size=None)

        Draw samples from a noncentral chi-square distribution.

        The noncentral :math:`\\chi^2` distribution is a generalization of
        the :math:`\\chi^2` distribution.

        Parameters
        ----------
        df : float or array_like of floats
            Degrees of freedom, must be > 0.
        nonc : float or array_like of floats
            Non-centrality, must be non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``df`` and ``nonc`` are both scalars.
            Otherwise, ``np.broadcast(df, nonc).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized noncentral chi-square distribution.

        Notes
        -----
        The probability density function for the noncentral Chi-square
        distribution is

        .. math:: P(x;df,nonc) = \\sum^{\\infty}_{i=0}
                               \\frac{e^{-nonc/2}(nonc/2)^{i}}{i!}
                               P_{Y_{df+2i}}(x),

        where :math:`Y_{q}` is the Chi-square with q degrees of freedom.

        References
        ----------
        .. [1] Wikipedia, "Noncentral chi-squared distribution"
               https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram

        >>> rng = np.random.default_rng()
        >>> import matplotlib.pyplot as plt
        >>> values = plt.hist(rng.noncentral_chisquare(3, 20, 100000),
        ...                   bins=200, density=True)
        >>> plt.show()

        Draw values from a noncentral chisquare with very small noncentrality,
        and compare to a chisquare.

        >>> plt.figure()
        >>> values = plt.hist(rng.noncentral_chisquare(3, .0000001, 100000),
        ...                   bins=np.arange(0., 25, .1), density=True)
        >>> values2 = plt.hist(rng.chisquare(3, 100000),
        ...                    bins=np.arange(0., 25, .1), density=True)
        >>> plt.plot(values[1][0:-1], values[0]-values2[0], 'ob')
        >>> plt.show()

        Demonstrate how large values of non-centrality lead to a more symmetric
        distribution.

        >>> plt.figure()
        >>> values = plt.hist(rng.noncentral_chisquare(3, 20, 100000),
        ...                   bins=200, density=True)
        >>> plt.show()

        """
        return cont(&random_noncentral_chisquare, &self._bitgen, size, self.lock, 2,
                    df, 'df', CONS_POSITIVE,
                    nonc, 'nonc', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE, None)

    def standard_cauchy(self, size=None):
        """
        standard_cauchy(size=None)

        Draw samples from a standard Cauchy distribution with mode = 0.

        Also known as the Lorentz distribution.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        samples : ndarray or scalar
            The drawn samples.

        Notes
        -----
        The probability density function for the full Cauchy distribution is

        .. math:: P(x; x_0, \\gamma) = \\frac{1}{\\pi \\gamma \\bigl[ 1+
                  (\\frac{x-x_0}{\\gamma})^2 \\bigr] }

        and the Standard Cauchy distribution just sets :math:`x_0=0` and
        :math:`\\gamma=1`

        The Cauchy distribution arises in the solution to the driven harmonic
        oscillator problem, and also describes spectral line broadening. It
        also describes the distribution of values at which a line tilted at
        a random angle will cut the x axis.

        When studying hypothesis tests that assume normality, seeing how the
        tests perform on data from a Cauchy distribution is a good indicator of
        their sensitivity to a heavy-tailed distribution, since the Cauchy looks
        very much like a Gaussian distribution, but with heavier tails.

        References
        ----------
        .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "Cauchy
              Distribution",
              https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
        .. [2] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A
              Wolfram Web Resource.
              https://mathworld.wolfram.com/CauchyDistribution.html
        .. [3] Wikipedia, "Cauchy distribution"
              https://en.wikipedia.org/wiki/Cauchy_distribution

        Examples
        --------
        Draw samples and plot the distribution:

        >>> import matplotlib.pyplot as plt
        >>> rng = np.random.default_rng()
        >>> s = rng.standard_cauchy(1000000)
        >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
        >>> plt.hist(s, bins=100)
        >>> plt.show()

        """
        return cont(&random_standard_cauchy, &self._bitgen, size, self.lock, 0,
                    0.0, '', CONS_NONE, 0.0, '', CONS_NONE, 0.0, '', CONS_NONE, None)

    def standard_t(self, df, size=None):
        """
        standard_t(df, size=None)

        Draw samples from a standard Student's t distribution with `df` degrees
        of freedom.

        A special case of the hyperbolic distribution.  As `df` gets
        large, the result resembles that of the standard normal
        distribution (`standard_normal`).

        Parameters
        ----------
        df : float or array_like of floats
            Degrees of freedom, must be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``df`` is a scalar.  Otherwise,
            ``np.array(df).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized standard Student's t distribution.

        Notes
        -----
        The probability density function for the t distribution is

        .. math:: P(x, df) = \\frac{\\Gamma(\\frac{df+1}{2})}{\\sqrt{\\pi df}
                  \\Gamma(\\frac{df}{2})}\\Bigl( 1+\\frac{x^2}{df} \\Bigr)^{-(df+1)/2}

        The t test is based on an assumption that the data come from a
        Normal distribution. The t test provides a way to test whether
        the sample mean (that is the mean calculated from the data) is
        a good estimate of the true mean.

        The derivation of the t-distribution was first published in
        1908 by William Gosset while working for the Guinness Brewery
        in Dublin. Due to proprietary issues, he had to publish under
        a pseudonym, and so he used the name Student.

        References
        ----------
        .. [1] Dalgaard, Peter, "Introductory Statistics With R",
               Springer, 2002.
        .. [2] Wikipedia, "Student's t-distribution"
               https://en.wikipedia.org/wiki/Student's_t-distribution

        Examples
        --------
        From Dalgaard page 83 [1]_, suppose the daily energy intake for 11
        women in kilojoules (kJ) is:

        >>> intake = np.array([5260., 5470, 5640, 6180, 6390, 6515, 6805, 7515, \\
        ...                    7515, 8230, 8770])

        Does their energy intake deviate systematically from the recommended
        value of 7725 kJ? Our null hypothesis will be the absence of deviation,
        and the alternate hypothesis will be the presence of an effect that could be
        either positive or negative, hence making our test 2-tailed. 

        Because we are estimating the mean and we have N=11 values in our sample,
        we have N-1=10 degrees of freedom. We set our significance level to 95% and 
        compute the t statistic using the empirical mean and empirical standard 
        deviation of our intake. We use a ddof of 1 to base the computation of our 
        empirical standard deviation on an unbiased estimate of the variance (note:
        the final estimate is not unbiased due to the concave nature of the square 
        root).

        >>> np.mean(intake)
        6753.636363636364
        >>> intake.std(ddof=1)
        1142.1232221373727
        >>> t = (np.mean(intake)-7725)/(intake.std(ddof=1)/np.sqrt(len(intake)))
        >>> t
        -2.8207540608310198

        We draw 1000000 samples from Student's t distribution with the adequate
        degrees of freedom.

        >>> import matplotlib.pyplot as plt
        >>> rng = np.random.default_rng()
        >>> s = rng.standard_t(10, size=1000000)
        >>> h = plt.hist(s, bins=100, density=True)

        Does our t statistic land in one of the two critical regions found at 
        both tails of the distribution?

        >>> np.sum(np.abs(t) < np.abs(s)) / float(len(s))
        0.018318  #random < 0.05, statistic is in critical region

        The probability value for this 2-tailed test is about 1.83%, which is 
        lower than the 5% pre-determined significance threshold. 

        Therefore, the probability of observing values as extreme as our intake
        conditionally on the null hypothesis being true is too low, and we reject 
        the null hypothesis of no deviation. 

        """
        return cont(&random_standard_t, &self._bitgen, size, self.lock, 1,
                    df, 'df', CONS_POSITIVE,
                    0, '', CONS_NONE,
                    0, '', CONS_NONE,
                    None)

    def vonmises(self, mu, kappa, size=None):
        """
        vonmises(mu, kappa, size=None)

        Draw samples from a von Mises distribution.

        Samples are drawn from a von Mises distribution with specified mode
        (mu) and concentration (kappa), on the interval [-pi, pi].

        The von Mises distribution (also known as the circular normal
        distribution) is a continuous probability distribution on the unit
        circle.  It may be thought of as the circular analogue of the normal
        distribution.

        Parameters
        ----------
        mu : float or array_like of floats
            Mode ("center") of the distribution.
        kappa : float or array_like of floats
            Concentration of the distribution, has to be >=0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``mu`` and ``kappa`` are both scalars.
            Otherwise, ``np.broadcast(mu, kappa).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized von Mises distribution.

        See Also
        --------
        scipy.stats.vonmises : probability density function, distribution, or
            cumulative density function, etc.

        Notes
        -----
        The probability density for the von Mises distribution is

        .. math:: p(x) = \\frac{e^{\\kappa cos(x-\\mu)}}{2\\pi I_0(\\kappa)},

        where :math:`\\mu` is the mode and :math:`\\kappa` the concentration,
        and :math:`I_0(\\kappa)` is the modified Bessel function of order 0.

        The von Mises is named for Richard Edler von Mises, who was born in
        Austria-Hungary, in what is now the Ukraine.  He fled to the United
        States in 1939 and became a professor at Harvard.  He worked in
        probability theory, aerodynamics, fluid mechanics, and philosophy of
        science.

        References
        ----------
        .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
               Mathematical Functions with Formulas, Graphs, and Mathematical
               Tables, 9th printing," New York: Dover, 1972.
        .. [2] von Mises, R., "Mathematical Theory of Probability
               and Statistics", New York: Academic Press, 1964.

        Examples
        --------
        Draw samples from the distribution:

        >>> mu, kappa = 0.0, 4.0 # mean and concentration
        >>> rng = np.random.default_rng()
        >>> s = rng.vonmises(mu, kappa, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> from scipy.special import i0  # doctest: +SKIP
        >>> plt.hist(s, 50, density=True)
        >>> x = np.linspace(-np.pi, np.pi, num=51)
        >>> y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))  # doctest: +SKIP
        >>> plt.plot(x, y, linewidth=2, color='r')  # doctest: +SKIP
        >>> plt.show()

        """
        return cont(&random_vonmises, &self._bitgen, size, self.lock, 2,
                    mu, 'mu', CONS_NONE,
                    kappa, 'kappa', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE, None)

    def pareto(self, a, size=None):
        """
        pareto(a, size=None)

        Draw samples from a Pareto II (AKA Lomax) distribution with
        specified shape.

        Parameters
        ----------
        a : float or array_like of floats
            Shape of the distribution. Must be positive.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``a`` is a scalar.  Otherwise,
            ``np.array(a).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the Pareto II distribution.

        See Also
        --------
        scipy.stats.pareto : Pareto I distribution
        scipy.stats.lomax : Lomax (Pareto II) distribution
        scipy.stats.genpareto : Generalized Pareto distribution

        Notes
        -----
        The probability density for the Pareto II distribution is

        .. math:: p(x) = \\frac{a}{{x+1}^{a+1}} , x \ge 0

        where :math:`a > 0` is the shape.

        The Pareto II distribution is a shifted and scaled version of the
        Pareto I distribution, which can be found in `scipy.stats.pareto`.

        References
        ----------
        .. [1] Francis Hunt and Paul Johnson, On the Pareto Distribution of
               Sourceforge projects.
        .. [2] Pareto, V. (1896). Course of Political Economy. Lausanne.
        .. [3] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme
               Values, Birkhauser Verlag, Basel, pp 23-30.
        .. [4] Wikipedia, "Pareto distribution",
               https://en.wikipedia.org/wiki/Pareto_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 3.
        >>> rng = np.random.default_rng()
        >>> s = rng.pareto(a, 10000)

        Display the histogram of the samples, along with the probability
        density function:

        >>> import matplotlib.pyplot as plt
        >>> x = np.linspace(0, 3, 50)
        >>> pdf = a / (x+1)**(a+1)
        >>> plt.hist(s, bins=x, density=True, label='histogram')
        >>> plt.plot(x, pdf, linewidth=2, color='r', label='pdf')
        >>> plt.xlim(x.min(), x.max())
        >>> plt.legend()
        >>> plt.show()

        """
        return cont(&random_pareto, &self._bitgen, size, self.lock, 1,
                    a, 'a', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)

    def weibull(self, a, size=None):
        """
        weibull(a, size=None)

        Draw samples from a Weibull distribution.

        Draw samples from a 1-parameter Weibull distribution with the given
        shape parameter `a`.

        .. math:: X = (-ln(U))^{1/a}

        Here, U is drawn from the uniform distribution over (0,1].

        The more common 2-parameter Weibull, including a scale parameter
        :math:`\\lambda` is just :math:`X = \\lambda(-ln(U))^{1/a}`.

        Parameters
        ----------
        a : float or array_like of floats
            Shape parameter of the distribution.  Must be nonnegative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``a`` is a scalar.  Otherwise,
            ``np.array(a).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Weibull distribution.

        See Also
        --------
        scipy.stats.weibull_max
        scipy.stats.weibull_min
        scipy.stats.genextreme
        gumbel

        Notes
        -----
        The Weibull (or Type III asymptotic extreme value distribution
        for smallest values, SEV Type III, or Rosin-Rammler
        distribution) is one of a class of Generalized Extreme Value
        (GEV) distributions used in modeling extreme value problems.
        This class includes the Gumbel and Frechet distributions.

        The probability density for the Weibull distribution is

        .. math:: p(x) = \\frac{a}
                         {\\lambda}(\\frac{x}{\\lambda})^{a-1}e^{-(x/\\lambda)^a},

        where :math:`a` is the shape and :math:`\\lambda` the scale.

        The function has its peak (the mode) at
        :math:`\\lambda(\\frac{a-1}{a})^{1/a}`.

        When ``a = 1``, the Weibull distribution reduces to the exponential
        distribution.

        References
        ----------
        .. [1] Waloddi Weibull, Royal Technical University, Stockholm,
               1939 "A Statistical Theory Of The Strength Of Materials",
               Ingeniorsvetenskapsakademiens Handlingar Nr 151, 1939,
               Generalstabens Litografiska Anstalts Forlag, Stockholm.
        .. [2] Waloddi Weibull, "A Statistical Distribution Function of
               Wide Applicability", Journal Of Applied Mechanics ASME Paper
               1951.
        .. [3] Wikipedia, "Weibull distribution",
               https://en.wikipedia.org/wiki/Weibull_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> rng = np.random.default_rng()
        >>> a = 5. # shape
        >>> s = rng.weibull(a, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> def weibull(x, n, a):
        ...     return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)
        >>> count, bins, _ = plt.hist(rng.weibull(5., 1000))
        >>> x = np.linspace(0, 2, 1000)
        >>> bin_spacing = np.mean(np.diff(bins))
        >>> plt.plot(x, weibull(x, 1., 5.) * bin_spacing * s.size, label='Weibull PDF')
        >>> plt.legend()
        >>> plt.show()

        """
        return cont(&random_weibull, &self._bitgen, size, self.lock, 1,
                    a, 'a', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)

    def power(self, a, size=None):
        """
        power(a, size=None)

        Draws samples in [0, 1] from a power distribution with positive
        exponent a - 1.

        Also known as the power function distribution.

        Parameters
        ----------
        a : float or array_like of floats
            Parameter of the distribution. Must be non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``a`` is a scalar.  Otherwise,
            ``np.array(a).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized power distribution.

        Raises
        ------
        ValueError
            If a <= 0.

        Notes
        -----
        The probability density function is

        .. math:: P(x; a) = ax^{a-1}, 0 \\le x \\le 1, a>0.

        The power function distribution is just the inverse of the Pareto
        distribution. It may also be seen as a special case of the Beta
        distribution.

        It is used, for example, in modeling the over-reporting of insurance
        claims.

        References
        ----------
        .. [1] Christian Kleiber, Samuel Kotz, "Statistical size distributions
               in economics and actuarial sciences", Wiley, 2003.
        .. [2] Heckert, N. A. and Filliben, James J. "NIST Handbook 148:
               Dataplot Reference Manual, Volume 2: Let Subcommands and Library
               Functions", National Institute of Standards and Technology
               Handbook Series, June 2003.
               https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/powpdf.pdf

        Examples
        --------
        Draw samples from the distribution:

        >>> rng = np.random.default_rng()
        >>> a = 5. # shape
        >>> samples = 1000
        >>> s = rng.power(a, samples)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, bins=30)
        >>> x = np.linspace(0, 1, 100)
        >>> y = a*x**(a-1.)
        >>> normed_y = samples*np.diff(bins)[0]*y
        >>> plt.plot(x, normed_y)
        >>> plt.show()

        Compare the power function distribution to the inverse of the Pareto.

        >>> from scipy import stats  # doctest: +SKIP
        >>> rvs = rng.power(5, 1000000)
        >>> rvsp = rng.pareto(5, 1000000)
        >>> xx = np.linspace(0,1,100)
        >>> powpdf = stats.powerlaw.pdf(xx,5)  # doctest: +SKIP

        >>> plt.figure()
        >>> plt.hist(rvs, bins=50, density=True)
        >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
        >>> plt.title('power(5)')

        >>> plt.figure()
        >>> plt.hist(1./(1.+rvsp), bins=50, density=True)
        >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
        >>> plt.title('inverse of 1 + Generator.pareto(5)')

        >>> plt.figure()
        >>> plt.hist(1./(1.+rvsp), bins=50, density=True)
        >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
        >>> plt.title('inverse of stats.pareto(5)')

        """
        return cont(&random_power, &self._bitgen, size, self.lock, 1,
                    a, 'a', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        """
        laplace(loc=0.0, scale=1.0, size=None)

        Draw samples from the Laplace or double exponential distribution with
        specified location (or mean) and scale (decay).

        The Laplace distribution is similar to the Gaussian/normal distribution,
        but is sharper at the peak and has fatter tails. It represents the
        difference between two independent, identically distributed exponential
        random variables.

        Parameters
        ----------
        loc : float or array_like of floats, optional
            The position, :math:`\\mu`, of the distribution peak. Default is 0.
        scale : float or array_like of floats, optional
            :math:`\\lambda`, the exponential decay. Default is 1. Must be non-
            negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``loc`` and ``scale`` are both scalars.
            Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Laplace distribution.

        Notes
        -----
        It has the probability density function

        .. math:: f(x; \\mu, \\lambda) = \\frac{1}{2\\lambda}
                                       \\exp\\left(-\\frac{|x - \\mu|}{\\lambda}\\right).

        The first law of Laplace, from 1774, states that the frequency
        of an error can be expressed as an exponential function of the
        absolute magnitude of the error, which leads to the Laplace
        distribution. For many problems in economics and health
        sciences, this distribution seems to model the data better
        than the standard Gaussian distribution.

        References
        ----------
        .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
               Mathematical Functions with Formulas, Graphs, and Mathematical
               Tables, 9th printing," New York: Dover, 1972.
        .. [2] Kotz, Samuel, et. al. "The Laplace Distribution and
               Generalizations, " Birkhauser, 2001.
        .. [3] Weisstein, Eric W. "Laplace Distribution."
               From MathWorld--A Wolfram Web Resource.
               https://mathworld.wolfram.com/LaplaceDistribution.html
        .. [4] Wikipedia, "Laplace distribution",
               https://en.wikipedia.org/wiki/Laplace_distribution

        Examples
        --------
        Draw samples from the distribution

        >>> loc, scale = 0., 1.
        >>> rng = np.random.default_rng()
        >>> s = rng.laplace(loc, scale, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, 30, density=True)
        >>> x = np.arange(-8., 8., .01)
        >>> pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
        >>> plt.plot(x, pdf)

        Plot Gaussian for comparison:

        >>> g = (1/(scale * np.sqrt(2 * np.pi)) *
        ...      np.exp(-(x - loc)**2 / (2 * scale**2)))
        >>> plt.plot(x,g)

        """
        return cont(&random_laplace, &self._bitgen, size, self.lock, 2,
                    loc, 'loc', CONS_NONE,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE, None)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        """
        gumbel(loc=0.0, scale=1.0, size=None)

        Draw samples from a Gumbel distribution.

        Draw samples from a Gumbel distribution with specified location and
        scale.  For more information on the Gumbel distribution, see
        Notes and References below.

        Parameters
        ----------
        loc : float or array_like of floats, optional
            The location of the mode of the distribution. Default is 0.
        scale : float or array_like of floats, optional
            The scale parameter of the distribution. Default is 1. Must be non-
            negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``loc`` and ``scale`` are both scalars.
            Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Gumbel distribution.

        See Also
        --------
        scipy.stats.gumbel_l
        scipy.stats.gumbel_r
        scipy.stats.genextreme
        weibull

        Notes
        -----
        The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme
        Value Type I) distribution is one of a class of Generalized Extreme
        Value (GEV) distributions used in modeling extreme value problems.
        The Gumbel is a special case of the Extreme Value Type I distribution
        for maximums from distributions with "exponential-like" tails.

        The probability density for the Gumbel distribution is

        .. math:: p(x) = \\frac{e^{-(x - \\mu)/ \\beta}}{\\beta} e^{ -e^{-(x - \\mu)/
                  \\beta}},

        where :math:`\\mu` is the mode, a location parameter, and
        :math:`\\beta` is the scale parameter.

        The Gumbel (named for German mathematician Emil Julius Gumbel) was used
        very early in the hydrology literature, for modeling the occurrence of
        flood events. It is also used for modeling maximum wind speed and
        rainfall rates.  It is a "fat-tailed" distribution - the probability of
        an event in the tail of the distribution is larger than if one used a
        Gaussian, hence the surprisingly frequent occurrence of 100-year
        floods. Floods were initially modeled as a Gaussian process, which
        underestimated the frequency of extreme events.

        It is one of a class of extreme value distributions, the Generalized
        Extreme Value (GEV) distributions, which also includes the Weibull and
        Frechet.

        The function has a mean of :math:`\\mu + 0.57721\\beta` and a variance
        of :math:`\\frac{\\pi^2}{6}\\beta^2`.

        References
        ----------
        .. [1] Gumbel, E. J., "Statistics of Extremes,"
               New York: Columbia University Press, 1958.
        .. [2] Reiss, R.-D. and Thomas, M., "Statistical Analysis of Extreme
               Values from Insurance, Finance, Hydrology and Other Fields,"
               Basel: Birkhauser Verlag, 2001.

        Examples
        --------
        Draw samples from the distribution:

        >>> rng = np.random.default_rng()
        >>> mu, beta = 0, 0.1 # location and scale
        >>> s = rng.gumbel(mu, beta, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, 30, density=True)
        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
        ...          * np.exp( -np.exp( -(bins - mu) /beta) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        Show how an extreme value distribution can arise from a Gaussian process
        and compare to a Gaussian:

        >>> means = []
        >>> maxima = []
        >>> for i in range(0,1000) :
        ...    a = rng.normal(mu, beta, 1000)
        ...    means.append(a.mean())
        ...    maxima.append(a.max())
        >>> count, bins, _ = plt.hist(maxima, 30, density=True)
        >>> beta = np.std(maxima) * np.sqrt(6) / np.pi
        >>> mu = np.mean(maxima) - 0.57721*beta
        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
        ...          * np.exp(-np.exp(-(bins - mu)/beta)),
        ...          linewidth=2, color='r')
        >>> plt.plot(bins, 1/(beta * np.sqrt(2 * np.pi))
        ...          * np.exp(-(bins - mu)**2 / (2 * beta**2)),
        ...          linewidth=2, color='g')
        >>> plt.show()

        """
        return cont(&random_gumbel, &self._bitgen, size, self.lock, 2,
                    loc, 'loc', CONS_NONE,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE, None)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        """
        logistic(loc=0.0, scale=1.0, size=None)

        Draw samples from a logistic distribution.

        Samples are drawn from a logistic distribution with specified
        parameters, loc (location or mean, also median), and scale (>0).

        Parameters
        ----------
        loc : float or array_like of floats, optional
            Parameter of the distribution. Default is 0.
        scale : float or array_like of floats, optional
            Parameter of the distribution. Must be non-negative.
            Default is 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``loc`` and ``scale`` are both scalars.
            Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized logistic distribution.

        See Also
        --------
        scipy.stats.logistic : probability density function, distribution or
            cumulative density function, etc.

        Notes
        -----
        The probability density for the Logistic distribution is

        .. math:: P(x) = \\frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2},

        where :math:`\\mu` = location and :math:`s` = scale.

        The Logistic distribution is used in Extreme Value problems where it
        can act as a mixture of Gumbel distributions, in Epidemiology, and by
        the World Chess Federation (FIDE) where it is used in the Elo ranking
        system, assuming the performance of each player is a logistically
        distributed random variable.

        References
        ----------
        .. [1] Reiss, R.-D. and Thomas M. (2001), "Statistical Analysis of
               Extreme Values, from Insurance, Finance, Hydrology and Other
               Fields," Birkhauser Verlag, Basel, pp 132-133.
        .. [2] Weisstein, Eric W. "Logistic Distribution." From
               MathWorld--A Wolfram Web Resource.
               https://mathworld.wolfram.com/LogisticDistribution.html
        .. [3] Wikipedia, "Logistic-distribution",
               https://en.wikipedia.org/wiki/Logistic_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> loc, scale = 10, 1
        >>> rng = np.random.default_rng()
        >>> s = rng.logistic(loc, scale, 10000)
        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, bins=50, label='Sampled data')

        #   plot sampled data against the exact distribution

        >>> def logistic(x, loc, scale):
        ...     return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)
        >>> logistic_values  = logistic(bins, loc, scale)
        >>> bin_spacing = np.mean(np.diff(bins))
        >>> plt.plot(bins, logistic_values  * bin_spacing * s.size, label='Logistic PDF')
        >>> plt.legend()
        >>> plt.show()

        """
        return cont(&random_logistic, &self._bitgen, size, self.lock, 2,
                    loc, 'loc', CONS_NONE,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE, None)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        """
        lognormal(mean=0.0, sigma=1.0, size=None)

        Draw samples from a log-normal distribution.

        Draw samples from a log-normal distribution with specified mean,
        standard deviation, and array shape.  Note that the mean and standard
        deviation are not the values for the distribution itself, but of the
        underlying normal distribution it is derived from.

        Parameters
        ----------
        mean : float or array_like of floats, optional
            Mean value of the underlying normal distribution. Default is 0.
        sigma : float or array_like of floats, optional
            Standard deviation of the underlying normal distribution. Must be
            non-negative. Default is 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``mean`` and ``sigma`` are both scalars.
            Otherwise, ``np.broadcast(mean, sigma).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized log-normal distribution.

        See Also
        --------
        scipy.stats.lognorm : probability density function, distribution,
            cumulative density function, etc.

        Notes
        -----
        A variable `x` has a log-normal distribution if `log(x)` is normally
        distributed.  The probability density function for the log-normal
        distribution is:

        .. math:: p(x) = \\frac{1}{\\sigma x \\sqrt{2\\pi}}
                         e^{(-\\frac{(ln(x)-\\mu)^2}{2\\sigma^2})}

        where :math:`\\mu` is the mean and :math:`\\sigma` is the standard
        deviation of the normally distributed logarithm of the variable.
        A log-normal distribution results if a random variable is the *product*
        of a large number of independent, identically-distributed variables in
        the same way that a normal distribution results if the variable is the
        *sum* of a large number of independent, identically-distributed
        variables.

        References
        ----------
        .. [1] Limpert, E., Stahel, W. A., and Abbt, M., "Log-normal
               Distributions across the Sciences: Keys and Clues,"
               BioScience, Vol. 51, No. 5, May, 2001.
               https://stat.ethz.ch/~stahel/lognormal/bioscience.pdf
        .. [2] Reiss, R.D. and Thomas, M., "Statistical Analysis of Extreme
               Values," Basel: Birkhauser Verlag, 2001, pp. 31-32.

        Examples
        --------
        Draw samples from the distribution:

        >>> rng = np.random.default_rng()
        >>> mu, sigma = 3., 1. # mean and standard deviation
        >>> s = rng.lognormal(mu, sigma, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, 100, density=True, align='mid')

        >>> x = np.linspace(min(bins), max(bins), 10000)
        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        ...        / (x * sigma * np.sqrt(2 * np.pi)))

        >>> plt.plot(x, pdf, linewidth=2, color='r')
        >>> plt.axis('tight')
        >>> plt.show()

        Demonstrate that taking the products of random samples from a uniform
        distribution can be fit well by a log-normal probability density
        function.

        >>> # Generate a thousand samples: each is the product of 100 random
        >>> # values, drawn from a normal distribution.
        >>> rng = rng
        >>> b = []
        >>> for i in range(1000):
        ...    a = 10. + rng.standard_normal(100)
        ...    b.append(np.prod(a))

        >>> b = np.array(b) / np.min(b) # scale values to be positive
        >>> count, bins, _ = plt.hist(b, 100, density=True, align='mid')
        >>> sigma = np.std(np.log(b))
        >>> mu = np.mean(np.log(b))

        >>> x = np.linspace(min(bins), max(bins), 10000)
        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        ...        / (x * sigma * np.sqrt(2 * np.pi)))

        >>> plt.plot(x, pdf, color='r', linewidth=2)
        >>> plt.show()

        """
        return cont(&random_lognormal, &self._bitgen, size, self.lock, 2,
                    mean, 'mean', CONS_NONE,
                    sigma, 'sigma', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE, None)

    def rayleigh(self, scale=1.0, size=None):
        """
        rayleigh(scale=1.0, size=None)

        Draw samples from a Rayleigh distribution.

        The :math:`\\chi` and Weibull distributions are generalizations of the
        Rayleigh.

        Parameters
        ----------
        scale : float or array_like of floats, optional
            Scale, also equals the mode. Must be non-negative. Default is 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``scale`` is a scalar.  Otherwise,
            ``np.array(scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Rayleigh distribution.

        Notes
        -----
        The probability density function for the Rayleigh distribution is

        .. math:: P(x;scale) = \\frac{x}{scale^2}e^{\\frac{-x^2}{2 \\cdotp scale^2}}

        The Rayleigh distribution would arise, for example, if the East
        and North components of the wind velocity had identical zero-mean
        Gaussian distributions.  Then the wind speed would have a Rayleigh
        distribution.

        References
        ----------
        .. [1] Brighton Webs Ltd., "Rayleigh Distribution,"
               https://web.archive.org/web/20090514091424/http://brighton-webs.co.uk:80/distributions/rayleigh.asp
        .. [2] Wikipedia, "Rayleigh distribution"
               https://en.wikipedia.org/wiki/Rayleigh_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram

        >>> from matplotlib.pyplot import hist
        >>> rng = np.random.default_rng()
        >>> values = hist(rng.rayleigh(3, 100000), bins=200, density=True)

        Wave heights tend to follow a Rayleigh distribution. If the mean wave
        height is 1 meter, what fraction of waves are likely to be larger than 3
        meters?

        >>> meanvalue = 1
        >>> modevalue = np.sqrt(2 / np.pi) * meanvalue
        >>> s = rng.rayleigh(modevalue, 1000000)

        The percentage of waves larger than 3 meters is:

        >>> 100.*sum(s>3)/1000000.
        0.087300000000000003 # random

        """
        return cont(&random_rayleigh, &self._bitgen, size, self.lock, 1,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)

    def wald(self, mean, scale, size=None):
        """
        wald(mean, scale, size=None)

        Draw samples from a Wald, or inverse Gaussian, distribution.

        As the scale approaches infinity, the distribution becomes more like a
        Gaussian. Some references claim that the Wald is an inverse Gaussian
        with mean equal to 1, but this is by no means universal.

        The inverse Gaussian distribution was first studied in relationship to
        Brownian motion. In 1956 M.C.K. Tweedie used the name inverse Gaussian
        because there is an inverse relationship between the time to cover a
        unit distance and distance covered in unit time.

        Parameters
        ----------
        mean : float or array_like of floats
            Distribution mean, must be > 0.
        scale : float or array_like of floats
            Scale parameter, must be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``mean`` and ``scale`` are both scalars.
            Otherwise, ``np.broadcast(mean, scale).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Wald distribution.

        Notes
        -----
        The probability density function for the Wald distribution is

        .. math:: P(x;mean,scale) = \\sqrt{\\frac{scale}{2\\pi x^3}}e^
                                    \\frac{-scale(x-mean)^2}{2\\cdotp mean^2x}

        As noted above the inverse Gaussian distribution first arise
        from attempts to model Brownian motion. It is also a
        competitor to the Weibull for use in reliability modeling and
        modeling stock returns and interest rate processes.

        References
        ----------
        .. [1] Brighton Webs Ltd., Wald Distribution,
               https://web.archive.org/web/20090423014010/http://www.brighton-webs.co.uk:80/distributions/wald.asp
        .. [2] Chhikara, Raj S., and Folks, J. Leroy, "The Inverse Gaussian
               Distribution: Theory : Methodology, and Applications", CRC Press,
               1988.
        .. [3] Wikipedia, "Inverse Gaussian distribution"
               https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram:

        >>> import matplotlib.pyplot as plt
        >>> rng = np.random.default_rng()
        >>> h = plt.hist(rng.wald(3, 2, 100000), bins=200, density=True)
        >>> plt.show()

        """
        return cont(&random_wald, &self._bitgen, size, self.lock, 2,
                    mean, 'mean', CONS_POSITIVE,
                    scale, 'scale', CONS_POSITIVE,
                    0.0, '', CONS_NONE, None)

    def triangular(self, left, mode, right, size=None):
        """
        triangular(left, mode, right, size=None)

        Draw samples from the triangular distribution over the
        interval ``[left, right]``.

        The triangular distribution is a continuous probability
        distribution with lower limit left, peak at mode, and upper
        limit right. Unlike the other distributions, these parameters
        directly define the shape of the pdf.

        Parameters
        ----------
        left : float or array_like of floats
            Lower limit.
        mode : float or array_like of floats
            The value where the peak of the distribution occurs.
            The value must fulfill the condition ``left <= mode <= right``.
        right : float or array_like of floats
            Upper limit, must be larger than `left`.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``left``, ``mode``, and ``right``
            are all scalars.  Otherwise, ``np.broadcast(left, mode, right).size``
            samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized triangular distribution.

        Notes
        -----
        The probability density function for the triangular distribution is

        .. math:: P(x;l, m, r) = \\begin{cases}
                  \\frac{2(x-l)}{(r-l)(m-l)}& \\text{for $l \\leq x \\leq m$},\\\\
                  \\frac{2(r-x)}{(r-l)(r-m)}& \\text{for $m \\leq x \\leq r$},\\\\
                  0& \\text{otherwise}.
                  \\end{cases}

        The triangular distribution is often used in ill-defined
        problems where the underlying distribution is not known, but
        some knowledge of the limits and mode exists. Often it is used
        in simulations.

        References
        ----------
        .. [1] Wikipedia, "Triangular distribution"
               https://en.wikipedia.org/wiki/Triangular_distribution

        Examples
        --------
        Draw values from the distribution and plot the histogram:

        >>> import matplotlib.pyplot as plt
        >>> rng = np.random.default_rng()
        >>> h = plt.hist(rng.triangular(-3, 0, 8, 100000), bins=200,
        ...              density=True)
        >>> plt.show()

        """
        cdef bint is_scalar = True
        cdef double fleft, fmode, fright
        cdef np.ndarray oleft, omode, oright

        oleft = <np.ndarray>np.PyArray_FROM_OTF(left, np.NPY_DOUBLE, np.NPY_ARRAY_ALIGNED)
        omode = <np.ndarray>np.PyArray_FROM_OTF(mode, np.NPY_DOUBLE, np.NPY_ARRAY_ALIGNED)
        oright = <np.ndarray>np.PyArray_FROM_OTF(right, np.NPY_DOUBLE, np.NPY_ARRAY_ALIGNED)

        if np.PyArray_NDIM(oleft) == np.PyArray_NDIM(omode) == np.PyArray_NDIM(oright) == 0:
            fleft = PyFloat_AsDouble(left)
            fright = PyFloat_AsDouble(right)
            fmode = PyFloat_AsDouble(mode)

            if fleft > fmode:
                raise ValueError("left > mode")
            if fmode > fright:
                raise ValueError("mode > right")
            if fleft == fright:
                raise ValueError("left == right")
            return cont(&random_triangular, &self._bitgen, size, self.lock, 3,
                        fleft, '', CONS_NONE,
                        fmode, '', CONS_NONE,
                        fright, '', CONS_NONE, None)

        if np.any(np.greater(oleft, omode)):
            raise ValueError("left > mode")
        if np.any(np.greater(omode, oright)):
            raise ValueError("mode > right")
        if np.any(np.equal(oleft, oright)):
            raise ValueError("left == right")

        return cont_broadcast_3(&random_triangular, &self._bitgen, size, self.lock,
                            oleft, '', CONS_NONE,
                            omode, '', CONS_NONE,
                            oright, '', CONS_NONE)

    # Complicated, discrete distributions:
    def binomial(self, n, p, size=None):
        """
        binomial(n, p, size=None)

        Draw samples from a binomial distribution.

        Samples are drawn from a binomial distribution with specified
        parameters, n trials and p probability of success where
        n an integer >= 0 and p is in the interval [0,1]. (n may be
        input as a float, but it is truncated to an integer in use)

        Parameters
        ----------
        n : int or array_like of ints
            Parameter of the distribution, >= 0. Floats are also accepted,
            but they will be truncated to integers.
        p : float or array_like of floats
            Parameter of the distribution, >= 0 and <=1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``n`` and ``p`` are both scalars.
            Otherwise, ``np.broadcast(n, p).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized binomial distribution, where
            each sample is equal to the number of successes over the n trials.

        See Also
        --------
        scipy.stats.binom : probability density function, distribution or
            cumulative density function, etc.

        Notes
        -----
        The probability mass function (PMF) for the binomial distribution is

        .. math:: P(N) = \\binom{n}{N}p^N(1-p)^{n-N},

        where :math:`n` is the number of trials, :math:`p` is the probability
        of success, and :math:`N` is the number of successes.

        When estimating the standard error of a proportion in a population by
        using a random sample, the normal distribution works well unless the
        product p*n <=5, where p = population proportion estimate, and n =
        number of samples, in which case the binomial distribution is used
        instead. For example, a sample of 15 people shows 4 who are left
        handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27*15 = 4,
        so the binomial distribution should be used in this case.

        References
        ----------
        .. [1] Dalgaard, Peter, "Introductory Statistics with R",
               Springer-Verlag, 2002.
        .. [2] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
               Fifth Edition, 2002.
        .. [3] Lentner, Marvin, "Elementary Applied Statistics", Bogden
               and Quigley, 1972.
        .. [4] Weisstein, Eric W. "Binomial Distribution." From MathWorld--A
               Wolfram Web Resource.
               https://mathworld.wolfram.com/BinomialDistribution.html
        .. [5] Wikipedia, "Binomial distribution",
               https://en.wikipedia.org/wiki/Binomial_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> rng = np.random.default_rng()
        >>> n, p, size = 10, .5, 10000  
        >>> s = rng.binomial(n, p, 10000)

        Assume a company drills 9 wild-cat oil exploration wells, each with
        an estimated probability of success of ``p=0.1``. All nine wells fail. 
        What is the probability of that happening?

        Over ``size = 20,000`` trials the probability of this happening 
        is on average:

        >>> n, p, size = 9, 0.1, 20000
        >>> np.sum(rng.binomial(n=n, p=p, size=size) == 0)/size
        0.39015  # may vary

        The following can be used to visualize a sample with ``n=100``, 
        ``p=0.4`` and the corresponding probability density function:

        >>> import matplotlib.pyplot as plt
        >>> from scipy.stats import binom
        >>> n, p, size = 100, 0.4, 10000
        >>> sample = rng.binomial(n, p, size=size)
        >>> count, bins, _ = plt.hist(sample, 30, density=True)
        >>> x = np.arange(n)
        >>> y = binom.pmf(x, n, p)
        >>> plt.plot(x, y, linewidth=2, color='r')

        """

        # Uses a custom implementation since self._binomial is required
        cdef double _dp = 0
        cdef int64_t _in = 0
        cdef bint is_scalar = True
        cdef np.npy_intp i, cnt
        cdef np.ndarray randoms
        cdef np.int64_t *randoms_data
        cdef np.broadcast it

        p_arr = <np.ndarray>np.PyArray_FROM_OTF(p, np.NPY_DOUBLE, np.NPY_ARRAY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(p_arr) == 0
        n_arr = <np.ndarray>np.PyArray_FROM_OTF(n, np.NPY_INT64, np.NPY_ARRAY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(n_arr) == 0

        if not is_scalar:
            check_array_constraint(p_arr, 'p', CONS_BOUNDED_0_1)
            check_array_constraint(n_arr, 'n', CONS_NON_NEGATIVE)
            if size is not None:
                randoms = <np.ndarray>np.empty(size, np.int64)
            else:
                it = np.PyArray_MultiIterNew2(p_arr, n_arr)
                randoms = <np.ndarray>np.empty(it.shape, np.int64)

            cnt = np.PyArray_SIZE(randoms)

            it = np.PyArray_MultiIterNew3(randoms, p_arr, n_arr)
            validate_output_shape(it.shape, randoms)
            with self.lock, nogil:
                for i in range(cnt):
                    _dp = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
                    _in = (<int64_t*>np.PyArray_MultiIter_DATA(it, 2))[0]
                    (<int64_t*>np.PyArray_MultiIter_DATA(it, 0))[0] = random_binomial(&self._bitgen, _dp, _in, &self._binomial)

                    np.PyArray_MultiIter_NEXT(it)

            return randoms

        _dp = PyFloat_AsDouble(p)
        _in = <int64_t>n
        check_constraint(_dp, 'p', CONS_BOUNDED_0_1)
        check_constraint(<double>_in, 'n', CONS_NON_NEGATIVE)

        if size is None:
            with self.lock:
                return random_binomial(&self._bitgen, _dp, _in, &self._binomial)

        randoms = <np.ndarray>np.empty(size, np.int64)
        cnt = np.PyArray_SIZE(randoms)
        randoms_data = <np.int64_t *>np.PyArray_DATA(randoms)

        with self.lock, nogil:
            for i in range(cnt):
                randoms_data[i] = random_binomial(&self._bitgen, _dp, _in,
                                                  &self._binomial)

        return randoms

    def negative_binomial(self, n, p, size=None):
        """
        negative_binomial(n, p, size=None)

        Draw samples from a negative binomial distribution.

        Samples are drawn from a negative binomial distribution with specified
        parameters, `n` successes and `p` probability of success where `n`
        is > 0 and `p` is in the interval (0, 1].

        Parameters
        ----------
        n : float or array_like of floats
            Parameter of the distribution, > 0.
        p : float or array_like of floats
            Parameter of the distribution. Must satisfy 0 < p <= 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``n`` and ``p`` are both scalars.
            Otherwise, ``np.broadcast(n, p).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized negative binomial distribution,
            where each sample is equal to N, the number of failures that
            occurred before a total of n successes was reached.

        Notes
        -----
        The probability mass function of the negative binomial distribution is

        .. math:: P(N;n,p) = \\frac{\\Gamma(N+n)}{N!\\Gamma(n)}p^{n}(1-p)^{N},

        where :math:`n` is the number of successes, :math:`p` is the
        probability of success, :math:`N+n` is the number of trials, and
        :math:`\\Gamma` is the gamma function. When :math:`n` is an integer,
        :math:`\\frac{\\Gamma(N+n)}{N!\\Gamma(n)} = \\binom{N+n-1}{N}`, which is
        the more common form of this term in the pmf. The negative
        binomial distribution gives the probability of N failures given n
        successes, with a success on the last trial.

        If one throws a die repeatedly until the third time a "1" appears,
        then the probability distribution of the number of non-"1"s that
        appear before the third "1" is a negative binomial distribution.

        Because this method internally calls ``Generator.poisson`` with an
        intermediate random value, a ValueError is raised when the choice of 
        :math:`n` and :math:`p` would result in the mean + 10 sigma of the sampled
        intermediate distribution exceeding the max acceptable value of the 
        ``Generator.poisson`` method. This happens when :math:`p` is too low 
        (a lot of failures happen for every success) and :math:`n` is too big (
        a lot of successes are allowed).
        Therefore, the :math:`n` and :math:`p` values must satisfy the constraint:

        .. math:: n\\frac{1-p}{p}+10n\\sqrt{n}\\frac{1-p}{p}<2^{63}-1-10\\sqrt{2^{63}-1},

        Where the left side of the equation is the derived mean + 10 sigma of
        a sample from the gamma distribution internally used as the :math:`lam`
        parameter of a poisson sample, and the right side of the equation is
        the constraint for maximum value of :math:`lam` in ``Generator.poisson``.

        References
        ----------
        .. [1] Weisstein, Eric W. "Negative Binomial Distribution." From
               MathWorld--A Wolfram Web Resource.
               https://mathworld.wolfram.com/NegativeBinomialDistribution.html
        .. [2] Wikipedia, "Negative binomial distribution",
               https://en.wikipedia.org/wiki/Negative_binomial_distribution

        Examples
        --------
        Draw samples from the distribution:

        A real world example. A company drills wild-cat oil
        exploration wells, each with an estimated probability of
        success of 0.1.  What is the probability of having one success
        for each successive well, that is what is the probability of a
        single success after drilling 5 wells, after 6 wells, etc.?

        >>> rng = np.random.default_rng()
        >>> s = rng.negative_binomial(1, 0.1, 100000)
        >>> for i in range(1, 11): # doctest: +SKIP
        ...    probability = sum(s<i) / 100000.
        ...    print(i, "wells drilled, probability of one success =", probability)

        """

        cdef bint is_scalar = True
        cdef double *_dn
        cdef double *_dp
        cdef double _dmax_lam

        p_arr = <np.ndarray>np.PyArray_FROM_OTF(p, np.NPY_DOUBLE, np.NPY_ARRAY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(p_arr) == 0
        n_arr = <np.ndarray>np.PyArray_FROM_OTF(n, np.NPY_DOUBLE, np.NPY_ARRAY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(n_arr) == 0

        if not is_scalar:
            check_array_constraint(n_arr, 'n', CONS_POSITIVE_NOT_NAN)
            check_array_constraint(p_arr, 'p', CONS_BOUNDED_GT_0_1)
            # Check that the choice of negative_binomial parameters won't result in a
            # call to the poisson distribution function with a value of lam too large.
            max_lam_arr = (1 - p_arr) / p_arr * (n_arr + 10 * np.sqrt(n_arr))
            if np.any(np.greater(max_lam_arr, POISSON_LAM_MAX)):
                raise ValueError("n too large or p too small, see Generator.negative_binomial Notes")

        else:
            _dn = <double*>np.PyArray_DATA(n_arr)
            _dp = <double*>np.PyArray_DATA(p_arr)

            check_constraint(_dn[0], 'n', CONS_POSITIVE_NOT_NAN)
            check_constraint(_dp[0], 'p', CONS_BOUNDED_GT_0_1)
            # Check that the choice of negative_binomial parameters won't result in a
            # call to the poisson distribution function with a value of lam too large.
            _dmax_lam = (1 - _dp[0]) / _dp[0] * (_dn[0] + 10 * sqrt(_dn[0]))
            if _dmax_lam > POISSON_LAM_MAX:
                raise ValueError("n too large or p too small, see Generator.negative_binomial Notes")

        return disc(&random_negative_binomial, &self._bitgen, size, self.lock, 2, 0,
                    n_arr, 'n', CONS_NONE,
                    p_arr, 'p', CONS_NONE,
                    0.0, '', CONS_NONE)

    def poisson(self, lam=1.0, size=None):
        """
        poisson(lam=1.0, size=None)

        Draw samples from a Poisson distribution.

        The Poisson distribution is the limit of the binomial distribution
        for large N.

        Parameters
        ----------
        lam : float or array_like of floats
            Expected number of events occurring in a fixed-time interval,
            must be >= 0. A sequence must be broadcastable over the requested
            size.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``lam`` is a scalar. Otherwise,
            ``np.array(lam).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Poisson distribution.

        Notes
        -----
        The probability mass function (PMF) of Poisson distribution is

        .. math:: f(k; \\lambda)=\\frac{\\lambda^k e^{-\\lambda}}{k!}

        For events with an expected separation :math:`\\lambda` the Poisson
        distribution :math:`f(k; \\lambda)` describes the probability of
        :math:`k` events occurring within the observed
        interval :math:`\\lambda`.

        Because the output is limited to the range of the C int64 type, a
        ValueError is raised when `lam` is within 10 sigma of the maximum
        representable value.

        References
        ----------
        .. [1] Weisstein, Eric W. "Poisson Distribution."
               From MathWorld--A Wolfram Web Resource.
               https://mathworld.wolfram.com/PoissonDistribution.html
        .. [2] Wikipedia, "Poisson distribution",
               https://en.wikipedia.org/wiki/Poisson_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> rng = np.random.default_rng()
        >>> lam, size = 5, 10000
        >>> s = rng.poisson(lam=lam, size=size)

        Verify the mean and variance, which should be approximately ``lam``:
        
        >>> s.mean(), s.var()
        (4.9917 5.1088311)  # may vary

        Display the histogram and probability mass function:

        >>> import matplotlib.pyplot as plt
        >>> from scipy import stats
        >>> x = np.arange(0, 21)
        >>> pmf = stats.poisson.pmf(x, mu=lam)
        >>> plt.hist(s, bins=x, density=True, width=0.5)
        >>> plt.stem(x, pmf, 'C1-')
        >>> plt.show()

        Draw each 100 values for lambda 100 and 500:

        >>> s = rng.poisson(lam=(100., 500.), size=(100, 2))

        """
        return disc(&random_poisson, &self._bitgen, size, self.lock, 1, 0,
                    lam, 'lam', CONS_POISSON,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def zipf(self, a, size=None):
        """
        zipf(a, size=None)

        Draw samples from a Zipf distribution.

        Samples are drawn from a Zipf distribution with specified parameter
        `a` > 1.

        The Zipf distribution (also known as the zeta distribution) is a
        discrete probability distribution that satisfies Zipf's law: the
        frequency of an item is inversely proportional to its rank in a
        frequency table.

        Parameters
        ----------
        a : float or array_like of floats
            Distribution parameter. Must be greater than 1.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``a`` is a scalar. Otherwise,
            ``np.array(a).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized Zipf distribution.

        See Also
        --------
        scipy.stats.zipf : probability density function, distribution, or
            cumulative density function, etc.

        Notes
        -----
        The probability mass function (PMF) for the Zipf distribution is

        .. math:: p(k) = \\frac{k^{-a}}{\\zeta(a)},

        for integers :math:`k \geq 1`, where :math:`\\zeta` is the Riemann Zeta
        function.

        It is named for the American linguist George Kingsley Zipf, who noted
        that the frequency of any word in a sample of a language is inversely
        proportional to its rank in the frequency table.

        References
        ----------
        .. [1] Zipf, G. K., "Selected Studies of the Principle of Relative
               Frequency in Language," Cambridge, MA: Harvard Univ. Press,
               1932.

        Examples
        --------
        Draw samples from the distribution:

        >>> a = 4.0
        >>> n = 20000
        >>> rng = np.random.default_rng()
        >>> s = rng.zipf(a, size=n)

        Display the histogram of the samples, along with
        the expected histogram based on the probability
        density function:

        >>> import matplotlib.pyplot as plt
        >>> from scipy.special import zeta  # doctest: +SKIP

        `bincount` provides a fast histogram for small integers.

        >>> count = np.bincount(s)
        >>> k = np.arange(1, s.max() + 1)

        >>> plt.bar(k, count[1:], alpha=0.5, label='sample count')
        >>> plt.plot(k, n*(k**-a)/zeta(a), 'k.-', alpha=0.5,
        ...          label='expected count')   # doctest: +SKIP
        >>> plt.semilogy()
        >>> plt.grid(alpha=0.4)
        >>> plt.legend()
        >>> plt.title(f'Zipf sample, a={a}, size={n}')
        >>> plt.show()

        """
        return disc(&random_zipf, &self._bitgen, size, self.lock, 1, 0,
                    a, 'a', CONS_GT_1,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def geometric(self, p, size=None):
        """
        geometric(p, size=None)

        Draw samples from the geometric distribution.

        Bernoulli trials are experiments with one of two outcomes:
        success or failure (an example of such an experiment is flipping
        a coin).  The geometric distribution models the number of trials
        that must be run in order to achieve success.  It is therefore
        supported on the positive integers, ``k = 1, 2, ...``.

        The probability mass function of the geometric distribution is

        .. math:: f(k) = (1 - p)^{k - 1} p

        where `p` is the probability of success of an individual trial.

        Parameters
        ----------
        p : float or array_like of floats
            The probability of success of an individual trial.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``p`` is a scalar.  Otherwise,
            ``np.array(p).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized geometric distribution.

        References
        ----------

        .. [1] Wikipedia, "Geometric distribution",
               https://en.wikipedia.org/wiki/Geometric_distribution

        Examples
        --------
        Draw 10,000 values from the geometric distribution, with the 
        probability of an individual success equal to ``p = 0.35``:

        >>> p, size = 0.35, 10000
        >>> rng = np.random.default_rng()
        >>> sample = rng.geometric(p=p, size=size)

        What proportion of trials succeeded after a single run?

        >>> (sample == 1).sum()/size
        0.34889999999999999  # may vary

        The geometric distribution with ``p=0.35`` looks as follows:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(sample, bins=30, density=True)
        >>> plt.plot(bins, (1-p)**(bins-1)*p)
        >>> plt.xlim([0, 25])
        >>> plt.show()
        
        """
        return disc(&random_geometric, &self._bitgen, size, self.lock, 1, 0,
                    p, 'p', CONS_BOUNDED_GT_0_1,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        """
        hypergeometric(ngood, nbad, nsample, size=None)

        Draw samples from a Hypergeometric distribution.

        Samples are drawn from a hypergeometric distribution with specified
        parameters, `ngood` (ways to make a good selection), `nbad` (ways to make
        a bad selection), and `nsample` (number of items sampled, which is less
        than or equal to the sum ``ngood + nbad``).

        Parameters
        ----------
        ngood : int or array_like of ints
            Number of ways to make a good selection.  Must be nonnegative and
            less than 10**9.
        nbad : int or array_like of ints
            Number of ways to make a bad selection.  Must be nonnegative and
            less than 10**9.
        nsample : int or array_like of ints
            Number of items sampled.  Must be nonnegative and less than
            ``ngood + nbad``.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if `ngood`, `nbad`, and `nsample`
            are all scalars.  Otherwise, ``np.broadcast(ngood, nbad, nsample).size``
            samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized hypergeometric distribution. Each
            sample is the number of good items within a randomly selected subset of
            size `nsample` taken from a set of `ngood` good items and `nbad` bad items.

        See Also
        --------
        multivariate_hypergeometric : Draw samples from the multivariate
            hypergeometric distribution.
        scipy.stats.hypergeom : probability density function, distribution or
            cumulative density function, etc.

        Notes
        -----
        The probability mass function (PMF) for the Hypergeometric distribution is

        .. math:: P(x) = \\frac{\\binom{g}{x}\\binom{b}{n-x}}{\\binom{g+b}{n}},

        where :math:`0 \\le x \\le n` and :math:`n-b \\le x \\le g`

        for P(x) the probability of ``x`` good results in the drawn sample,
        g = `ngood`, b = `nbad`, and n = `nsample`.

        Consider an urn with black and white marbles in it, `ngood` of them
        are black and `nbad` are white. If you draw `nsample` balls without
        replacement, then the hypergeometric distribution describes the
        distribution of black balls in the drawn sample.

        Note that this distribution is very similar to the binomial
        distribution, except that in this case, samples are drawn without
        replacement, whereas in the Binomial case samples are drawn with
        replacement (or the sample space is infinite). As the sample space
        becomes large, this distribution approaches the binomial.

        The arguments `ngood` and `nbad` each must be less than `10**9`. For
        extremely large arguments, the algorithm that is used to compute the
        samples [4]_ breaks down because of loss of precision in floating point
        calculations.  For such large values, if `nsample` is not also large,
        the distribution can be approximated with the binomial distribution,
        `binomial(n=nsample, p=ngood/(ngood + nbad))`.

        References
        ----------
        .. [1] Lentner, Marvin, "Elementary Applied Statistics", Bogden
               and Quigley, 1972.
        .. [2] Weisstein, Eric W. "Hypergeometric Distribution." From
               MathWorld--A Wolfram Web Resource.
               https://mathworld.wolfram.com/HypergeometricDistribution.html
        .. [3] Wikipedia, "Hypergeometric distribution",
               https://en.wikipedia.org/wiki/Hypergeometric_distribution
        .. [4] Stadlober, Ernst, "The ratio of uniforms approach for generating
               discrete random variates", Journal of Computational and Applied
               Mathematics, 31, pp. 181-189 (1990).

        Examples
        --------
        Draw samples from the distribution:

        >>> rng = np.random.default_rng()
        >>> ngood, nbad, nsamp = 100, 2, 10
        # number of good, number of bad, and number of samples
        >>> s = rng.hypergeometric(ngood, nbad, nsamp, 1000)
        >>> from matplotlib.pyplot import hist
        >>> hist(s)
        #   note that it is very unlikely to grab both bad items

        Suppose you have an urn with 15 white and 15 black marbles.
        If you pull 15 marbles at random, how likely is it that
        12 or more of them are one color?

        >>> s = rng.hypergeometric(15, 15, 15, 100000)
        >>> sum(s>=12)/100000. + sum(s<=3)/100000.
        #   answer = 0.003 ... pretty unlikely!

        """
        cdef double HYPERGEOM_MAX = 10**9
        cdef bint is_scalar = True
        cdef np.ndarray ongood, onbad, onsample
        cdef int64_t lngood, lnbad, lnsample

        ongood = <np.ndarray>np.PyArray_FROM_OTF(ngood, np.NPY_INT64, np.NPY_ARRAY_ALIGNED)
        onbad = <np.ndarray>np.PyArray_FROM_OTF(nbad, np.NPY_INT64, np.NPY_ARRAY_ALIGNED)
        onsample = <np.ndarray>np.PyArray_FROM_OTF(nsample, np.NPY_INT64, np.NPY_ARRAY_ALIGNED)

        if np.PyArray_NDIM(ongood) == np.PyArray_NDIM(onbad) == np.PyArray_NDIM(onsample) == 0:

            lngood = <int64_t>ngood
            lnbad = <int64_t>nbad
            lnsample = <int64_t>nsample

            if lngood >= HYPERGEOM_MAX or lnbad >= HYPERGEOM_MAX:
                raise ValueError("both ngood and nbad must be less than %d" %
                                 HYPERGEOM_MAX)
            if lngood + lnbad < lnsample:
                raise ValueError("ngood + nbad < nsample")
            return disc(&random_hypergeometric, &self._bitgen, size, self.lock, 0, 3,
                        lngood, 'ngood', CONS_NON_NEGATIVE,
                        lnbad, 'nbad', CONS_NON_NEGATIVE,
                        lnsample, 'nsample', CONS_NON_NEGATIVE)

        if np.any(ongood >= HYPERGEOM_MAX) or np.any(onbad >= HYPERGEOM_MAX):
            raise ValueError("both ngood and nbad must be less than %d" %
                             HYPERGEOM_MAX)

        if np.any(np.less(np.add(ongood, onbad), onsample)):
            raise ValueError("ngood + nbad < nsample")

        return discrete_broadcast_iii(&random_hypergeometric, &self._bitgen, size, self.lock,
                                      ongood, 'ngood', CONS_NON_NEGATIVE,
                                      onbad, 'nbad', CONS_NON_NEGATIVE,
                                      onsample, 'nsample', CONS_NON_NEGATIVE)

    def logseries(self, p, size=None):
        """
        logseries(p, size=None)

        Draw samples from a logarithmic series distribution.

        Samples are drawn from a log series distribution with specified
        shape parameter, 0 <= ``p`` < 1.

        Parameters
        ----------
        p : float or array_like of floats
            Shape parameter for the distribution.  Must be in the range [0, 1).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``p`` is a scalar.  Otherwise,
            ``np.array(p).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized logarithmic series distribution.

        See Also
        --------
        scipy.stats.logser : probability density function, distribution or
            cumulative density function, etc.

        Notes
        -----
        The probability mass function for the Log Series distribution is

        .. math:: P(k) = \\frac{-p^k}{k \\ln(1-p)},

        where p = probability.

        The log series distribution is frequently used to represent species
        richness and occurrence, first proposed by Fisher, Corbet, and
        Williams in 1943 [2].  It may also be used to model the numbers of
        occupants seen in cars [3].

        References
        ----------
        .. [1] Buzas, Martin A.; Culver, Stephen J.,  Understanding regional
               species diversity through the log series distribution of
               occurrences: BIODIVERSITY RESEARCH Diversity & Distributions,
               Volume 5, Number 5, September 1999 , pp. 187-195(9).
        .. [2] Fisher, R.A,, A.S. Corbet, and C.B. Williams. 1943. The
               relation between the number of species and the number of
               individuals in a random sample of an animal population.
               Journal of Animal Ecology, 12:42-58.
        .. [3] D. J. Hand, F. Daly, D. Lunn, E. Ostrowski, A Handbook of Small
               Data Sets, CRC Press, 1994.
        .. [4] Wikipedia, "Logarithmic distribution",
               https://en.wikipedia.org/wiki/Logarithmic_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> a = .6
        >>> rng = np.random.default_rng()
        >>> s = rng.logseries(a, 10000)
        >>> import matplotlib.pyplot as plt
        >>> bins = np.arange(-.5, max(s) + .5 )
        >>> count, bins, _ = plt.hist(s, bins=bins, label='Sample count')

        #   plot against distribution

        >>> def logseries(k, p):
        ...     return -p**k/(k*np.log(1-p))
        >>> centres = np.arange(1, max(s) + 1)
        >>> plt.plot(centres, logseries(centres, a) * s.size, 'r', label='logseries PMF')
        >>> plt.legend()
        >>> plt.show()

        """
        return disc(&random_logseries, &self._bitgen, size, self.lock, 1, 0,
                 p, 'p', CONS_BOUNDED_LT_0_1,
                 0.0, '', CONS_NONE,
                 0.0, '', CONS_NONE)

    # Multivariate distributions:
    def multivariate_normal(self, mean, cov, size=None, check_valid='warn',
                            tol=1e-8, *, method='svd'):
        """
        multivariate_normal(mean, cov, size=None, check_valid='warn',
                            tol=1e-8, *, method='svd')

        Draw random samples from a multivariate normal distribution.

        The multivariate normal, multinormal or Gaussian distribution is a
        generalization of the one-dimensional normal distribution to higher
        dimensions.  Such a distribution is specified by its mean and
        covariance matrix.  These parameters are analogous to the mean
        (average or "center") and variance (the squared standard deviation,
        or "width") of the one-dimensional normal distribution.

        Parameters
        ----------
        mean : 1-D array_like, of length N
            Mean of the N-dimensional distribution.
        cov : 2-D array_like, of shape (N, N)
            Covariance matrix of the distribution. It must be symmetric and
            positive-semidefinite for proper sampling.
        size : int or tuple of ints, optional
            Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
            generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
            each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
            If no shape is specified, a single (`N`-D) sample is returned.
        check_valid : { 'warn', 'raise', 'ignore' }, optional
            Behavior when the covariance matrix is not positive semidefinite.
        tol : float, optional
            Tolerance when checking the singular values in covariance matrix.
            cov is cast to double before the check.
        method : { 'svd', 'eigh', 'cholesky'}, optional
            The cov input is used to compute a factor matrix A such that
            ``A @ A.T = cov``. This argument is used to select the method
            used to compute the factor matrix A. The default method 'svd' is
            the slowest, while 'cholesky' is the fastest but less robust than
            the slowest method. The method `eigh` uses eigen decomposition to
            compute A and is faster than svd but slower than cholesky.

        Returns
        -------
        out : ndarray
            The drawn samples, of shape *size*, if that was provided.  If not,
            the shape is ``(N,)``.

            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
            value drawn from the distribution.

        Notes
        -----
        The mean is a coordinate in N-dimensional space, which represents the
        location where samples are most likely to be generated.  This is
        analogous to the peak of the bell curve for the one-dimensional or
        univariate normal distribution.

        Covariance indicates the level to which two variables vary together.
        From the multivariate normal distribution, we draw N-dimensional
        samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
        element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
        The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
        "spread").

        Instead of specifying the full covariance matrix, popular
        approximations include:

        - Spherical covariance (`cov` is a multiple of the identity matrix)
        - Diagonal covariance (`cov` has non-negative elements, and only on
          the diagonal)

        This geometrical property can be seen in two dimensions by plotting
        generated data-points:

        >>> mean = [0, 0]
        >>> cov = [[1, 0], [0, 100]]  # diagonal covariance

        Diagonal covariance means that points are oriented along x or y-axis:

        >>> import matplotlib.pyplot as plt
        >>> rng = np.random.default_rng()
        >>> x, y = rng.multivariate_normal(mean, cov, 5000).T
        >>> plt.plot(x, y, 'x')
        >>> plt.axis('equal')
        >>> plt.show()

        Note that the covariance matrix must be positive semidefinite (a.k.a.
        nonnegative-definite). Otherwise, the behavior of this method is
        undefined and backwards compatibility is not guaranteed.

        This function internally uses linear algebra routines, and thus results
        may not be identical (even up to precision) across architectures, OSes,
        or even builds. For example, this is likely if ``cov`` has multiple equal
        singular values and ``method`` is ``'svd'`` (default). In this case,
        ``method='cholesky'`` may be more robust.

        References
        ----------
        .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
               Processes," 3rd ed., New York: McGraw-Hill, 1991.
        .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
               Classification," 2nd ed., New York: Wiley, 2001.

        Examples
        --------
        >>> mean = (1, 2)
        >>> cov = [[1, 0], [0, 1]]
        >>> rng = np.random.default_rng()
        >>> x = rng.multivariate_normal(mean, cov, (3, 3))
        >>> x.shape
        (3, 3, 2)

        We can use a different method other than the default to factorize cov:

        >>> y = rng.multivariate_normal(mean, cov, (3, 3), method='cholesky')
        >>> y.shape
        (3, 3, 2)

        Here we generate 800 samples from the bivariate normal distribution
        with mean [0, 0] and covariance matrix [[6, -3], [-3, 3.5]].  The
        expected variances of the first and second components of the sample
        are 6 and 3.5, respectively, and the expected correlation
        coefficient is -3/sqrt(6*3.5) ≈ -0.65465.

        >>> cov = np.array([[6, -3], [-3, 3.5]])
        >>> pts = rng.multivariate_normal([0, 0], cov, size=800)

        Check that the mean, covariance, and correlation coefficient of the
        sample are close to the expected values:

        >>> pts.mean(axis=0)
        array([ 0.0326911 , -0.01280782])  # may vary
        >>> np.cov(pts.T)
        array([[ 5.96202397, -2.85602287],
               [-2.85602287,  3.47613949]])  # may vary
        >>> np.corrcoef(pts.T)[0, 1]
        -0.6273591314603949  # may vary

        We can visualize this data with a scatter plot.  The orientation
        of the point cloud illustrates the negative correlation of the
        components of this sample.

        >>> import matplotlib.pyplot as plt
        >>> plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)
        >>> plt.axis('equal')
        >>> plt.grid()
        >>> plt.show()

        """
        if method not in {'eigh', 'svd', 'cholesky'}:
            raise ValueError(
                "method must be one of {'eigh', 'svd', 'cholesky'}")

        # Check preconditions on arguments
        mean = np.array(mean)
        cov = np.array(cov)

        if (np.issubdtype(mean.dtype, np.complexfloating) or
                np.issubdtype(cov.dtype, np.complexfloating)):
            raise TypeError("mean and cov must not be complex")

        if size is None:
            shape = []
        elif isinstance(size, (int, np.integer)):
            shape = [size]
        else:
            shape = size

        if len(mean.shape) != 1:
            raise ValueError("mean must be 1 dimensional")
        if (len(cov.shape) != 2) or (cov.shape[0] != cov.shape[1]):
            raise ValueError("cov must be 2 dimensional and square")
        if mean.shape[0] != cov.shape[0]:
            raise ValueError("mean and cov must have same length")

        # Compute shape of output and create a matrix of independent
        # standard normally distributed random numbers. The matrix has rows
        # with the same length as mean and as many rows are necessary to
        # form a matrix of shape final_shape.
        final_shape = list(shape[:])
        final_shape.append(mean.shape[0])
        x = self.standard_normal(final_shape).reshape(-1, mean.shape[0])

        # Transform matrix of standard normals into matrix where each row
        # contains multivariate normals with the desired covariance.
        # Compute A such that dot(transpose(A),A) == cov.
        # Then the matrix products of the rows of x and A has the desired
        # covariance. Note that sqrt(s)*v where (u,s,v) is the singular value
        # decomposition of cov is such an A.
        #
        # Also check that cov is positive-semidefinite. If so, the u.T and v
        # matrices should be equal up to roundoff error if cov is
        # symmetric and the singular value of the corresponding row is
        # not zero. We continue to use the SVD rather than Cholesky in
        # order to preserve current outputs. Note that symmetry has not
        # been checked.

        # GH10839, ensure double to make tol meaningful
        cov = cov.astype(np.double)
        if method == 'svd':
            from numpy.linalg import svd
            (u, s, vh) = svd(cov)
        elif method == 'eigh':
            from numpy.linalg import eigh
            # could call linalg.svd(hermitian=True), but that calculates a vh we don't need
            (s, u)  = eigh(cov)
        else:
            from numpy.linalg import cholesky
            l = cholesky(cov)

        # make sure check_valid is ignored when method == 'cholesky'
        # since the decomposition will have failed if cov is not valid.
        if check_valid != 'ignore' and method != 'cholesky':
            if check_valid != 'warn' and check_valid != 'raise':
                raise ValueError(
                    "check_valid must equal 'warn', 'raise', or 'ignore'")
            if method == 'svd':
                psd = np.allclose(np.dot(vh.T * s, vh), cov, rtol=tol, atol=tol)
            else:
                psd = not np.any(s < -tol)
            if not psd:
                if check_valid == 'warn':
                    warnings.warn("covariance is not symmetric positive-semidefinite.",
                                  RuntimeWarning)
                else:
                    raise ValueError("covariance is not symmetric positive-semidefinite.")

        if method == 'cholesky':
            _factor = l
        elif method == 'eigh':
            # if check_valid == 'ignore' we need to ensure that np.sqrt does not
            # return a NaN if s is a very small negative number that is
            # approximately zero or when the covariance is not positive-semidefinite
            _factor = u * np.sqrt(abs(s))
        else:
            _factor = u * np.sqrt(s)

        x = mean + x @ _factor.T
        x.shape = tuple(final_shape)
        return x

    def multinomial(self, object n, object pvals, size=None):
        """
        multinomial(n, pvals, size=None)

        Draw samples from a multinomial distribution.

        The multinomial distribution is a multivariate generalization of the
        binomial distribution.  Take an experiment with one of ``p``
        possible outcomes.  An example of such an experiment is throwing a dice,
        where the outcome can be 1 through 6.  Each sample drawn from the
        distribution represents `n` such experiments.  Its values,
        ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the
        outcome was ``i``.

        Parameters
        ----------
        n : int or array-like of ints
            Number of experiments.
        pvals : array-like of floats
            Probabilities of each of the ``p`` different outcomes with shape
            ``(k0, k1, ..., kn, p)``. Each element ``pvals[i,j,...,:]`` must
            sum to 1 (however, the last element is always assumed to account
            for the remaining probability, as long as
            ``sum(pvals[..., :-1], axis=-1) <= 1.0``. Must have at least 1
            dimension where pvals.shape[-1] > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn each with ``p`` elements. Default
            is None where the output size is determined by the broadcast shape
            of ``n`` and all by the final dimension of ``pvals``, which is
            denoted as ``b=(b0, b1, ..., bq)``. If size is not None, then it
            must be compatible with the broadcast shape ``b``. Specifically,
            size must have ``q`` or more elements and size[-(q-j):] must equal
            ``bj``.

        Returns
        -------
        out : ndarray
            The drawn samples, of shape size, if provided. When size is
            provided, the output shape is size + (p,)  If not specified,
            the shape is determined by the broadcast shape of ``n`` and
            ``pvals``, ``(b0, b1, ..., bq)`` augmented with the dimension of
            the multinomial, ``p``, so that that output shape is
            ``(b0, b1, ..., bq, p)``.

            Each entry ``out[i,j,...,:]`` is a ``p``-dimensional value drawn
            from the distribution.

        Examples
        --------
        Throw a dice 20 times:

        >>> rng = np.random.default_rng()
        >>> rng.multinomial(20, [1/6.]*6, size=1)
        array([[4, 1, 7, 5, 2, 1]])  # random

        It landed 4 times on 1, once on 2, etc.

        Now, throw the dice 20 times, and 20 times again:

        >>> rng.multinomial(20, [1/6.]*6, size=2)
        array([[3, 4, 3, 3, 4, 3],
               [2, 4, 3, 4, 0, 7]])  # random

        For the first run, we threw 3 times 1, 4 times 2, etc.  For the second,
        we threw 2 times 1, 4 times 2, etc.

        Now, do one experiment throwing the dice 10 time, and 10 times again,
        and another throwing the dice 20 times, and 20 times again:

        >>> rng.multinomial([[10], [20]], [1/6.]*6, size=(2, 2))
        array([[[2, 4, 0, 1, 2, 1],
                [1, 3, 0, 3, 1, 2]],
               [[1, 4, 4, 4, 4, 3],
                [3, 3, 2, 5, 5, 2]]])  # random

        The first array shows the outcomes of throwing the dice 10 times, and
        the second shows the outcomes from throwing the dice 20 times.

        A loaded die is more likely to land on number 6:

        >>> rng.multinomial(100, [1/7.]*5 + [2/7.])
        array([11, 16, 14, 17, 16, 26])  # random

        Simulate 10 throws of a 4-sided die and 20 throws of a 6-sided die

        >>> rng.multinomial([10, 20],[[1/4]*4 + [0]*2, [1/6]*6])
        array([[2, 1, 4, 3, 0, 0],
               [3, 3, 3, 6, 1, 4]], dtype=int64)  # random

        Generate categorical random variates from two categories where the
        first has 3 outcomes and the second has 2.

        >>> rng.multinomial(1, [[.1, .5, .4 ], [.3, .7, .0]])
        array([[0, 0, 1],
               [0, 1, 0]], dtype=int64)  # random

        ``argmax(axis=-1)`` is then used to return the categories.

        >>> pvals = [[.1, .5, .4 ], [.3, .7, .0]]
        >>> rvs = rng.multinomial(1, pvals, size=(4,2))
        >>> rvs.argmax(axis=-1)
        array([[0, 1],
               [2, 0],
               [2, 1],
               [2, 0]], dtype=int64)  # random

        The same output dimension can be produced using broadcasting.

        >>> rvs = rng.multinomial([[1]] * 4, pvals)
        >>> rvs.argmax(axis=-1)
        array([[0, 1],
               [2, 0],
               [2, 1],
               [2, 0]], dtype=int64)  # random

        The probability inputs should be normalized. As an implementation
        detail, the value of the last entry is ignored and assumed to take
        up any leftover probability mass, but this should not be relied on.
        A biased coin which has twice as much weight on one side as on the
        other should be sampled like so:

        >>> rng.multinomial(100, [1.0 / 3, 2.0 / 3])  # RIGHT
        array([38, 62])  # random

        not like:

        >>> rng.multinomial(100, [1.0, 2.0])  # WRONG
        Traceback (most recent call last):
        ValueError: pvals < 0, pvals > 1 or pvals contains NaNs

        """

        cdef np.npy_intp d, i, sz, offset, pi
        cdef np.ndarray parr, mnarr, on, temp_arr
        cdef double *pix
        cdef int ndim
        cdef int64_t *mnix
        cdef int64_t ni
        cdef np.broadcast it
        on = <np.ndarray>np.PyArray_FROM_OTF(n,
                                             np.NPY_INT64,
                                             np.NPY_ARRAY_ALIGNED |
                                             np.NPY_ARRAY_C_CONTIGUOUS)
        parr = <np.ndarray>np.PyArray_FROM_OTF(pvals,
                                               np.NPY_DOUBLE,
                                               np.NPY_ARRAY_ALIGNED |
                                               np.NPY_ARRAY_C_CONTIGUOUS)
        ndim = parr.ndim
        d = parr.shape[ndim - 1] if ndim >= 1 else 0
        if d == 0:
            raise ValueError(
                "pvals must have at least 1 dimension and the last dimension "
                "of pvals must be greater than 0."
            )

        check_array_constraint(parr, 'pvals', CONS_BOUNDED_0_1)
        pix = <double*>np.PyArray_DATA(parr)
        sz = np.PyArray_SIZE(parr)
        # Cython 0.29.20 would not correctly translate the range-based for
        # loop to a C for loop
        # for offset in range(<np.npy_intp>0, sz, d):
        offset = 0
        while offset < sz:
            if kahan_sum(pix + offset, d-1) > (1.0 + 1e-12):
                # When floating, but not float dtype, and close, improve the error
                # 1.0001 works for float16 and float32
                slice_repr = "[:-1]" if ndim == 1 else "[...,:-1]"
                if (isinstance(pvals, np.ndarray)
                        and np.issubdtype(pvals.dtype, np.floating)
                        and pvals.dtype != float
                        and pvals.sum() < 1.0001):
                    msg = (f"sum(pvals{slice_repr}.astype(np.float64)) > 1.0."
                           " The pvals array is cast to 64-bit floating"
                           " point prior to checking the sum. Precision "
                           "changes when casting may cause problems even "
                           "if the sum of the original pvals is valid.")
                else:
                    msg = f"sum(pvals{slice_repr}) > 1.0"
                raise ValueError(msg)
            offset += d

        if np.PyArray_NDIM(on) != 0 or ndim > 1: # vector
            check_array_constraint(on, 'n', CONS_NON_NEGATIVE)
            # This provides the offsets to use in the C-contig parr when
            # broadcasting
            offsets = <np.ndarray>np.arange(
                0, np.PyArray_SIZE(parr), d, dtype=np.intp
            ).reshape((<object>parr).shape[:ndim - 1])
            if size is None:
                it = np.PyArray_MultiIterNew2(on, offsets)
            else:
                temp = np.empty(size, dtype=np.int8)
                temp_arr = <np.ndarray>temp
                it = np.PyArray_MultiIterNew3(on, offsets, temp_arr)
                # Validate size and the broadcast shape
                try:
                    size = (operator.index(size),)
                except:
                    size = tuple(size)
                # This test verifies that an axis with dim 1 in size has not
                # been increased by broadcasting with the input
                if it.shape != size:
                    raise ValueError(
                        f"Output size {size} is not compatible with "
                        f"broadcast dimensions of inputs {it.shape}."
                    )
            shape = it.shape + (d,)
            multin = np.zeros(shape, dtype=np.int64)
            mnarr = <np.ndarray>multin
            mnix = <int64_t*>np.PyArray_DATA(mnarr)
            offset = 0
            sz = it.size
            with self.lock, nogil:
                for i in range(sz):
                    ni = (<int64_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
                    pi = (<np.npy_intp*>np.PyArray_MultiIter_DATA(it, 1))[0]
                    random_multinomial(&self._bitgen, ni, &mnix[offset], &pix[pi], d, &self._binomial)
                    offset += d
                    np.PyArray_MultiIter_NEXT(it)
            return multin

        if size is None:
            shape = (d,)
        else:
            try:
                shape = (operator.index(size), d)
            except:
                shape = tuple(size) + (d,)

        multin = np.zeros(shape, dtype=np.int64)
        mnarr = <np.ndarray>multin
        mnix = <int64_t*>np.PyArray_DATA(mnarr)
        sz = np.PyArray_SIZE(mnarr)
        ni = n
        check_constraint(ni, 'n', CONS_NON_NEGATIVE)
        offset = 0
        with self.lock, nogil:
            for i in range(sz // d):
                random_multinomial(&self._bitgen, ni, &mnix[offset], pix, d, &self._binomial)
                offset += d

        return multin

    def multivariate_hypergeometric(self, object colors, object nsample,
                                    size=None, method='marginals'):
        """
        multivariate_hypergeometric(colors, nsample, size=None,
                                    method='marginals')

        Generate variates from a multivariate hypergeometric distribution.

        The multivariate hypergeometric distribution is a generalization
        of the hypergeometric distribution.

        Choose ``nsample`` items at random without replacement from a
        collection with ``N`` distinct types.  ``N`` is the length of
        ``colors``, and the values in ``colors`` are the number of occurrences
        of that type in the collection.  The total number of items in the
        collection is ``sum(colors)``.  Each random variate generated by this
        function is a vector of length ``N`` holding the counts of the
        different types that occurred in the ``nsample`` items.

        The name ``colors`` comes from a common description of the
        distribution: it is the probability distribution of the number of
        marbles of each color selected without replacement from an urn
        containing marbles of different colors; ``colors[i]`` is the number
        of marbles in the urn with color ``i``.

        Parameters
        ----------
        colors : sequence of integers
            The number of each type of item in the collection from which
            a sample is drawn.  The values in ``colors`` must be nonnegative.
            To avoid loss of precision in the algorithm, ``sum(colors)``
            must be less than ``10**9`` when `method` is "marginals".
        nsample : int
            The number of items selected.  ``nsample`` must not be greater
            than ``sum(colors)``.
        size : int or tuple of ints, optional
            The number of variates to generate, either an integer or a tuple
            holding the shape of the array of variates.  If the given size is,
            e.g., ``(k, m)``, then ``k * m`` variates are drawn, where one
            variate is a vector of length ``len(colors)``, and the return value
            has shape ``(k, m, len(colors))``.  If `size` is an integer, the
            output has shape ``(size, len(colors))``.  Default is None, in
            which case a single variate is returned as an array with shape
            ``(len(colors),)``.
        method : string, optional
            Specify the algorithm that is used to generate the variates.
            Must be 'count' or 'marginals' (the default).  See the Notes
            for a description of the methods.

        Returns
        -------
        variates : ndarray
            Array of variates drawn from the multivariate hypergeometric
            distribution.

        See Also
        --------
        hypergeometric : Draw samples from the (univariate) hypergeometric
            distribution.

        Notes
        -----
        The two methods do not return the same sequence of variates.

        The "count" algorithm is roughly equivalent to the following numpy
        code::

            choices = np.repeat(np.arange(len(colors)), colors)
            selection = np.random.choice(choices, nsample, replace=False)
            variate = np.bincount(selection, minlength=len(colors))

        The "count" algorithm uses a temporary array of integers with length
        ``sum(colors)``.

        The "marginals" algorithm generates a variate by using repeated
        calls to the univariate hypergeometric sampler.  It is roughly
        equivalent to::

            variate = np.zeros(len(colors), dtype=np.int64)
            # `remaining` is the cumulative sum of `colors` from the last
            # element to the first; e.g. if `colors` is [3, 1, 5], then
            # `remaining` is [9, 6, 5].
            remaining = np.cumsum(colors[::-1])[::-1]
            for i in range(len(colors)-1):
                if nsample < 1:
                    break
                variate[i] = hypergeometric(colors[i], remaining[i+1],
                                           nsample)
                nsample -= variate[i]
            variate[-1] = nsample

        The default method is "marginals".  For some cases (e.g. when
        `colors` contains relatively small integers), the "count" method
        can be significantly faster than the "marginals" method.  If
        performance of the algorithm is important, test the two methods
        with typical inputs to decide which works best.

        Examples
        --------
        >>> colors = [16, 8, 4]
        >>> seed = 4861946401452
        >>> gen = np.random.Generator(np.random.PCG64(seed))
        >>> gen.multivariate_hypergeometric(colors, 6)
        array([5, 0, 1])
        >>> gen.multivariate_hypergeometric(colors, 6, size=3)
        array([[5, 0, 1],
               [2, 2, 2],
               [3, 3, 0]])
        >>> gen.multivariate_hypergeometric(colors, 6, size=(2, 2))
        array([[[3, 2, 1],
                [3, 2, 1]],
               [[4, 1, 1],
                [3, 2, 1]]])

        """
        cdef int64_t nsamp
        cdef size_t num_colors
        cdef int64_t total
        cdef int64_t *colors_ptr
        cdef int64_t max_index
        cdef size_t num_variates
        cdef int64_t *variates_ptr
        cdef int result

        if method not in ['count', 'marginals']:
            raise ValueError('method must be "count" or "marginals".')

        try:
            operator.index(nsample)
        except TypeError:
            raise ValueError('nsample must be an integer')

        if nsample < 0:
            raise ValueError("nsample must be nonnegative.")
        if nsample > INT64_MAX:
            raise ValueError("nsample must not exceed %d" % INT64_MAX)
        nsamp = nsample

        # Validation of colors, a 1-d sequence of nonnegative integers.
        invalid_colors = False
        try:
            colors = np.asarray(colors)
            if colors.ndim != 1:
                invalid_colors = True
            elif colors.size > 0 and not np.issubdtype(colors.dtype,
                                                       np.integer):
                invalid_colors = True
            elif np.any((colors < 0) | (colors > INT64_MAX)):
                invalid_colors = True
        except ValueError:
            invalid_colors = True
        if invalid_colors:
            raise ValueError('colors must be a one-dimensional sequence '
                             'of nonnegative integers not exceeding %d.' %
                             INT64_MAX)

        colors = np.ascontiguousarray(colors, dtype=np.int64)
        num_colors = colors.size

        colors_ptr = <int64_t *> np.PyArray_DATA(colors)

        total = _safe_sum_nonneg_int64(num_colors, colors_ptr)
        if total == -1:
            raise ValueError("sum(colors) must not exceed the maximum value "
                             "of a 64 bit signed integer (%d)" % INT64_MAX)

        if method == 'marginals' and total >= 1000000000:
            raise ValueError('When method is "marginals", sum(colors) must '
                             'be less than 1000000000.')

        # The C code that implements the 'count' method will malloc an
        # array of size total*sizeof(size_t). Here we ensure that that
        # product does not overflow.
        if SIZE_MAX > <uint64_t>INT64_MAX:
            max_index = INT64_MAX // sizeof(size_t)
        else:
            max_index = SIZE_MAX // sizeof(size_t)
        if method == 'count' and total > max_index:
            raise ValueError("When method is 'count', sum(colors) must not "
                             "exceed %d" % max_index)
        if nsamp > total:
            raise ValueError("nsample > sum(colors)")

        # Figure out the shape of the return array.
        if size is None:
            shape = (num_colors,)
        elif np.isscalar(size):
            shape = (size, num_colors)
        else:
            shape = tuple(size) + (num_colors,)
        variates = np.zeros(shape, dtype=np.int64)

        if num_colors == 0:
            return variates

        # One variate is a vector of length num_colors.
        num_variates = variates.size // num_colors
        variates_ptr = <int64_t *> np.PyArray_DATA(variates)

        if method == 'count':
            with self.lock, nogil:
                result = random_multivariate_hypergeometric_count(&self._bitgen,
                                        total, num_colors, colors_ptr, nsamp,
                                        num_variates, variates_ptr)
            if result == -1:
                raise MemoryError("Insufficient memory for multivariate_"
                                  "hypergeometric with method='count' and "
                                  "sum(colors)=%d" % total)
        else:
            with self.lock, nogil:
                random_multivariate_hypergeometric_marginals(&self._bitgen,
                                        total, num_colors, colors_ptr, nsamp,
                                        num_variates, variates_ptr)
        return variates

    def dirichlet(self, object alpha, size=None):
        """
        dirichlet(alpha, size=None)

        Draw samples from the Dirichlet distribution.

        Draw `size` samples of dimension k from a Dirichlet distribution. A
        Dirichlet-distributed random variable can be seen as a multivariate
        generalization of a Beta distribution. The Dirichlet distribution
        is a conjugate prior of a multinomial distribution in Bayesian
        inference.

        Parameters
        ----------
        alpha : sequence of floats, length k
            Parameter of the distribution (length ``k`` for sample of
            length ``k``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            vector of length ``k`` is returned.

        Returns
        -------
        samples : ndarray,
            The drawn samples, of shape ``(size, k)``.

        Raises
        ------
        ValueError
            If any value in ``alpha`` is less than zero

        Notes
        -----
        The Dirichlet distribution is a distribution over vectors
        :math:`x` that fulfil the conditions :math:`x_i>0` and
        :math:`\\sum_{i=1}^k x_i = 1`.

        The probability density function :math:`p` of a
        Dirichlet-distributed random vector :math:`X` is
        proportional to

        .. math:: p(x) \\propto \\prod_{i=1}^{k}{x^{\\alpha_i-1}_i},

        where :math:`\\alpha` is a vector containing the positive
        concentration parameters.

        The method uses the following property for computation: let :math:`Y`
        be a random vector which has components that follow a standard gamma
        distribution, then :math:`X = \\frac{1}{\\sum_{i=1}^k{Y_i}} Y`
        is Dirichlet-distributed

        References
        ----------
        .. [1] David McKay, "Information Theory, Inference and Learning
               Algorithms," chapter 23,
               https://www.inference.org.uk/mackay/itila/
        .. [2] Wikipedia, "Dirichlet distribution",
               https://en.wikipedia.org/wiki/Dirichlet_distribution

        Examples
        --------
        Taking an example cited in Wikipedia, this distribution can be used if
        one wanted to cut strings (each of initial length 1.0) into K pieces
        with different lengths, where each piece had, on average, a designated
        average length, but allowing some variation in the relative sizes of
        the pieces.

        >>> rng = np.random.default_rng()
        >>> s = rng.dirichlet((10, 5, 3), 20).transpose()

        >>> import matplotlib.pyplot as plt
        >>> plt.barh(range(20), s[0])
        >>> plt.barh(range(20), s[1], left=s[0], color='g')
        >>> plt.barh(range(20), s[2], left=s[0]+s[1], color='r')
        >>> plt.title("Lengths of Strings")

        """

        # =================
        # Pure python algo
        # =================
        # alpha   = N.atleast_1d(alpha)
        # k       = alpha.size

        # if n == 1:
        #     val = N.zeros(k)
        #     for i in range(k):
        #         val[i]   = sgamma(alpha[i], n)
        #     val /= N.sum(val)
        # else:
        #     val = N.zeros((k, n))
        #     for i in range(k):
        #         val[i]   = sgamma(alpha[i], n)
        #     val /= N.sum(val, axis = 0)
        #     val = val.T
        # return val

        cdef np.npy_intp k, totsize, i, j
        cdef np.ndarray alpha_arr, val_arr, alpha_csum_arr
        cdef double csum
        cdef double *alpha_data
        cdef double *alpha_csum_data
        cdef double *val_data
        cdef double acc, invacc, v

        k = len(alpha)
        alpha_arr = <np.ndarray>np.PyArray_FROMANY(
            alpha, np.NPY_DOUBLE, 1, 1,
            np.NPY_ARRAY_ALIGNED | np.NPY_ARRAY_C_CONTIGUOUS)
        if np.any(np.less(alpha_arr, 0)):
            raise ValueError('alpha < 0')

        alpha_data = <double*>np.PyArray_DATA(alpha_arr)

        if size is None:
            shape = (k,)
        else:
            try:
                shape = (operator.index(size), k)
            except:
                shape = tuple(size) + (k,)

        diric = np.zeros(shape, np.float64)
        val_arr = <np.ndarray>diric
        val_data= <double*>np.PyArray_DATA(val_arr)

        i = 0
        totsize = np.PyArray_SIZE(val_arr)

        # Select one of the following two algorithms for the generation
        #  of Dirichlet random variates (RVs)
        #
        # A) Small alpha case: Use the stick-breaking approach with beta
        #    random variates (RVs).
        # B) Standard case: Perform unit normalisation of a vector
        #    of gamma random variates
        #
        # A) prevents NaNs resulting from 0/0 that may occur in B)
        # when all values in the vector ':math:\\alpha' are smaller
        # than 1, then there is a nonzero probability that all
        # generated gamma RVs will be 0. When that happens, the
        # normalization process ends up computing 0/0, giving nan. A)
        # does not use divisions, so that a situation in which 0/0 has
        # to be computed cannot occur. A) is slower than B) as
        # generation of beta RVs is slower than generation of gamma
        # RVs. A) is selected whenever `alpha.max() < t`, where `t <
        # 1` is a threshold that controls the probability of
        # generating a NaN value when B) is used. For a given
        # threshold `t` this probability can be bounded by
        # `gammainc(t, d)` where `gammainc` is the regularized
        # incomplete gamma function and `d` is the smallest positive
        # floating point number that can be represented with a given
        # precision. For the chosen threshold `t=0.1` this probability
        # is smaller than `1.8e-31` for double precision floating
        # point numbers.

        if (k > 0) and (alpha_arr.max() < 0.1):
            # Small alpha case: Use stick-breaking approach with beta
            # random variates (RVs).
            # alpha_csum_data will hold the cumulative sum, right to
            # left, of alpha_arr.
            # Use a numpy array for memory management only.  We could just as
            # well have malloc'd alpha_csum_data.  alpha_arr is a C-contiguous
            # double array, therefore so is alpha_csum_arr.
            alpha_csum_arr = np.empty_like(alpha_arr)
            alpha_csum_data = <double*>np.PyArray_DATA(alpha_csum_arr)
            csum = 0.0
            for j in range(k - 1, -1, -1):
                csum += alpha_data[j]
                alpha_csum_data[j] = csum

            # If csum == 0, then all the values in alpha are 0, and there is
            # nothing to do, because diric was created with np.zeros().
            if csum > 0:
                with self.lock, nogil:
                    while i < totsize:
                        acc = 1.
                        for j in range(k - 1):
                            v = random_beta(&self._bitgen, alpha_data[j],
                                            alpha_csum_data[j + 1])
                            val_data[i + j] = acc * v
                            acc *= (1. - v)
                            if alpha_csum_data[j + 1] == 0:
                                # v must be 1, so acc is now 0. All
                                # remaining elements will be left at 0.
                                break
                        val_data[i + k - 1] = acc
                        i = i + k
        else:
            # Standard case: Unit normalisation of a vector of gamma random
            # variates
            with self.lock, nogil:
                while i < totsize:
                    acc = 0.
                    for j in range(k):
                        val_data[i + j] = random_standard_gamma(&self._bitgen,
                                                                alpha_data[j])
                        acc = acc + val_data[i + j]
                    invacc = 1. / acc
                    for j in range(k):
                        val_data[i + j] = val_data[i + j] * invacc
                    i = i + k

        return diric

    def permuted(self, object x, *, axis=None, out=None):
        """
        permuted(x, axis=None, out=None)

        Randomly permute `x` along axis `axis`.

        Unlike `shuffle`, each slice along the given axis is shuffled
        independently of the others.

        Parameters
        ----------
        x : array_like, at least one-dimensional
            Array to be shuffled.
        axis : int, optional
            Slices of `x` in this axis are shuffled. Each slice
            is shuffled independently of the others.  If `axis` is
            None, the flattened array is shuffled.
        out : ndarray, optional
            If given, this is the destination of the shuffled array.
            If `out` is None, a shuffled copy of the array is returned.

        Returns
        -------
        ndarray
            If `out` is None, a shuffled copy of `x` is returned.
            Otherwise, the shuffled array is stored in `out`,
            and `out` is returned

        See Also
        --------
        shuffle
        permutation
        
        Notes
        -----
        An important distinction between methods ``shuffle``  and ``permuted`` is 
        how they both treat the ``axis`` parameter which can be found at 
        :ref:`generator-handling-axis-parameter`.

        Examples
        --------
        Create a `numpy.random.Generator` instance:

        >>> rng = np.random.default_rng()

        Create a test array:

        >>> x = np.arange(24).reshape(3, 8)
        >>> x
        array([[ 0,  1,  2,  3,  4,  5,  6,  7],
               [ 8,  9, 10, 11, 12, 13, 14, 15],
               [16, 17, 18, 19, 20, 21, 22, 23]])

        Shuffle the rows of `x`:

        >>> y = rng.permuted(x, axis=1)
        >>> y
        array([[ 4,  3,  6,  7,  1,  2,  5,  0],  # random
               [15, 10, 14,  9, 12, 11,  8, 13],
               [17, 16, 20, 21, 18, 22, 23, 19]])

        `x` has not been modified:

        >>> x
        array([[ 0,  1,  2,  3,  4,  5,  6,  7],
               [ 8,  9, 10, 11, 12, 13, 14, 15],
               [16, 17, 18, 19, 20, 21, 22, 23]])

        To shuffle the rows of `x` in-place, pass `x` as the `out`
        parameter:

        >>> y = rng.permuted(x, axis=1, out=x)
        >>> x
        array([[ 3,  0,  4,  7,  1,  6,  2,  5],  # random
               [ 8, 14, 13,  9, 12, 11, 15, 10],
               [17, 18, 16, 22, 19, 23, 20, 21]])

        Note that when the ``out`` parameter is given, the return
        value is ``out``:

        >>> y is x
        True

        """

        cdef int ax
        cdef np.npy_intp axlen, axstride, itemsize
        cdef void *buf
        cdef np.flatiter it
        cdef np.ndarray to_shuffle
        cdef int status
        cdef int flags

        x = np.asarray(x)

        if out is None:
            out = x.copy(order='K')
        else:
            if type(out) is not np.ndarray:
                raise TypeError('out must be a numpy array')
            PyArray_FailUnlessWriteable(<np.PyArrayObject *>out, "out")
            if out.shape != x.shape:
                raise ValueError('out must have the same shape as x')
            np.copyto(out, x, casting='safe')

        if axis is None:
            if x.ndim > 1:
                if not (np.PyArray_FLAGS(out) & (np.NPY_ARRAY_C_CONTIGUOUS |
                                                 np.NPY_ARRAY_F_CONTIGUOUS)): 
                    flags = (np.NPY_ARRAY_C_CONTIGUOUS |
                             NPY_ARRAY_WRITEBACKIFCOPY)
                    to_shuffle = PyArray_FromArray(<np.PyArrayObject *>out,
                                                   <np.PyArray_Descr *>NULL, flags)
                    self.shuffle(to_shuffle.ravel(order='K'))
                    # Because we only execute this block if out is not
                    # contiguous, we know this call will always result in a
                    # copy of to_shuffle back to out. I.e. status will be 1.
                    status = PyArray_ResolveWritebackIfCopy(to_shuffle)
                    assert status == 1
                else:
                    # out is n-d with n > 1, but is either C- or F-contiguous,
                    # so we know out.ravel(order='A') is a view.
                    self.shuffle(out.ravel(order='A'))
            else:
                # out is 1-d
                self.shuffle(out)
            return out

        ax = normalize_axis_index(axis, np.ndim(out))
        itemsize = out.itemsize
        axlen = out.shape[ax]
        axstride = out.strides[ax]

        it = np.PyArray_IterAllButAxis(out, &ax)

        buf = PyMem_Malloc(itemsize)
        if buf == NULL:
            raise MemoryError('memory allocation failed in permuted')

        if out.dtype.hasobject:
            # Keep the GIL when shuffling an object array.
            with self.lock:
                while np.PyArray_ITER_NOTDONE(it):
                    _shuffle_raw_wrap(&self._bitgen, axlen, 0, itemsize,
                                      axstride,
                                      <char *>np.PyArray_ITER_DATA(it),
                                      <char *>buf)
                    np.PyArray_ITER_NEXT(it)
        else:
            # out is not an object array, so we can release the GIL.
            with self.lock, nogil:
                while np.PyArray_ITER_NOTDONE(it):
                    _shuffle_raw_wrap(&self._bitgen, axlen, 0, itemsize,
                                      axstride,
                                      <char *>np.PyArray_ITER_DATA(it),
                                      <char *>buf)
                    np.PyArray_ITER_NEXT(it)

        PyMem_Free(buf)
        return out

    def shuffle(self, object x, axis=0):
        """
        shuffle(x, axis=0)

        Modify an array or sequence in-place by shuffling its contents.

        The order of sub-arrays is changed but their contents remains the same.

        Parameters
        ----------
        x : ndarray or MutableSequence
            The array, list or mutable sequence to be shuffled.
        axis : int, optional
            The axis which `x` is shuffled along. Default is 0.
            It is only supported on `ndarray` objects.

        Returns
        -------
        None

        See Also
        --------
        permuted
        permutation

        Notes
        -----
        An important distinction between methods ``shuffle``  and ``permuted`` is 
        how they both treat the ``axis`` parameter which can be found at 
        :ref:`generator-handling-axis-parameter`.

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> arr = np.arange(10)
        >>> arr
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> rng.shuffle(arr)
        >>> arr
        array([2, 0, 7, 5, 1, 4, 8, 9, 3, 6]) # random

        >>> arr = np.arange(9).reshape((3, 3))
        >>> arr
        array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])
        >>> rng.shuffle(arr)
        >>> arr
        array([[3, 4, 5], # random
               [6, 7, 8],
               [0, 1, 2]])

        >>> arr = np.arange(9).reshape((3, 3))
        >>> arr
        array([[0, 1, 2],
               [3, 4, 5],
               [6, 7, 8]])
        >>> rng.shuffle(arr, axis=1)
        >>> arr
        array([[2, 0, 1], # random
               [5, 3, 4],
               [8, 6, 7]])

        """
        cdef:
            np.npy_intp i, j, n = len(x), stride, itemsize
            char* x_ptr
            char* buf_ptr

        if isinstance(x, np.ndarray):
            if not x.flags.writeable:
                raise ValueError('array is read-only')
            # Only call ndim on ndarrays, see GH 18142
            axis = normalize_axis_index(axis, np.ndim(x))

        if type(x) is np.ndarray and x.ndim == 1 and x.size:
            # Fast, statically typed path: shuffle the underlying buffer.
            # Only for non-empty, 1d objects of class ndarray (subclasses such
            # as MaskedArrays may not support this approach).
            x_ptr = np.PyArray_BYTES(x)
            stride = x.strides[0]
            itemsize = x.dtype.itemsize
            # As the array x could contain python objects we use a buffer
            # of bytes for the swaps to avoid leaving one of the objects
            # within the buffer and erroneously decrementing it's refcount
            # when the function exits.
            buf = np.empty(itemsize, dtype=np.int8)  # GC'd at function exit
            buf_ptr = np.PyArray_BYTES(buf)
            if x.dtype.hasobject:
                with self.lock:
                    _shuffle_raw_wrap(&self._bitgen, n, 1, itemsize, stride,
                                      x_ptr, buf_ptr)
            else:
                # Same as above, but the GIL is released.
                with self.lock, nogil:
                    _shuffle_raw_wrap(&self._bitgen, n, 1, itemsize, stride,
                                      x_ptr, buf_ptr)
        elif isinstance(x, np.ndarray):
            if x.size == 0:
                # shuffling is a no-op
                return

            x = np.swapaxes(x, 0, axis)
            buf = np.empty_like(x[0, ...])
            with self.lock:
                for i in reversed(range(1, len(x))):
                    j = random_interval(&self._bitgen, i)
                    if i == j:
                        # i == j is not needed and memcpy is undefined.
                        continue
                    buf[...] = x[j, ...]
                    x[j, ...] = x[i, ...]
                    x[i, ...] = buf
        else:
            # Untyped path.
            if not isinstance(x, Sequence):
                # See gh-18206. We may decide to deprecate here in the future.
                warnings.warn(
                    f"you are shuffling a '{type(x).__name__}' object "
                    "which is not a subclass of 'Sequence'; "
                    "`shuffle` is not guaranteed to behave correctly. "
                    "E.g., non-numpy array/tensor objects with view semantics "
                    "may contain duplicates after shuffling.",
                    UserWarning, stacklevel=1)  # Cython does not add a level

            if axis != 0:
                raise NotImplementedError("Axis argument is only supported "
                                          "on ndarray objects")
            with self.lock:
                for i in reversed(range(1, n)):
                    j = random_interval(&self._bitgen, i)
                    x[i], x[j] = x[j], x[i]

    def permutation(self, object x, axis=0):
        """
        permutation(x, axis=0)

        Randomly permute a sequence, or return a permuted range.

        Parameters
        ----------
        x : int or array_like
            If `x` is an integer, randomly permute ``np.arange(x)``.
            If `x` is an array, make a copy and shuffle the elements
            randomly.
        axis : int, optional
            The axis which `x` is shuffled along. Default is 0.

        Returns
        -------
        out : ndarray
            Permuted sequence or array range.

        Examples
        --------
        >>> rng = np.random.default_rng()
        >>> rng.permutation(10)
        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random

        >>> rng.permutation([1, 4, 9, 12, 15])
        array([15,  1,  9,  4, 12]) # random

        >>> arr = np.arange(9).reshape((3, 3))
        >>> rng.permutation(arr)
        array([[6, 7, 8], # random
               [0, 1, 2],
               [3, 4, 5]])

        >>> rng.permutation("abc")
        Traceback (most recent call last):
            ...
        numpy.exceptions.AxisError: axis 0 is out of bounds for array of dimension 0

        >>> arr = np.arange(9).reshape((3, 3))
        >>> rng.permutation(arr, axis=1)
        array([[0, 2, 1], # random
               [3, 5, 4],
               [6, 8, 7]])

        """
        if isinstance(x, (int, np.integer)):
            arr = np.arange(x)
            self.shuffle(arr)
            return arr

        arr = np.asarray(x)

        axis = normalize_axis_index(axis, arr.ndim)

        # shuffle has fast-path for 1-d
        if arr.ndim == 1:
            # Return a copy if same memory
            if np.may_share_memory(arr, x):
                arr = np.array(arr)
            self.shuffle(arr)
            return arr

        # Shuffle index array, dtype to ensure fast path
        idx = np.arange(arr.shape[axis], dtype=np.intp)
        self.shuffle(idx)
        slices = [slice(None)]*arr.ndim
        slices[axis] = idx
        return arr[tuple(slices)]


@cython.embedsignature(True)
def default_rng(seed=None):
    """Construct a new Generator with the default BitGenerator (PCG64).

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator, RandomState}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then all values must be non-negative and will be
        passed to `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.
        Additionally, when passed a `BitGenerator`, it will be wrapped by
        `Generator`. If passed a `Generator`, it will be returned unaltered.
        When passed a legacy `RandomState` instance it will be coerced to a `Generator`.

    Returns
    -------
    Generator
        The initialized generator object.

    Notes
    -----
    If ``seed`` is not a `BitGenerator` or a `Generator`, a new `BitGenerator`
    is instantiated. This function does not manage a default global instance.

    See :ref:`seeding_and_entropy` for more information about seeding.

    Examples
    --------
    `default_rng` is the recommended constructor for the random number class
    `Generator`. Here are several ways we can construct a random 
    number generator using `default_rng` and the `Generator` class. 

    Here we use `default_rng` to generate a random float:
 
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> print(rng)
    Generator(PCG64)
    >>> rfloat = rng.random()
    >>> rfloat
    0.22733602246716966
    >>> type(rfloat)
    <class 'float'>
     
    Here we use `default_rng` to generate 3 random integers between 0 
    (inclusive) and 10 (exclusive):
        
    >>> import numpy as np
    >>> rng = np.random.default_rng(12345)
    >>> rints = rng.integers(low=0, high=10, size=3)
    >>> rints
    array([6, 2, 7])
    >>> type(rints[0])
    <class 'numpy.int64'>
    
    Here we specify a seed so that we have reproducible results:
    
    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> print(rng)
    Generator(PCG64)
    >>> arr1 = rng.random((3, 3))
    >>> arr1
    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235],
           [0.7611397 , 0.78606431, 0.12811363]])

    If we exit and restart our Python interpreter, we'll see that we
    generate the same random numbers again:

    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=42)
    >>> arr2 = rng.random((3, 3))
    >>> arr2
    array([[0.77395605, 0.43887844, 0.85859792],
           [0.69736803, 0.09417735, 0.97562235],
           [0.7611397 , 0.78606431, 0.12811363]])

    """
    if _check_bit_generator(seed):
        # We were passed a BitGenerator, so just wrap it up.
        return Generator(seed)
    elif isinstance(seed, Generator):
        # Pass through a Generator.
        return seed
    elif isinstance(seed, np.random.RandomState):
        gen = np.random.Generator(seed._bit_generator)
        return gen

    # Otherwise we need to instantiate a new BitGenerator and Generator as
    # normal.
    return Generator(PCG64(seed))


default_rng.__module__ = "numpy.random"
