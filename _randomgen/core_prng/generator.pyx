import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from common cimport *
from libc.stdlib cimport malloc, free

cimport numpy as np
import numpy as np
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer

from common cimport *

try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock

from core_prng.xoroshiro128 import Xoroshiro128
import core_prng.pickle

np.import_array()

cdef extern from "src/distributions/distributions.h":
    double random_sample(prng_t *prng_state) nogil
    double random_standard_exponential(prng_t *prng_state) nogil
    double random_standard_exponential_zig(prng_t *prng_state) nogil
    double random_gauss(prng_t *prng_state) nogil
    double random_gauss_zig(prng_t* prng_state) nogil
    double random_standard_gamma(prng_t *prng_state, double shape) nogil
    double random_standard_gamma_zig(prng_t *prng_state, double shape) nogil

    float random_sample_f(prng_t *prng_state) nogil
    float random_standard_exponential_f(prng_t *prng_state) nogil
    float random_standard_exponential_zig_f(prng_t *prng_state) nogil
    float random_gauss_f(prng_t *prng_state) nogil
    float random_gauss_zig_f(prng_t* prng_state) nogil
    float random_standard_gamma_f(prng_t *prng_state, float shape) nogil
    float random_standard_gamma_zig_f(prng_t *prng_state, float shape) nogil

cdef class RandomGenerator:
    """
    Prototype Random Generator that consumes randoms from a CorePRNG class

    Parameters
    ----------
    prng : CorePRNG, optional
        Object exposing a PyCapsule containing state and function pointers

    Examples
    --------
    >>> from core_prng.generator import RandomGenerator
    >>> rg = RandomGenerator()
    >>> rg.random_integer()
    """
    cdef public object __core_prng
    cdef prng_t *_prng
    cdef object lock

    def __init__(self, prng=None):
        if prng is None:
            prng = Xoroshiro128()
        self.__core_prng = prng

        capsule = prng._prng_capsule
        cdef const char *anon_name = "CorePRNG"
        if not PyCapsule_IsValid(capsule, anon_name):
            raise ValueError("Invalid pointer to anon_func_state")
        self._prng = <prng_t *> PyCapsule_GetPointer(capsule, anon_name)
        self.lock = Lock()
        with self.lock:
            self._prng.has_gauss = 0
            self._prng.has_gauss_f = 0

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        return (core_prng.pickle.__generator_ctor,
                (self.state['prng'],),
                self.state)

    @property
    def state(self):
        """Get or set the underlying PRNG's state"""
        state = self.__core_prng.state
        state['has_gauss'] = self._prng.has_gauss
        state['has_gauss_f'] = self._prng.has_gauss_f
        state['gauss'] = self._prng.gauss
        state['gauss_f'] = self._prng.gauss_f
        return state

    @state.setter
    def state(self, value):
        self._prng.has_gauss = value['has_gauss']
        self._prng.has_gauss_f = value['has_gauss_f']
        self._prng.gauss = value['gauss']
        self._prng.gauss_f = value['gauss_f']
        self.__core_prng.state = value

    def random_integer(self, bits=64):
        #print("In random_integer")
        if bits == 64:
            return self._prng.next_uint64(self._prng.state)
        elif bits == 32:
            return self._prng.next_uint32(self._prng.state)
        else:
            raise ValueError('bits must be 32 or 64')

    def random_double(self, bits=64):
        if bits == 64:
            return self._prng.next_double(self._prng.state)
        elif bits == 32:
            return random_sample_f(self._prng)
        else:
            raise ValueError('bits must be 32 or 64')

    def random_sample(self, size=None, dtype=np.float64, out=None):
        """
        random_sample(size=None, dtype='d', out=None)

        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random_sample` by `(b-a)` and add `a`::

          (b - a) * random_sample() + a

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : {str, dtype}, optional
            Desired dtype of the result, either 'd' (or 'float64') or 'f'
            (or 'float32'). All dtypes are determined by their name. The
            default value is 'd'.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is not None,
            it must have the same shape as the provided size and must match the type of
            the output values.

        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned).

        Examples
        --------
        >>> np.random.random_sample()
        0.47108547995356098
        >>> type(np.random.random_sample())
        <type 'float'>
        >>> np.random.random_sample((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428])

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * np.random.random_sample((3, 2)) - 5
        array([[-3.99149989, -0.52338984],
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])
        """
        cdef double temp
        key = np.dtype(dtype).name
        if key == 'float64':
            return double_fill(&random_sample, self._prng, size, self.lock, out)
        elif key == 'float32':
            return float_fill(&random_sample_f, self._prng, size, self.lock, out)
        else:
            raise TypeError('Unsupported dtype "%s" for random_sample' % key)

    def standard_exponential(self, size=None, dtype=np.float64, method=u'zig', out=None):
        """
        standard_exponential(size=None, dtype='d', method='zig', out=None)

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
            Desired dtype of the result, either 'd' (or 'float64') or 'f'
            (or 'float32'). All dtypes are determined by their name. The
            default value is 'd'.
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

        >>> n = np.random.standard_exponential((3, 8000))
        """
        key = np.dtype(dtype).name
        if key == 'float64':
            if method == u'zig':
                return double_fill(&random_standard_exponential_zig, self._prng, size, self.lock, out)
            else:
                return double_fill(&random_standard_exponential, self._prng, size, self.lock, out)
        elif key == 'float32':
            if method == u'zig':
                return float_fill(&random_standard_exponential_zig_f, self._prng, size, self.lock, out)
            else:
                return float_fill(&random_standard_exponential_f, self._prng, size, self.lock, out)
        else:
            raise TypeError('Unsupported dtype "%s" for standard_exponential'
                            % key)

    # Complicated, continuous distributions:
    def standard_normal(self, size=None, dtype=np.float64, method=u'zig', out=None):
        """
        standard_normal(size=None, dtype='d', method='zig', out=None)

        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        dtype : {str, dtype}, optional
            Desired dtype of the result, either 'd' (or 'float64') or 'f'
            (or 'float32'). All dtypes are determined by their name. The
            default value is 'd'.
        method : str, optional
            Either 'bm' or 'zig'. 'bm' uses the Box-Muller transformations
            method.  'zig' uses the much faster Ziggurat method of Marsaglia and Tsang.
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
        >>> s = np.random.standard_normal(8000)
        >>> s
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311, #random
               -0.38672696, -0.4685006 ])                               #random
        >>> s.shape
        (8000,)
        >>> s = np.random.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        """
        key = np.dtype(dtype).name
        if key == 'float64':
            if method == u'zig':
                return double_fill(&random_gauss_zig, self._prng, size, self.lock, out)
            else:
                return double_fill(&random_gauss, self._prng, size, self.lock, out)
        elif key == 'float32':
            if method == u'zig':
                return float_fill(&random_gauss_zig_f, self._prng, size, self.lock, out)
            else:
                return float_fill(&random_gauss_f, self._prng, size, self.lock, out)
        else:
            raise TypeError('Unsupported dtype "%s" for standard_normal' % key)


    def standard_gamma(self, shape, size=None, dtype=np.float64, method='zig',
                       out=None):
        """
        standard_gamma(shape, size=None, dtype='d', method='inv', out=None)

        Draw samples from a standard Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,
        shape (sometimes designated "k") and scale=1.

        Parameters
        ----------
        shape : float or array_like of floats
            Parameter, should be > 0.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``shape`` is a scalar.  Otherwise,
            ``np.array(shape).size`` samples are drawn.
        dtype : {str, dtype}, optional
            Desired dtype of the result, either 'd' (or 'float64') or 'f'
            (or 'float32'). All dtypes are determined by their name. The
            default value is 'd'.
        method : str, optional
            Either 'inv' or 'zig'. 'inv' uses the default inverse CDF method.
            'zig' uses the much faster Ziggurat method of Marsaglia and Tsang.
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
               http://mathworld.wolfram.com/GammaDistribution.html
        .. [2] Wikipedia, "Gamma distribution",
               http://en.wikipedia.org/wiki/Gamma_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> shape, scale = 2., 1. # mean and width
        >>> s = np.random.standard_gamma(shape, 1000000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps
        >>> count, bins, ignored = plt.hist(s, 50, normed=True)
        >>> y = bins**(shape-1) * ((np.exp(-bins/scale))/ \\
        ...                       (sps.gamma(shape) * scale**shape))
        >>> plt.plot(bins, y, linewidth=2, color='r')
        >>> plt.show()
        """
        cdef void *func
        if method != u'zig' and method != u'inv':
            raise ValueError("method must be either 'inv' or 'zig'")
        key = np.dtype(dtype).name
        if key == 'float64':
            if method == 'inv':
                func = <void *>&random_standard_gamma
            else:
                func = <void *>&random_standard_gamma_zig
            return cont(func, self._prng, size, self.lock, 1,
                        shape, 'shape', CONS_NON_NEGATIVE,
                        0.0, '', CONS_NONE,
                        0.0, '', CONS_NONE,
                        out)
        if key == 'float32':
            if method == 'inv':
                func = <void *>&random_standard_gamma_f
            else:
                func = <void *>&random_standard_gamma_zig_f
            return cont_f(func, self._prng, size, self.lock,
                          shape, 'shape', CONS_NON_NEGATIVE,
                          out)
        else:
            raise TypeError('Unsupported dtype "%s" for standard_gamma' % key)
