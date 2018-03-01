from libc.stdint cimport uint64_t, uint32_t
import numpy as np
cimport numpy as np
from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from common cimport *
cimport common
try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock

from core_prng.splitmix64 import SplitMix64

np.import_array()

cdef extern from "src/distributions/distributions.h":
    double random_double(void *void_state) nogil
    float random_float(void *void_state) nogil
    uint32_t random_uint32(void *void_state) nogil
    double random_standard_exponential(void *void_state) nogil

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
            prng = SplitMix64()
        self.__core_prng = prng

        capsule = prng._prng_capsule
        cdef const char *anon_name = "CorePRNG"
        if not PyCapsule_IsValid(capsule, anon_name):
            raise ValueError("Invalid pointer to anon_func_state")
        self._prng = <prng_t *>PyCapsule_GetPointer(capsule, anon_name)
        self.lock = Lock()

    @property
    def state(self):
        """Get or set the underlying PRNG's state"""
        return self.__core_prng.state

    @state.setter
    def state(self, value):
        self.__core_prng.state = value

    def random_integer(self, bits=64):
        #print("In random_integer")
        if bits==64:
            return self._prng.next_uint64(self._prng.state)
        elif bits==32:
            return random_uint32(self._prng)
        else:
            raise ValueError('bits must be 32 or 64')

    def random_double(self, bits=64):
        if bits==64:
            return self._prng.next_double(self._prng.state)
        elif bits==32:
            return random_float(self._prng)
        else:
            raise ValueError('bits must be 32 or 64')

    def random_sample(self, size=None, dtype=np.float64):
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
            return double_fill(&random_double, self._prng, size, self.lock)
        elif key == 'float32':
            return float_fill(&random_float, self._prng, size, self.lock)
        else:
            raise TypeError('Unsupported dtype "%s" for random_sample' % key)


    def standard_exponential(self, size=None, dtype=np.float64):
        """
        standard_exponential(size=None, dtype='d', method='inv', out=None)

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
            return double_fill(&random_standard_exponential, self._prng, size, self.lock)
        elif key == 'float32':
            return float_fill_from_double(&random_standard_exponential, self._prng, size, self.lock)
        else:
            raise TypeError('Unsupported dtype "%s" for standard_exponential'
                            % key)
