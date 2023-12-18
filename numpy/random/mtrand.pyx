#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True, language_level=3, binding=False
import operator
import warnings
from collections.abc import Sequence

import numpy as np

from cpython.pycapsule cimport PyCapsule_IsValid, PyCapsule_GetPointer
from cpython cimport (Py_INCREF, PyFloat_AsDouble)
cimport cython
cimport numpy as np

from libc cimport string
from libc.stdint cimport int64_t, uint64_t
from ._bounded_integers cimport (_rand_bool, _rand_int32, _rand_int64,
         _rand_int16, _rand_int8, _rand_uint64, _rand_uint32, _rand_uint16,
         _rand_uint8,)
from ._mt19937 import MT19937 as _MT19937
from numpy.random cimport bitgen_t
from ._common cimport (POISSON_LAM_MAX, CONS_POSITIVE, CONS_NONE,
            CONS_NON_NEGATIVE, CONS_BOUNDED_0_1, CONS_BOUNDED_GT_0_1,
            CONS_BOUNDED_LT_0_1, CONS_GTE_1, CONS_GT_1, LEGACY_CONS_POISSON,
            double_fill, cont, kahan_sum, cont_broadcast_3,
            check_array_constraint, check_constraint, disc, discrete_broadcast_iii,
            validate_output_shape
        )

cdef extern from "numpy/random/distributions.h":
    struct s_binomial_t:
        int has_binomial
        double psave
        int64_t nsave
        double r
        double q
        double fm
        int64_t m
        double p1
        double xm
        double xl
        double xr
        double c
        double laml
        double lamr
        double p2
        double p3
        double p4

    ctypedef s_binomial_t binomial_t

    void random_standard_uniform_fill(bitgen_t* bitgen_state, np.npy_intp cnt, double *out) nogil
    int64_t random_positive_int(bitgen_t *bitgen_state) nogil
    double random_uniform(bitgen_t *bitgen_state, double lower, double range) nogil
    double random_laplace(bitgen_t *bitgen_state, double loc, double scale) nogil
    double random_gumbel(bitgen_t *bitgen_state, double loc, double scale) nogil
    double random_logistic(bitgen_t *bitgen_state, double loc, double scale) nogil
    double random_rayleigh(bitgen_t *bitgen_state, double mode) nogil
    double random_triangular(bitgen_t *bitgen_state, double left, double mode,
                                 double right) nogil
    uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max) nogil

cdef extern from "include/legacy-distributions.h":
    struct aug_bitgen:
        bitgen_t *bit_generator
        int has_gauss
        double gauss

    ctypedef aug_bitgen aug_bitgen_t

    double legacy_gauss(aug_bitgen_t *aug_state) nogil
    double legacy_pareto(aug_bitgen_t *aug_state, double a) nogil
    double legacy_weibull(aug_bitgen_t *aug_state, double a) nogil
    double legacy_standard_gamma(aug_bitgen_t *aug_state, double shape) nogil
    double legacy_normal(aug_bitgen_t *aug_state, double loc, double scale) nogil
    double legacy_standard_t(aug_bitgen_t *aug_state, double df) nogil

    double legacy_standard_exponential(aug_bitgen_t *aug_state) nogil
    double legacy_power(aug_bitgen_t *aug_state, double a) nogil
    double legacy_gamma(aug_bitgen_t *aug_state, double shape, double scale) nogil
    double legacy_power(aug_bitgen_t *aug_state, double a) nogil
    double legacy_chisquare(aug_bitgen_t *aug_state, double df) nogil
    double legacy_rayleigh(aug_bitgen_t *aug_state, double mode) nogil
    double legacy_noncentral_chisquare(aug_bitgen_t *aug_state, double df,
                                    double nonc) nogil
    double legacy_noncentral_f(aug_bitgen_t *aug_state, double dfnum, double dfden,
                            double nonc) nogil
    double legacy_wald(aug_bitgen_t *aug_state, double mean, double scale) nogil
    double legacy_lognormal(aug_bitgen_t *aug_state, double mean, double sigma) nogil
    int64_t legacy_random_binomial(bitgen_t *bitgen_state, double p,
                                   int64_t n, binomial_t *binomial) nogil
    int64_t legacy_negative_binomial(aug_bitgen_t *aug_state, double n, double p) nogil
    int64_t legacy_random_hypergeometric(bitgen_t *bitgen_state, int64_t good, int64_t bad, int64_t sample) nogil
    int64_t legacy_logseries(bitgen_t *bitgen_state, double p) nogil
    int64_t legacy_random_poisson(bitgen_t *bitgen_state, double lam) nogil
    int64_t legacy_random_zipf(bitgen_t *bitgen_state, double a) nogil
    int64_t legacy_random_geometric(bitgen_t *bitgen_state, double p) nogil
    void legacy_random_multinomial(bitgen_t *bitgen_state, long n, long *mnix, double *pix, np.npy_intp d, binomial_t *binomial) nogil
    double legacy_standard_cauchy(aug_bitgen_t *state) nogil
    double legacy_beta(aug_bitgen_t *aug_state, double a, double b) nogil
    double legacy_f(aug_bitgen_t *aug_state, double dfnum, double dfden) nogil
    double legacy_exponential(aug_bitgen_t *aug_state, double scale) nogil
    double legacy_power(aug_bitgen_t *state, double a) nogil
    double legacy_vonmises(bitgen_t *bitgen_state, double mu, double kappa) nogil

np.import_array()

cdef object int64_to_long(object x):
    """
    Convert int64 to long for legacy compatibility, which used long for integer
    distributions
    """
    cdef int64_t x64

    if np.isscalar(x):
        x64 = x
        return <long>x64
    return x.astype('l', casting='unsafe')


cdef class RandomState:
    """
    RandomState(seed=None)

    Container for the slow Mersenne Twister pseudo-random number generator.
    Consider using a different BitGenerator with the Generator container
    instead.

    `RandomState` and `Generator` expose a number of methods for generating
    random numbers drawn from a variety of probability distributions. In
    addition to the distribution-specific arguments, each method takes a
    keyword argument `size` that defaults to ``None``. If `size` is ``None``,
    then a single value is generated and returned. If `size` is an integer,
    then a 1-D array filled with generated values is returned. If `size` is a
    tuple, then an array with that shape is filled and returned.

    **Compatibility Guarantee**

    A fixed bit generator using a fixed seed and a fixed series of calls to
    'RandomState' methods using the same parameters will always produce the
    same results up to roundoff error except when the values were incorrect.
    `RandomState` is effectively frozen and will only receive updates that
    are required by changes in the internals of Numpy. More substantial
    changes, including algorithmic improvements, are reserved for
    `Generator`.

    Parameters
    ----------
    seed : {None, int, array_like, BitGenerator}, optional
        Random seed used to initialize the pseudo-random number generator or
        an instantized BitGenerator.  If an integer or array, used as a seed for
        the MT19937 BitGenerator. Values can be any integer between 0 and
        2**32 - 1 inclusive, an array (or other sequence) of such integers,
        or ``None`` (the default).  If `seed` is ``None``, then the `MT19937`
        BitGenerator is initialized by reading data from ``/dev/urandom``
        (or the Windows analogue) if available or seed from the clock
        otherwise.

    Notes
    -----
    The Python stdlib module "random" also contains a Mersenne Twister
    pseudo-random number generator with a number of methods that are similar
    to the ones available in `RandomState`. `RandomState`, besides being
    NumPy-aware, has the advantage that it provides a much larger number
    of probability distributions to choose from.

    See Also
    --------
    Generator
    MT19937
    numpy.random.BitGenerator

    """
    cdef public object _bit_generator
    cdef bitgen_t _bitgen
    cdef aug_bitgen_t _aug_state
    cdef binomial_t _binomial
    cdef object lock
    _poisson_lam_max = POISSON_LAM_MAX

    def __init__(self, seed=None):
        if seed is None:
            bit_generator = _MT19937()
        elif not hasattr(seed, 'capsule'):
            bit_generator = _MT19937()
            bit_generator._legacy_seeding(seed)
        else:
            bit_generator = seed

        self._initialize_bit_generator(bit_generator)

    def __repr__(self):
        return self.__str__() + ' at 0x{:X}'.format(id(self))

    def __str__(self):
        _str = self.__class__.__name__
        _str += '(' + self._bit_generator.__class__.__name__ + ')'
        return _str

    # Pickling support:
    def __getstate__(self):
        return self.get_state(legacy=False)

    def __setstate__(self, state):
        self.set_state(state)

    def __reduce__(self):
        ctor, name_tpl, _ = self._bit_generator.__reduce__()

        from ._pickle import __randomstate_ctor
        return __randomstate_ctor, (name_tpl[0], ctor), self.get_state(legacy=False)

    cdef _initialize_bit_generator(self, bit_generator):
        self._bit_generator = bit_generator
        capsule = bit_generator.capsule
        cdef const char *name = "BitGenerator"
        if not PyCapsule_IsValid(capsule, name):
            raise ValueError("Invalid bit generator. The bit generator must "
                             "be instantized.")
        self._bitgen = (<bitgen_t *> PyCapsule_GetPointer(capsule, name))[0]
        self._aug_state.bit_generator = &self._bitgen
        self._reset_gauss()
        self.lock = bit_generator.lock

    cdef _reset_gauss(self):
        self._aug_state.has_gauss = 0
        self._aug_state.gauss = 0.0

    def seed(self, seed=None):
        """
        seed(seed=None)

        Reseed a legacy MT19937 BitGenerator

        Notes
        -----
        This is a convenience, legacy function.

        The best practice is to **not** reseed a BitGenerator, rather to
        recreate a new one. This method is here for legacy reasons.
        This example demonstrates best practice.

        >>> from numpy.random import MT19937
        >>> from numpy.random import RandomState, SeedSequence
        >>> rs = RandomState(MT19937(SeedSequence(123456789)))
        # Later, you want to restart the stream
        >>> rs = RandomState(MT19937(SeedSequence(987654321)))
        """
        if not isinstance(self._bit_generator, _MT19937):
            raise TypeError('can only re-seed a MT19937 BitGenerator')
        self._bit_generator._legacy_seeding(seed)
        self._reset_gauss()

    def get_state(self, legacy=True):
        """
        get_state(legacy=True)

        Return a tuple representing the internal state of the generator.

        For more details, see `set_state`.

        Parameters
        ----------
        legacy : bool, optional
            Flag indicating to return a legacy tuple state when the BitGenerator
            is MT19937, instead of a dict. Raises ValueError if the underlying
            bit generator is not an instance of MT19937.

        Returns
        -------
        out : {tuple(str, ndarray of 624 uints, int, int, float), dict}
            If legacy is True, the returned tuple has the following items:

            1. the string 'MT19937'.
            2. a 1-D array of 624 unsigned integer keys.
            3. an integer ``pos``.
            4. an integer ``has_gauss``.
            5. a float ``cached_gaussian``.

            If `legacy` is False, or the BitGenerator is not MT19937, then
            state is returned as a dictionary.

        See Also
        --------
        set_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in NumPy. If the internal state is manually altered,
        the user should know exactly what he/she is doing.

        """
        st = self._bit_generator.state
        if st['bit_generator'] != 'MT19937' and legacy:
            warnings.warn('get_state and legacy can only be used with the '
                          'MT19937 BitGenerator. To silence this warning, '
                          'set `legacy` to False.', RuntimeWarning)
            legacy = False
        st['has_gauss'] = self._aug_state.has_gauss
        st['gauss'] = self._aug_state.gauss
        if legacy and not isinstance(self._bit_generator, _MT19937):
            raise ValueError(
                "legacy can only be True when the underlyign bitgenerator is "
                "an instance of MT19937."
            )
        if legacy:
            return (st['bit_generator'], st['state']['key'], st['state']['pos'],
                    st['has_gauss'], st['gauss'])
        return st

    def set_state(self, state):
        """
        set_state(state)

        Set the internal state of the generator from a tuple.

        For use if one has reason to manually (re-)set the internal state of
        the bit generator used by the RandomState instance. By default,
        RandomState uses the "Mersenne Twister"[1]_ pseudo-random number
        generating algorithm.

        Parameters
        ----------
        state : {tuple(str, ndarray of 624 uints, int, int, float), dict}
            The `state` tuple has the following items:

            1. the string 'MT19937', specifying the Mersenne Twister algorithm.
            2. a 1-D array of 624 unsigned integers ``keys``.
            3. an integer ``pos``.
            4. an integer ``has_gauss``.
            5. a float ``cached_gaussian``.

            If state is a dictionary, it is directly set using the BitGenerators
            `state` property.

        Returns
        -------
        out : None
            Returns 'None' on success.

        See Also
        --------
        get_state

        Notes
        -----
        `set_state` and `get_state` are not needed to work with any of the
        random distributions in NumPy. If the internal state is manually altered,
        the user should know exactly what he/she is doing.

        For backwards compatibility, the form (str, array of 624 uints, int) is
        also accepted although it is missing some information about the cached
        Gaussian value: ``state = ('MT19937', keys, pos)``.

        References
        ----------
        .. [1] M. Matsumoto and T. Nishimura, "Mersenne Twister: A
           623-dimensionally equidistributed uniform pseudorandom number
           generator," *ACM Trans. on Modeling and Computer Simulation*,
           Vol. 8, No. 1, pp. 3-30, Jan. 1998.

        """
        if isinstance(state, dict):
            if 'bit_generator' not in state or 'state' not in state:
                raise ValueError('state dictionary is not valid.')
            st = state
        else:
            if not isinstance(state, (tuple, list)):
                raise TypeError('state must be a dict or a tuple.')
            with cython.boundscheck(True):
                if state[0] != 'MT19937':
                    raise ValueError('set_state can only be used with legacy '
                                     'MT19937 state instances.')
                st = {'bit_generator': state[0],
                      'state': {'key': state[1], 'pos': state[2]}}
                if len(state) > 3:
                    st['has_gauss'] = state[3]
                    st['gauss'] = state[4]
                    value = st

        self._aug_state.gauss = st.get('gauss', 0.0)
        self._aug_state.has_gauss = st.get('has_gauss', 0)
        self._bit_generator.state = st

    def random_sample(self, size=None):
        """
        random_sample(size=None)

        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval.  To sample :math:`Unif[a, b), b > a` multiply
        the output of `random_sample` by `(b-a)` and add `a`::

          (b - a) * random_sample() + a

        .. note::
            New code should use the `~numpy.random.Generator.random`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray of floats
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned).

        See Also
        --------
        random.Generator.random: which should be used for new code.

        Examples
        --------
        >>> np.random.random_sample()
        0.47108547995356098 # random
        >>> type(np.random.random_sample())
        <class 'float'>
        >>> np.random.random_sample((5,))
        array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428]) # random

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * np.random.random_sample((3, 2)) - 5
        array([[-3.99149989, -0.52338984], # random
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])

        """
        cdef double temp
        return double_fill(&random_standard_uniform_fill, &self._bitgen, size, self.lock, None)

    def random(self, size=None):
        """
        random(size=None)

        Return random floats in the half-open interval [0.0, 1.0). Alias for
        `random_sample` to ease forward-porting to the new random API.
        """
        return self.random_sample(size=size)

    def beta(self, a, b, size=None):
        """
        beta(a, b, size=None)

        Draw samples from a Beta distribution.

        The Beta distribution is a special case of the Dirichlet distribution,
        and is related to the Gamma distribution.  It has the probability
        distribution function

        .. math:: f(x; a,b) = \\frac{1}{B(\\alpha, \\beta)} x^{\\alpha - 1}
                                                         (1 - x)^{\\beta - 1},

        where the normalization, B, is the beta function,

        .. math:: B(\\alpha, \\beta) = \\int_0^1 t^{\\alpha - 1}
                                     (1 - t)^{\\beta - 1} dt.

        It is often seen in Bayesian inference and order statistics.

        .. note::
            New code should use the `~numpy.random.Generator.beta`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.


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

        See Also
        --------
        random.Generator.beta: which should be used for new code.
        """
        return cont(&legacy_beta, &self._aug_state, size, self.lock, 2,
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

        .. note::
            New code should use the `~numpy.random.Generator.exponential`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        A real world example: Assume a company has 10000 customer support 
        agents and the average time between customer calls is 4 minutes.

        >>> n = 10000
        >>> time_between_calls = np.random.default_rng().exponential(scale=4, size=n)

        What is the probability that a customer will call in the next 
        4 to 5 minutes? 
        
        >>> x = ((time_between_calls < 5).sum())/n 
        >>> y = ((time_between_calls < 4).sum())/n
        >>> x-y
        0.08 # may vary

        See Also
        --------
        random.Generator.exponential: which should be used for new code.

        References
        ----------
        .. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
               Random Signal Principles", 4th ed, 2001, p. 57.
        .. [2] Wikipedia, "Poisson process",
               https://en.wikipedia.org/wiki/Poisson_process
        .. [3] Wikipedia, "Exponential distribution",
               https://en.wikipedia.org/wiki/Exponential_distribution

        """
        return cont(&legacy_exponential, &self._aug_state, size, self.lock, 1,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE,
                    None)

    def standard_exponential(self, size=None):
        """
        standard_exponential(size=None)

        Draw samples from the standard exponential distribution.

        `standard_exponential` is identical to the exponential distribution
        with a scale parameter of 1.

        .. note::
            New code should use the
            `~numpy.random.Generator.standard_exponential`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : float or ndarray
            Drawn samples.

        See Also
        --------
        random.Generator.standard_exponential: which should be used for new code.

        Examples
        --------
        Output a 3x8000 array:

        >>> n = np.random.standard_exponential((3, 8000))

        """
        return cont(&legacy_standard_exponential, &self._aug_state, size, self.lock, 0,
                    None, None, CONS_NONE,
                    None, None, CONS_NONE,
                    None, None, CONS_NONE,
                    None)

    def tomaxint(self, size=None):
        """
        tomaxint(size=None)

        Return a sample of uniformly distributed random integers in the interval
        [0, ``np.iinfo(np.int_).max``]. The `np.int_` type translates to the C long
        integer type and its precision is platform dependent.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            Drawn samples, with shape `size`.

        See Also
        --------
        randint : Uniform sampling over a given half-open interval of integers.
        random_integers : Uniform sampling over a given closed interval of
            integers.

        Examples
        --------
        >>> rs = np.random.RandomState() # need a RandomState object
        >>> rs.tomaxint((2,2,2))
        array([[[1170048599, 1600360186], # random
                [ 739731006, 1947757578]],
               [[1871712945,  752307660],
                [1601631370, 1479324245]]])
        >>> rs.tomaxint((2,2,2)) < np.iinfo(np.int_).max
        array([[[ True,  True],
                [ True,  True]],
               [[ True,  True],
                [ True,  True]]])

        """
        cdef np.npy_intp n
        cdef np.ndarray randoms
        cdef int64_t *randoms_data

        if size is None:
            with self.lock:
                return random_positive_int(&self._bitgen)

        randoms = <np.ndarray>np.empty(size, dtype=np.int64)
        randoms_data = <int64_t*>np.PyArray_DATA(randoms)
        n = np.PyArray_SIZE(randoms)

        for i in range(n):
            with self.lock, nogil:
                randoms_data[i] = random_positive_int(&self._bitgen)
        return randoms

    def randint(self, low, high=None, size=None, dtype=int):
        """
        randint(low, high=None, size=None, dtype=int)

        Return random integers from `low` (inclusive) to `high` (exclusive).

        Return random integers from the "discrete uniform" distribution of
        the specified dtype in the "half-open" interval [`low`, `high`). If
        `high` is None (the default), then results are from [0, `low`).

        .. note::
            New code should use the `~numpy.random.Generator.integers`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        low : int or array-like of ints
            Lowest (signed) integers to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is one above the
            *highest* such integer).
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
            The default value is int.

            .. versionadded:: 1.11.0

        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        See Also
        --------
        random_integers : similar to `randint`, only for the closed
            interval [`low`, `high`], and 1 is the lowest value if `high` is
            omitted.
        random.Generator.integers: which should be used for new code.

        Examples
        --------
        >>> np.random.randint(2, size=10)
        array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
        >>> np.random.randint(1, size=10)
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        Generate a 2 x 4 array of ints between 0 and 4, inclusive:

        >>> np.random.randint(5, size=(2, 4))
        array([[4, 0, 2, 1], # random
               [3, 2, 2, 0]])

        Generate a 1 x 3 array with 3 different upper bounds

        >>> np.random.randint(1, [3, 5, 10])
        array([2, 2, 9]) # random

        Generate a 1 by 3 array with 3 different lower bounds

        >>> np.random.randint([1, 5, 7], 10)
        array([9, 8, 7]) # random

        Generate a 2 by 4 array using broadcasting with dtype of uint8

        >>> np.random.randint([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)
        array([[ 8,  6,  9,  7], # random
               [ 1, 16,  9, 12]], dtype=uint8)
        """

        if high is None:
            high = low
            low = 0

        _dtype = np.dtype(dtype)

        if not _dtype.isnative:
            # numpy 1.17.0, 2019-05-28
            warnings.warn('Providing a dtype with a non-native byteorder is '
                          'not supported. If you require platform-independent '
                          'byteorder, call byteswap when required.\nIn future '
                          'version, providing byteorder will raise a '
                          'ValueError', DeprecationWarning)
            _dtype = _dtype.newbyteorder()

        # Implementation detail: the use a masked method to generate
        # bounded uniform integers. Lemire's method is preferable since it is
        # faster. randomgen allows a choice, we will always use the slower but
        # backward compatible one.
        cdef bint _masked = True
        cdef bint _endpoint = False

        if _dtype == np.int32:
            ret = _rand_int32(low, high, size, _masked, _endpoint, &self._bitgen, self.lock)
        elif _dtype == np.int64:
            ret = _rand_int64(low, high, size, _masked, _endpoint, &self._bitgen, self.lock)
        elif _dtype == np.int16:
            ret = _rand_int16(low, high, size, _masked, _endpoint, &self._bitgen, self.lock)
        elif _dtype == np.int8:
            ret = _rand_int8(low, high, size, _masked, _endpoint, &self._bitgen, self.lock)
        elif _dtype == np.uint64:
            ret = _rand_uint64(low, high, size, _masked, _endpoint, &self._bitgen, self.lock)
        elif _dtype == np.uint32:
            ret = _rand_uint32(low, high, size, _masked, _endpoint, &self._bitgen, self.lock)
        elif _dtype == np.uint16:
            ret = _rand_uint16(low, high, size, _masked, _endpoint, &self._bitgen, self.lock)
        elif _dtype == np.uint8:
            ret = _rand_uint8(low, high, size, _masked, _endpoint, &self._bitgen, self.lock)
        elif _dtype == np.bool_:
            ret = _rand_bool(low, high, size, _masked, _endpoint, &self._bitgen, self.lock)
        else:
            raise TypeError('Unsupported dtype %r for randint' % _dtype)

        if size is None and dtype in (bool, int):
            if np.array(ret).shape == ():
                return dtype(ret)
        return ret

    def bytes(self, np.npy_intp length):
        """
        bytes(length)

        Return random bytes.

        .. note::
            New code should use the `~numpy.random.Generator.bytes`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        length : int
            Number of random bytes.

        Returns
        -------
        out : bytes
            String of length `length`.

        See Also
        --------
        random.Generator.bytes: which should be used for new code.

        Examples
        --------
        >>> np.random.bytes(10)
        b' eh\\x85\\x022SZ\\xbf\\xa4' #random
        """
        cdef Py_ssize_t n_uint32 = ((length - 1) // 4 + 1)
        # Interpret the uint32s as little-endian to convert them to bytes
        # consistently.
        return self.randint(0, 4294967296, size=n_uint32,
                            dtype=np.uint32).astype('<u4').tobytes()[:length]

    @cython.wraparound(True)
    def choice(self, a, size=None, replace=True, p=None):
        """
        choice(a, size=None, replace=True, p=None)

        Generates a random sample from a given 1-D array

        .. versionadded:: 1.7.0

        .. note::
            New code should use the `~numpy.random.Generator.choice`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        a : 1-D array-like or int
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated as if it were ``np.arange(a)``
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        replace : boolean, optional
            Whether the sample is with or without replacement. Default is True,
            meaning that a value of ``a`` can be selected multiple times.
        p : 1-D array-like, optional
            The probabilities associated with each entry in a.
            If not given, the sample assumes a uniform distribution over all
            entries in ``a``.

        Returns
        -------
        samples : single item or ndarray
            The generated random samples

        Raises
        ------
        ValueError
            If a is an int and less than zero, if a or p are not 1-dimensional,
            if a is an array-like of size 0, if p is not a vector of
            probabilities, if a and p have different lengths, or if
            replace=False and the sample size is greater than the population
            size

        See Also
        --------
        randint, shuffle, permutation
        random.Generator.choice: which should be used in new code

        Notes
        -----
        Setting user-specified probabilities through ``p`` uses a more general but less
        efficient sampler than the default. The general sampler produces a different sample
        than the optimized sampler even if each element of ``p`` is 1 / len(a).

        Sampling random rows from a 2-D array is not possible with this function,
        but is possible with `Generator.choice` through its ``axis`` keyword.

        Examples
        --------
        Generate a uniform random sample from np.arange(5) of size 3:

        >>> np.random.choice(5, 3)
        array([0, 3, 4]) # random
        >>> #This is equivalent to np.random.randint(0,5,3)

        Generate a non-uniform random sample from np.arange(5) of size 3:

        >>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        array([3, 3, 0]) # random

        Generate a uniform random sample from np.arange(5) of size 3 without
        replacement:

        >>> np.random.choice(5, 3, replace=False)
        array([3,1,0]) # random
        >>> #This is equivalent to np.random.permutation(np.arange(5))[:3]

        Generate a non-uniform random sample from np.arange(5) of size
        3 without replacement:

        >>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
        array([2, 3, 0]) # random

        Any of the above can be repeated with an arbitrary array-like
        instead of just integers. For instance:

        >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
        >>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
        array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'], # random
              dtype='<U11')

        """

        # Format and Verify input
        a = np.array(a, copy=False)
        if a.ndim == 0:
            try:
                # __index__ must return an integer by python rules.
                pop_size = operator.index(a.item())
            except TypeError:
                raise ValueError("a must be 1-dimensional or an integer")
            if pop_size <= 0 and np.prod(size) != 0:
                raise ValueError("a must be greater than 0 unless no samples are taken")
        elif a.ndim != 1:
            raise ValueError("a must be 1-dimensional")
        else:
            pop_size = a.shape[0]
            if pop_size is 0 and np.prod(size) != 0:
                raise ValueError("'a' cannot be empty unless no samples are taken")

        if p is not None:
            d = len(p)

            atol = np.sqrt(np.finfo(np.float64).eps)
            if isinstance(p, np.ndarray):
                if np.issubdtype(p.dtype, np.floating):
                    atol = max(atol, np.sqrt(np.finfo(p.dtype).eps))

            p = <np.ndarray>np.PyArray_FROM_OTF(
                p, np.NPY_DOUBLE, np.NPY_ALIGNED | np.NPY_ARRAY_C_CONTIGUOUS)
            pix = <double*>np.PyArray_DATA(p)

            if p.ndim != 1:
                raise ValueError("'p' must be 1-dimensional")
            if p.size != pop_size:
                raise ValueError("'a' and 'p' must have same size")
            p_sum = kahan_sum(pix, d)
            if np.isnan(p_sum):
                raise ValueError("probabilities contain NaN")
            if np.logical_or.reduce(p < 0):
                raise ValueError("probabilities are not non-negative")
            if abs(p_sum - 1.) > atol:
                raise ValueError("probabilities do not sum to 1")

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
                uniform_samples = self.random_sample(shape)
                idx = cdf.searchsorted(uniform_samples, side='right')
                # searchsorted returns a scalar
                # force cast to int for LLP64
                idx = np.array(idx, copy=False).astype(int, casting='unsafe')
            else:
                idx = self.randint(0, pop_size, size=shape)
        else:
            if size > pop_size:
                raise ValueError("Cannot take a larger sample than "
                                 "population when 'replace=False'")
            elif size < 0:
                raise ValueError("Negative dimensions are not allowed")

            if p is not None:
                if np.count_nonzero(p > 0) < size:
                    raise ValueError("Fewer non-zero entries in p than size")
                n_uniq = 0
                p = p.copy()
                found = np.zeros(shape, dtype=int)
                flat_found = found.ravel()
                while n_uniq < size:
                    x = self.rand(size - n_uniq)
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
                idx = self.permutation(pop_size)[:size]
                idx.shape = shape

        if is_scalar and isinstance(idx, np.ndarray):
            # In most cases a scalar will have been made an array
            idx = idx.item(0)

        # Use samples as indices for a if a is array-like
        if a.ndim == 0:
            return idx

        if not is_scalar and idx.ndim == 0:
            # If size == () then the user requested a 0-d array as opposed to
            # a scalar object when size is None. However a[idx] is always a
            # scalar and not an array. So this makes sure the result is an
            # array, taking into account that np.array(item) may not work
            # for object arrays.
            res = np.empty((), dtype=a.dtype)
            res[()] = a[idx]
            return res

        return a[idx]

    def uniform(self, low=0.0, high=1.0, size=None):
        """
        uniform(low=0.0, high=1.0, size=None)

        Draw samples from a uniform distribution.

        Samples are uniformly distributed over the half-open interval
        ``[low, high)`` (includes low, but excludes high).  In other words,
        any value within the given interval is equally likely to be drawn
        by `uniform`.

        .. note::
            New code should use the `~numpy.random.Generator.uniform`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        low : float or array_like of floats, optional
            Lower boundary of the output interval.  All values generated will be
            greater than or equal to low.  The default value is 0.
        high : float or array_like of floats
            Upper boundary of the output interval.  All values generated will be
            less than or equal to high.  The high limit may be included in the 
            returned array of floats due to floating-point rounding in the 
            equation ``low + (high-low) * random_sample()``.  The default value 
            is 1.0.
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
        randint : Discrete uniform distribution, yielding integers.
        random_integers : Discrete uniform distribution over the closed
                          interval ``[low, high]``.
        random_sample : Floats uniformly distributed over ``[0, 1)``.
        random : Alias for `random_sample`.
        rand : Convenience function that accepts dimensions as input, e.g.,
               ``rand(2,2)`` would generate a 2-by-2 array of floats,
               uniformly distributed over ``[0, 1)``.
        random.Generator.uniform: which should be used for new code.

        Notes
        -----
        The probability density function of the uniform distribution is

        .. math:: p(x) = \\frac{1}{b - a}

        anywhere within the interval ``[a, b)``, and zero elsewhere.

        When ``high`` == ``low``, values of ``low`` will be returned.
        If ``high`` < ``low``, the results are officially undefined
        and may eventually raise an error, i.e. do not rely on this
        function to behave when passed arguments satisfying that
        inequality condition. The ``high`` limit may be included in the
        returned array of floats due to floating-point rounding in the
        equation ``low + (high-low) * random_sample()``. For example:

        >>> x = np.float32(5*0.99999999)
        >>> x
        5.0


        Examples
        --------
        Draw samples from the distribution:

        >>> s = np.random.uniform(-1,0,1000)

        All values are within the given interval:

        >>> np.all(s >= -1)
        True
        >>> np.all(s < 0)
        True

        Display the histogram of the samples, along with the
        probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 15, density=True)
        >>> plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
        >>> plt.show()

        """
        cdef bint is_scalar = True
        cdef np.ndarray alow, ahigh, arange
        cdef double _low, _high, range
        cdef object temp

        alow = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_DOUBLE, np.NPY_ALIGNED)
        ahigh = <np.ndarray>np.PyArray_FROM_OTF(high, np.NPY_DOUBLE, np.NPY_ALIGNED)

        if np.PyArray_NDIM(alow) == np.PyArray_NDIM(ahigh) == 0:
            _low = PyFloat_AsDouble(low)
            _high = PyFloat_AsDouble(high)
            range = _high - _low
            if not np.isfinite(range):
                raise OverflowError('Range exceeds valid bounds')

            return cont(&random_uniform, &self._bitgen, size, self.lock, 2,
                        _low, '', CONS_NONE,
                        range, '', CONS_NONE,
                        0.0, '', CONS_NONE,
                        None)

        temp = np.subtract(ahigh, alow)
        Py_INCREF(temp)
        # needed to get around Pyrex's automatic reference-counting
        # rules because EnsureArray steals a reference
        arange = <np.ndarray>np.PyArray_EnsureArray(temp)
        if not np.all(np.isfinite(arange)):
            raise OverflowError('Range exceeds valid bounds')
        return cont(&random_uniform, &self._bitgen, size, self.lock, 2,
                    alow, '', CONS_NONE,
                    arange, '', CONS_NONE,
                    0.0, '', CONS_NONE,
                    None)

    def rand(self, *args):
        """
        rand(d0, d1, ..., dn)

        Random values in a given shape.

        .. note::
            This is a convenience function for users porting code from Matlab,
            and wraps `random_sample`. That function takes a
            tuple to specify the size of the output, which is consistent with
            other NumPy functions like `numpy.zeros` and `numpy.ones`.

        Create an array of the given shape and populate it with
        random samples from a uniform distribution
        over ``[0, 1)``.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, must be non-negative.
            If no argument is given a single Python float is returned.

        Returns
        -------
        out : ndarray, shape ``(d0, d1, ..., dn)``
            Random values.

        See Also
        --------
        random

        Examples
        --------
        >>> np.random.rand(3,2)
        array([[ 0.14022471,  0.96360618],  #random
               [ 0.37601032,  0.25528411],  #random
               [ 0.49313049,  0.94909878]]) #random

        """
        if len(args) == 0:
            return self.random_sample()
        else:
            return self.random_sample(size=args)

    def randn(self, *args):
        """
        randn(d0, d1, ..., dn)

        Return a sample (or samples) from the "standard normal" distribution.

        .. note::
            This is a convenience function for users porting code from Matlab,
            and wraps `standard_normal`. That function takes a
            tuple to specify the size of the output, which is consistent with
            other NumPy functions like `numpy.zeros` and `numpy.ones`.

        .. note::
            New code should use the
            `~numpy.random.Generator.standard_normal`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        If positive int_like arguments are provided, `randn` generates an array
        of shape ``(d0, d1, ..., dn)``, filled
        with random floats sampled from a univariate "normal" (Gaussian)
        distribution of mean 0 and variance 1. A single float randomly sampled
        from the distribution is returned if no argument is provided.

        Parameters
        ----------
        d0, d1, ..., dn : int, optional
            The dimensions of the returned array, must be non-negative.
            If no argument is given a single Python float is returned.

        Returns
        -------
        Z : ndarray or float
            A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
            the standard normal distribution, or a single such float if
            no parameters were supplied.

        See Also
        --------
        standard_normal : Similar, but takes a tuple as its argument.
        normal : Also accepts mu and sigma arguments.
        random.Generator.standard_normal: which should be used for new code.

        Notes
        -----
        For random samples from the normal distribution with mean ``mu`` and
        standard deviation ``sigma``, use::

            sigma * np.random.randn(...) + mu

        Examples
        --------
        >>> np.random.randn()
        2.1923875335537315  # random

        Two-by-four array of samples from the normal distribution with
        mean 3 and standard deviation 2.5:

        >>> 3 + 2.5 * np.random.randn(2, 4)
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random

        """
        if len(args) == 0:
            return self.standard_normal()
        else:
            return self.standard_normal(size=args)

    def random_integers(self, low, high=None, size=None):
        """
        random_integers(low, high=None, size=None)

        Random integers of type `np.int_` between `low` and `high`, inclusive.

        Return random integers of type `np.int_` from the "discrete uniform"
        distribution in the closed interval [`low`, `high`].  If `high` is
        None (the default), then results are from [1, `low`]. The `np.int_`
        type translates to the C long integer type and its precision
        is platform dependent.

        This function has been deprecated. Use randint instead.

        .. deprecated:: 1.11.0

        Parameters
        ----------
        low : int
            Lowest (signed) integer to be drawn from the distribution (unless
            ``high=None``, in which case this parameter is the *highest* such
            integer).
        high : int, optional
            If provided, the largest (signed) integer to be drawn from the
            distribution (see above for behavior if ``high=None``).
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : int or ndarray of ints
            `size`-shaped array of random integers from the appropriate
            distribution, or a single such random int if `size` not provided.

        See Also
        --------
        randint : Similar to `random_integers`, only for the half-open
            interval [`low`, `high`), and 0 is the lowest value if `high` is
            omitted.

        Notes
        -----
        To sample from N evenly spaced floating-point numbers between a and b,
        use::

          a + (b - a) * (np.random.random_integers(N) - 1) / (N - 1.)

        Examples
        --------
        >>> np.random.random_integers(5)
        4 # random
        >>> type(np.random.random_integers(5))
        <class 'numpy.int64'>
        >>> np.random.random_integers(5, size=(3,2))
        array([[5, 4], # random
               [3, 3],
               [4, 5]])

        Choose five random numbers from the set of five evenly-spaced
        numbers between 0 and 2.5, inclusive (*i.e.*, from the set
        :math:`{0, 5/8, 10/8, 15/8, 20/8}`):

        >>> 2.5 * (np.random.random_integers(5, size=(5,)) - 1) / 4.
        array([ 0.625,  1.25 ,  0.625,  0.625,  2.5  ]) # random

        Roll two six sided dice 1000 times and sum the results:

        >>> d1 = np.random.random_integers(1, 6, 1000)
        >>> d2 = np.random.random_integers(1, 6, 1000)
        >>> dsums = d1 + d2

        Display results as a histogram:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(dsums, 11, density=True)
        >>> plt.show()

        """
        if high is None:
            warnings.warn(("This function is deprecated. Please call "
                           "randint(1, {low} + 1) instead".format(low=low)),
                          DeprecationWarning)
            high = low
            low = 1

        else:
            warnings.warn(("This function is deprecated. Please call "
                           "randint({low}, {high} + 1) "
                           "instead".format(low=low, high=high)),
                          DeprecationWarning)

        return self.randint(low, int(high) + 1, size=size, dtype='l')

    # Complicated, continuous distributions:
    def standard_normal(self, size=None):
        """
        standard_normal(size=None)

        Draw samples from a standard Normal distribution (mean=0, stdev=1).

        .. note::
            New code should use the
            `~numpy.random.Generator.standard_normal`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

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
        random.Generator.standard_normal: which should be used for new code.

        Notes
        -----
        For random samples from the normal distribution with mean ``mu`` and
        standard deviation ``sigma``, use one of::

            mu + sigma * np.random.standard_normal(size=...)
            np.random.normal(mu, sigma, size=...)

        Examples
        --------
        >>> np.random.standard_normal()
        2.1923875335537315 #random

        >>> s = np.random.standard_normal(8000)
        >>> s
        array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311,  # random
               -0.38672696, -0.4685006 ])                                # random
        >>> s.shape
        (8000,)
        >>> s = np.random.standard_normal(size=(3, 4, 2))
        >>> s.shape
        (3, 4, 2)

        Two-by-four array of samples from the normal distribution with
        mean 3 and standard deviation 2.5:

        >>> 3 + 2.5 * np.random.standard_normal(size=(2, 4))
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random

        """
        return cont(&legacy_gauss, &self._aug_state, size, self.lock, 0,
                    None, None, CONS_NONE,
                    None, None, CONS_NONE,
                    None, None, CONS_NONE,
                    None)

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

        .. note::
            New code should use the `~numpy.random.Generator.normal`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        random.Generator.normal: which should be used for new code.

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
        normal is more likely to return samples lying close to the mean, rather
        than those far away.

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
        >>> s = np.random.normal(mu, sigma, 1000)

        Verify the mean and the variance:

        >>> abs(mu - np.mean(s))
        0.0  # may vary

        >>> abs(sigma - np.std(s, ddof=1))
        0.1  # may vary

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, density=True)
        >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
        ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        Two-by-four array of samples from the normal distribution with
        mean 3 and standard deviation 2.5:

        >>> np.random.normal(3, 2.5, size=(2, 4))
        array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
               [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random

        """
        return cont(&legacy_normal, &self._aug_state, size, self.lock, 2,
                    loc, '', CONS_NONE,
                    scale, 'scale', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    None)

    def standard_gamma(self, shape, size=None):
        """
        standard_gamma(shape, size=None)

        Draw samples from a standard Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,
        shape (sometimes designated "k") and scale=1.

        .. note::
            New code should use the
            `~numpy.random.Generator.standard_gamma`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        shape : float or array_like of floats
            Parameter, must be non-negative.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  If size is ``None`` (default),
            a single value is returned if ``shape`` is a scalar.  Otherwise,
            ``np.array(shape).size`` samples are drawn.

        Returns
        -------
        out : ndarray or scalar
            Drawn samples from the parameterized standard gamma distribution.

        See Also
        --------
        scipy.stats.gamma : probability density function, distribution or
            cumulative density function, etc.
        random.Generator.standard_gamma: which should be used for new code.

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
               https://en.wikipedia.org/wiki/Gamma_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> shape, scale = 2., 1. # mean and width
        >>> s = np.random.standard_gamma(shape, 1000000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps  # doctest: +SKIP
        >>> count, bins, ignored = plt.hist(s, 50, density=True)
        >>> y = bins**(shape-1) * ((np.exp(-bins/scale))/  # doctest: +SKIP
        ...                       (sps.gamma(shape) * scale**shape))
        >>> plt.plot(bins, y, linewidth=2, color='r')  # doctest: +SKIP
        >>> plt.show()

        """
        return cont(&legacy_standard_gamma, &self._aug_state, size, self.lock, 1,
                    shape, 'shape', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE,
                    None)

    def gamma(self, shape, scale=1.0, size=None):
        """
        gamma(shape, scale=1.0, size=None)

        Draw samples from a Gamma distribution.

        Samples are drawn from a Gamma distribution with specified parameters,
        `shape` (sometimes designated "k") and `scale` (sometimes designated
        "theta"), where both parameters are > 0.

        .. note::
            New code should use the `~numpy.random.Generator.gamma`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        random.Generator.gamma: which should be used for new code.

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
               https://en.wikipedia.org/wiki/Gamma_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> shape, scale = 2., 2.  # mean=4, std=2*sqrt(2)
        >>> s = np.random.gamma(shape, scale, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.special as sps  # doctest: +SKIP
        >>> count, bins, ignored = plt.hist(s, 50, density=True)
        >>> y = bins**(shape-1)*(np.exp(-bins/scale) /  # doctest: +SKIP
        ...                      (sps.gamma(shape)*scale**shape))
        >>> plt.plot(bins, y, linewidth=2, color='r')  # doctest: +SKIP
        >>> plt.show()

        """
        return cont(&legacy_gamma, &self._aug_state, size, self.lock, 2,
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

        .. note::
            New code should use the `~numpy.random.Generator.f`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        random.Generator.f: which should be used for new code.

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
        >>> s = np.random.f(dfnum, dfden, 1000)

        The lower bound for the top 1% of the samples is :

        >>> np.sort(s)[-10]
        7.61988120985 # random

        So there is about a 1% chance that the F statistic will exceed 7.62,
        the measured value is 36, so the null hypothesis is rejected at the 1%
        level.

        """
        return cont(&legacy_f, &self._aug_state, size, self.lock, 2,
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

        .. note::
            New code should use the
            `~numpy.random.Generator.noncentral_f`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        dfnum : float or array_like of floats
            Numerator degrees of freedom, must be > 0.

            .. versionchanged:: 1.14.0
               Earlier NumPy versions required dfnum > 1.
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

        See Also
        --------
        random.Generator.noncentral_f: which should be used for new code.

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
               http://mathworld.wolfram.com/NoncentralF-Distribution.html
        .. [2] Wikipedia, "Noncentral F-distribution",
               https://en.wikipedia.org/wiki/Noncentral_F-distribution

        Examples
        --------
        In a study, testing for a specific alternative to the null hypothesis
        requires use of the Noncentral F distribution. We need to calculate the
        area in the tail of the distribution that exceeds the value of the F
        distribution for the null hypothesis.  We'll plot the two probability
        distributions for comparison.

        >>> dfnum = 3 # between group deg of freedom
        >>> dfden = 20 # within groups degrees of freedom
        >>> nonc = 3.0
        >>> nc_vals = np.random.noncentral_f(dfnum, dfden, nonc, 1000000)
        >>> NF = np.histogram(nc_vals, bins=50, density=True)
        >>> c_vals = np.random.f(dfnum, dfden, 1000000)
        >>> F = np.histogram(c_vals, bins=50, density=True)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(F[1][1:], F[0])
        >>> plt.plot(NF[1][1:], NF[0])
        >>> plt.show()

        """
        return cont(&legacy_noncentral_f, &self._aug_state, size, self.lock, 3,
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

        .. note::
            New code should use the `~numpy.random.Generator.chisquare`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.chisquare: which should be used for new code.

        Notes
        -----
        The variable obtained by summing the squares of `df` independent,
        standard normally distributed random variables:

        .. math:: Q = \\sum_{i=0}^{\\mathtt{df}} X^2_i

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
        >>> np.random.chisquare(2,4)
        array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272]) # random
        """
        return cont(&legacy_chisquare, &self._aug_state, size, self.lock, 1,
                    df, 'df', CONS_POSITIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)

    def noncentral_chisquare(self, df, nonc, size=None):
        """
        noncentral_chisquare(df, nonc, size=None)

        Draw samples from a noncentral chi-square distribution.

        The noncentral :math:`\\chi^2` distribution is a generalization of
        the :math:`\\chi^2` distribution.

        .. note::
            New code should use the
            `~numpy.random.Generator.noncentral_chisquare`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        df : float or array_like of floats
            Degrees of freedom, must be > 0.

            .. versionchanged:: 1.10.0
               Earlier NumPy versions required dfnum > 1.
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

        See Also
        --------
        random.Generator.noncentral_chisquare: which should be used for new code.

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

        >>> import matplotlib.pyplot as plt
        >>> values = plt.hist(np.random.noncentral_chisquare(3, 20, 100000),
        ...                   bins=200, density=True)
        >>> plt.show()

        Draw values from a noncentral chisquare with very small noncentrality,
        and compare to a chisquare.

        >>> plt.figure()
        >>> values = plt.hist(np.random.noncentral_chisquare(3, .0000001, 100000),
        ...                   bins=np.arange(0., 25, .1), density=True)
        >>> values2 = plt.hist(np.random.chisquare(3, 100000),
        ...                    bins=np.arange(0., 25, .1), density=True)
        >>> plt.plot(values[1][0:-1], values[0]-values2[0], 'ob')
        >>> plt.show()

        Demonstrate how large values of non-centrality lead to a more symmetric
        distribution.

        >>> plt.figure()
        >>> values = plt.hist(np.random.noncentral_chisquare(3, 20, 100000),
        ...                   bins=200, density=True)
        >>> plt.show()

        """
        return cont(&legacy_noncentral_chisquare, &self._aug_state, size, self.lock, 2,
                    df, 'df', CONS_POSITIVE,
                    nonc, 'nonc', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE, None)

    def standard_cauchy(self, size=None):
        """
        standard_cauchy(size=None)

        Draw samples from a standard Cauchy distribution with mode = 0.

        Also known as the Lorentz distribution.

        .. note::
            New code should use the
            `~numpy.random.Generator.standard_cauchy`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.standard_cauchy: which should be used for new code.

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
              http://mathworld.wolfram.com/CauchyDistribution.html
        .. [3] Wikipedia, "Cauchy distribution"
              https://en.wikipedia.org/wiki/Cauchy_distribution

        Examples
        --------
        Draw samples and plot the distribution:

        >>> import matplotlib.pyplot as plt
        >>> s = np.random.standard_cauchy(1000000)
        >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
        >>> plt.hist(s, bins=100)
        >>> plt.show()

        """
        return cont(&legacy_standard_cauchy, &self._aug_state, size, self.lock, 0,
                    0.0, '', CONS_NONE, 0.0, '', CONS_NONE, 0.0, '', CONS_NONE, None)

    def standard_t(self, df, size=None):
        """
        standard_t(df, size=None)

        Draw samples from a standard Student's t distribution with `df` degrees
        of freedom.

        A special case of the hyperbolic distribution.  As `df` gets
        large, the result resembles that of the standard normal
        distribution (`standard_normal`).

        .. note::
            New code should use the `~numpy.random.Generator.standard_t`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.standard_t: which should be used for new code.

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
        >>> s = np.random.standard_t(10, size=1000000)
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
        return cont(&legacy_standard_t, &self._aug_state, size, self.lock, 1,
                    df, 'df', CONS_POSITIVE,
                    0, '', CONS_NONE,
                    0, '', CONS_NONE,
                    None)

    def vonmises(self, mu, kappa, size=None):
        """
        vonmises(mu, kappa, size=None)

        Draw samples from a von Mises distribution.

        Samples are drawn from a von Mises distribution with specified mode
        (mu) and dispersion (kappa), on the interval [-pi, pi].

        The von Mises distribution (also known as the circular normal
        distribution) is a continuous probability distribution on the unit
        circle.  It may be thought of as the circular analogue of the normal
        distribution.

        .. note::
            New code should use the `~numpy.random.Generator.vonmises`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        mu : float or array_like of floats
            Mode ("center") of the distribution.
        kappa : float or array_like of floats
            Dispersion of the distribution, has to be >=0.
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
        random.Generator.vonmises: which should be used for new code.

        Notes
        -----
        The probability density for the von Mises distribution is

        .. math:: p(x) = \\frac{e^{\\kappa cos(x-\\mu)}}{2\\pi I_0(\\kappa)},

        where :math:`\\mu` is the mode and :math:`\\kappa` the dispersion,
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

        >>> mu, kappa = 0.0, 4.0 # mean and dispersion
        >>> s = np.random.vonmises(mu, kappa, 1000)

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
        return cont(&legacy_vonmises, &self._bitgen, size, self.lock, 2,
                    mu, 'mu', CONS_NONE,
                    kappa, 'kappa', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE, None)

    def pareto(self, a, size=None):
        """
        pareto(a, size=None)

        Draw samples from a Pareto II or Lomax distribution with
        specified shape.

        The Lomax or Pareto II distribution is a shifted Pareto
        distribution. The classical Pareto distribution can be
        obtained from the Lomax distribution by adding 1 and
        multiplying by the scale parameter ``m`` (see Notes).  The
        smallest value of the Lomax distribution is zero while for the
        classical Pareto distribution it is ``mu``, where the standard
        Pareto distribution has location ``mu = 1``.  Lomax can also
        be considered as a simplified version of the Generalized
        Pareto distribution (available in SciPy), with the scale set
        to one and the location set to zero.

        The Pareto distribution must be greater than zero, and is
        unbounded above.  It is also known as the "80-20 rule".  In
        this distribution, 80 percent of the weights are in the lowest
        20 percent of the range, while the other 20 percent fill the
        remaining 80 percent of the range.

        .. note::
            New code should use the `~numpy.random.Generator.pareto`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
            Drawn samples from the parameterized Pareto distribution.

        See Also
        --------
        scipy.stats.lomax : probability density function, distribution or
            cumulative density function, etc.
        scipy.stats.genpareto : probability density function, distribution or
            cumulative density function, etc.
        random.Generator.pareto: which should be used for new code.

        Notes
        -----
        The probability density for the Pareto distribution is

        .. math:: p(x) = \\frac{am^a}{x^{a+1}}

        where :math:`a` is the shape and :math:`m` the scale.

        The Pareto distribution, named after the Italian economist
        Vilfredo Pareto, is a power law probability distribution
        useful in many real world problems.  Outside the field of
        economics it is generally referred to as the Bradford
        distribution. Pareto developed the distribution to describe
        the distribution of wealth in an economy.  It has also found
        use in insurance, web page access statistics, oil field sizes,
        and many other problems, including the download frequency for
        projects in Sourceforge [1]_.  It is one of the so-called
        "fat-tailed" distributions.

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

        >>> a, m = 3., 2.  # shape and mode
        >>> s = (np.random.pareto(a, 1000) + 1) * m

        Display the histogram of the samples, along with the probability
        density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, _ = plt.hist(s, 100, density=True)
        >>> fit = a*m**a / bins**(a+1)
        >>> plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
        >>> plt.show()

        """
        return cont(&legacy_pareto, &self._aug_state, size, self.lock, 1,
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

        .. note::
            New code should use the `~numpy.random.Generator.weibull`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        random.Generator.weibull: which should be used for new code.

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

        >>> a = 5. # shape
        >>> s = np.random.weibull(a, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> x = np.arange(1,100.)/50.
        >>> def weib(x,n,a):
        ...     return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

        >>> count, bins, ignored = plt.hist(np.random.weibull(5.,1000))
        >>> x = np.arange(1,100.)/50.
        >>> scale = count.max()/weib(x, 1., 5.).max()
        >>> plt.plot(x, weib(x, 1., 5.)*scale)
        >>> plt.show()

        """
        return cont(&legacy_weibull, &self._aug_state, size, self.lock, 1,
                    a, 'a', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE,
                    0.0, '', CONS_NONE, None)

    def power(self, a, size=None):
        """
        power(a, size=None)

        Draws samples in [0, 1] from a power distribution with positive
        exponent a - 1.

        Also known as the power function distribution.

        .. note::
            New code should use the `~numpy.random.Generator.power`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.power: which should be used for new code.

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

        >>> a = 5. # shape
        >>> samples = 1000
        >>> s = np.random.power(a, samples)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, bins=30)
        >>> x = np.linspace(0, 1, 100)
        >>> y = a*x**(a-1.)
        >>> normed_y = samples*np.diff(bins)[0]*y
        >>> plt.plot(x, normed_y)
        >>> plt.show()

        Compare the power function distribution to the inverse of the Pareto.

        >>> from scipy import stats # doctest: +SKIP
        >>> rvs = np.random.power(5, 1000000)
        >>> rvsp = np.random.pareto(5, 1000000)
        >>> xx = np.linspace(0,1,100)
        >>> powpdf = stats.powerlaw.pdf(xx,5)  # doctest: +SKIP

        >>> plt.figure()
        >>> plt.hist(rvs, bins=50, density=True)
        >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
        >>> plt.title('np.random.power(5)')

        >>> plt.figure()
        >>> plt.hist(1./(1.+rvsp), bins=50, density=True)
        >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
        >>> plt.title('inverse of 1 + np.random.pareto(5)')

        >>> plt.figure()
        >>> plt.hist(1./(1.+rvsp), bins=50, density=True)
        >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
        >>> plt.title('inverse of stats.pareto(5)')

        """
        return cont(&legacy_power, &self._aug_state, size, self.lock, 1,
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

        .. note::
            New code should use the `~numpy.random.Generator.laplace`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.laplace: which should be used for new code.

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
               http://mathworld.wolfram.com/LaplaceDistribution.html
        .. [4] Wikipedia, "Laplace distribution",
               https://en.wikipedia.org/wiki/Laplace_distribution

        Examples
        --------
        Draw samples from the distribution

        >>> loc, scale = 0., 1.
        >>> s = np.random.laplace(loc, scale, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, density=True)
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

        .. note::
            New code should use the `~numpy.random.Generator.gumbel`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        random.Generator.gumbel: which should be used for new code.

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

        >>> mu, beta = 0, 0.1 # location and scale
        >>> s = np.random.gumbel(mu, beta, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 30, density=True)
        >>> plt.plot(bins, (1/beta)*np.exp(-(bins - mu)/beta)
        ...          * np.exp( -np.exp( -(bins - mu) /beta) ),
        ...          linewidth=2, color='r')
        >>> plt.show()

        Show how an extreme value distribution can arise from a Gaussian process
        and compare to a Gaussian:

        >>> means = []
        >>> maxima = []
        >>> for i in range(0,1000) :
        ...    a = np.random.normal(mu, beta, 1000)
        ...    means.append(a.mean())
        ...    maxima.append(a.max())
        >>> count, bins, ignored = plt.hist(maxima, 30, density=True)
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

        .. note::
            New code should use the `~numpy.random.Generator.logistic`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        random.Generator.logistic: which should be used for new code.

        Notes
        -----
        The probability density for the Logistic distribution is

        .. math:: P(x) = P(x) = \\frac{e^{-(x-\\mu)/s}}{s(1+e^{-(x-\\mu)/s})^2},

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
               http://mathworld.wolfram.com/LogisticDistribution.html
        .. [3] Wikipedia, "Logistic-distribution",
               https://en.wikipedia.org/wiki/Logistic_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> loc, scale = 10, 1
        >>> s = np.random.logistic(loc, scale, 10000)
        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, bins=50)

        #   plot against distribution

        >>> def logist(x, loc, scale):
        ...     return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)
        >>> lgst_val = logist(bins, loc, scale)
        >>> plt.plot(bins, lgst_val * count.max() / lgst_val.max())
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

        .. note::
            New code should use the `~numpy.random.Generator.lognormal`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        random.Generator.lognormal: which should be used for new code.

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

        >>> mu, sigma = 3., 1. # mean and standard deviation
        >>> s = np.random.lognormal(mu, sigma, 1000)

        Display the histogram of the samples, along with
        the probability density function:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 100, density=True, align='mid')

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
        >>> b = []
        >>> for i in range(1000):
        ...    a = 10. + np.random.standard_normal(100)
        ...    b.append(np.prod(a))

        >>> b = np.array(b) / np.min(b) # scale values to be positive
        >>> count, bins, ignored = plt.hist(b, 100, density=True, align='mid')
        >>> sigma = np.std(np.log(b))
        >>> mu = np.mean(np.log(b))

        >>> x = np.linspace(min(bins), max(bins), 10000)
        >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
        ...        / (x * sigma * np.sqrt(2 * np.pi)))

        >>> plt.plot(x, pdf, color='r', linewidth=2)
        >>> plt.show()

        """
        return cont(&legacy_lognormal, &self._aug_state, size, self.lock, 2,
                    mean, 'mean', CONS_NONE,
                    sigma, 'sigma', CONS_NON_NEGATIVE,
                    0.0, '', CONS_NONE, None)

    def rayleigh(self, scale=1.0, size=None):
        """
        rayleigh(scale=1.0, size=None)

        Draw samples from a Rayleigh distribution.

        The :math:`\\chi` and Weibull distributions are generalizations of the
        Rayleigh.

        .. note::
            New code should use the `~numpy.random.Generator.rayleigh`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.rayleigh: which should be used for new code.

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
        >>> values = hist(np.random.rayleigh(3, 100000), bins=200, density=True)

        Wave heights tend to follow a Rayleigh distribution. If the mean wave
        height is 1 meter, what fraction of waves are likely to be larger than 3
        meters?

        >>> meanvalue = 1
        >>> modevalue = np.sqrt(2 / np.pi) * meanvalue
        >>> s = np.random.rayleigh(modevalue, 1000000)

        The percentage of waves larger than 3 meters is:

        >>> 100.*sum(s>3)/1000000.
        0.087300000000000003 # random

        """
        return cont(&legacy_rayleigh, &self._bitgen, size, self.lock, 1,
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

        .. note::
            New code should use the `~numpy.random.Generator.wald`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.wald: which should be used for new code.

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
        >>> h = plt.hist(np.random.wald(3, 2, 100000), bins=200, density=True)
        >>> plt.show()

        """
        return cont(&legacy_wald, &self._aug_state, size, self.lock, 2,
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

        .. note::
            New code should use the `~numpy.random.Generator.triangular`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.triangular: which should be used for new code.

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
        >>> h = plt.hist(np.random.triangular(-3, 0, 8, 100000), bins=200,
        ...              density=True)
        >>> plt.show()

        """
        cdef bint is_scalar = True
        cdef double fleft, fmode, fright
        cdef np.ndarray oleft, omode, oright

        oleft = <np.ndarray>np.PyArray_FROM_OTF(left, np.NPY_DOUBLE, np.NPY_ALIGNED)
        omode = <np.ndarray>np.PyArray_FROM_OTF(mode, np.NPY_DOUBLE, np.NPY_ALIGNED)
        oright = <np.ndarray>np.PyArray_FROM_OTF(right, np.NPY_DOUBLE, np.NPY_ALIGNED)

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

        .. note::
            New code should use the `~numpy.random.Generator.binomial`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        random.Generator.binomial: which should be used for new code.

        Notes
        -----
        The probability density for the binomial distribution is

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
               http://mathworld.wolfram.com/BinomialDistribution.html
        .. [5] Wikipedia, "Binomial distribution",
               https://en.wikipedia.org/wiki/Binomial_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> n, p = 10, .5  # number of trials, probability of each trial
        >>> s = np.random.binomial(n, p, 1000)
        # result of flipping a coin 10 times, tested 1000 times.

        A real world example. A company drills 9 wild-cat oil exploration
        wells, each with an estimated probability of success of 0.1. All nine
        wells fail. What is the probability of that happening?

        Let's do 20,000 trials of the model, and count the number that
        generate zero positive results.

        >>> sum(np.random.binomial(9, 0.1, 20000) == 0)/20000.
        # answer = 0.38885, or 38%.

        """

        # Uses a custom implementation since self._binomial is required
        cdef double _dp = 0
        cdef long _in = 0
        cdef bint is_scalar = True
        cdef np.npy_intp i, cnt
        cdef np.ndarray randoms
        cdef long *randoms_data
        cdef np.broadcast it

        p_arr = <np.ndarray>np.PyArray_FROM_OTF(p, np.NPY_DOUBLE, np.NPY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(p_arr) == 0
        n_arr = <np.ndarray>np.PyArray_FROM_OTF(n, np.NPY_LONG, np.NPY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(n_arr) == 0

        if not is_scalar:
            check_array_constraint(p_arr, 'p', CONS_BOUNDED_0_1)
            check_array_constraint(n_arr, 'n', CONS_NON_NEGATIVE)
            if size is not None:
                randoms = <np.ndarray>np.empty(size, int)
            else:
                it = np.PyArray_MultiIterNew2(p_arr, n_arr)
                randoms = <np.ndarray>np.empty(it.shape, int)

            cnt = np.PyArray_SIZE(randoms)

            it = np.PyArray_MultiIterNew3(randoms, p_arr, n_arr)
            validate_output_shape(it.shape, randoms)
            with self.lock, nogil:
                for i in range(cnt):
                    _dp = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
                    _in = (<long*>np.PyArray_MultiIter_DATA(it, 2))[0]
                    (<long*>np.PyArray_MultiIter_DATA(it, 0))[0] = \
                        legacy_random_binomial(&self._bitgen, _dp, _in,
                                               &self._binomial)

                    np.PyArray_MultiIter_NEXT(it)

            return randoms

        _dp = PyFloat_AsDouble(p)
        _in = <long>n
        check_constraint(_dp, 'p', CONS_BOUNDED_0_1)
        check_constraint(<double>_in, 'n', CONS_NON_NEGATIVE)

        if size is None:
            with self.lock:
                return <long>legacy_random_binomial(&self._bitgen, _dp, _in,
                                                    &self._binomial)

        randoms = <np.ndarray>np.empty(size, int)
        cnt = np.PyArray_SIZE(randoms)
        randoms_data = <long *>np.PyArray_DATA(randoms)

        with self.lock, nogil:
            for i in range(cnt):
                randoms_data[i] = legacy_random_binomial(&self._bitgen, _dp, _in,
                                                         &self._binomial)

        return randoms

    def negative_binomial(self, n, p, size=None):
        """
        negative_binomial(n, p, size=None)

        Draw samples from a negative binomial distribution.

        Samples are drawn from a negative binomial distribution with specified
        parameters, `n` successes and `p` probability of success where `n`
        is > 0 and `p` is in the interval [0, 1].

        .. note::
            New code should use the
            `~numpy.random.Generator.negative_binomial`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        n : float or array_like of floats
            Parameter of the distribution, > 0.
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
            Drawn samples from the parameterized negative binomial distribution,
            where each sample is equal to N, the number of failures that
            occurred before a total of n successes was reached.

        See Also
        --------
        random.Generator.negative_binomial: which should be used for new code.

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

        References
        ----------
        .. [1] Weisstein, Eric W. "Negative Binomial Distribution." From
               MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/NegativeBinomialDistribution.html
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

        >>> s = np.random.negative_binomial(1, 0.1, 100000)
        >>> for i in range(1, 11): # doctest: +SKIP
        ...    probability = sum(s<i) / 100000.
        ...    print(i, "wells drilled, probability of one success =", probability)

        """
        out = disc(&legacy_negative_binomial, &self._aug_state, size, self.lock, 2, 0,
                   n, 'n', CONS_POSITIVE,
                   p, 'p', CONS_BOUNDED_0_1,
                   0.0, '', CONS_NONE)
        # Match historical output type
        return int64_to_long(out)

    def poisson(self, lam=1.0, size=None):
        """
        poisson(lam=1.0, size=None)

        Draw samples from a Poisson distribution.

        The Poisson distribution is the limit of the binomial distribution
        for large N.

        .. note::
            New code should use the `~numpy.random.Generator.poisson`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.poisson: which should be used for new code.

        Notes
        -----
        The Poisson distribution

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
               http://mathworld.wolfram.com/PoissonDistribution.html
        .. [2] Wikipedia, "Poisson distribution",
               https://en.wikipedia.org/wiki/Poisson_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> import numpy as np
        >>> s = np.random.poisson(5, 10000)

        Display histogram of the sample:

        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s, 14, density=True)
        >>> plt.show()

        Draw each 100 values for lambda 100 and 500:

        >>> s = np.random.poisson(lam=(100., 500.), size=(100, 2))

        """
        out = disc(&legacy_random_poisson, &self._bitgen, size, self.lock, 1, 0,
                   lam, 'lam', LEGACY_CONS_POISSON,
                   0.0, '', CONS_NONE,
                   0.0, '', CONS_NONE)
        # Match historical output type
        return int64_to_long(out)

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

        .. note::
            New code should use the `~numpy.random.Generator.zipf`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        random.Generator.zipf: which should be used for new code.

        Notes
        -----
        The probability density for the Zipf distribution is

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
        >>> s = np.random.zipf(a, n)

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
        out = disc(&legacy_random_zipf, &self._bitgen, size, self.lock, 1, 0,
                   a, 'a', CONS_GT_1,
                   0.0, '', CONS_NONE,
                   0.0, '', CONS_NONE)
        # Match historical output type
        return int64_to_long(out)

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

        .. note::
            New code should use the `~numpy.random.Generator.geometric`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        See Also
        --------
        random.Generator.geometric: which should be used for new code.

        Examples
        --------
        Draw ten thousand values from the geometric distribution,
        with the probability of an individual success equal to 0.35:

        >>> z = np.random.geometric(p=0.35, size=10000)

        How many trials succeeded after a single run?

        >>> (z == 1).sum() / 10000.
        0.34889999999999999 #random

        """
        out = disc(&legacy_random_geometric, &self._bitgen, size, self.lock, 1, 0,
                   p, 'p', CONS_BOUNDED_GT_0_1,
                   0.0, '', CONS_NONE,
                   0.0, '', CONS_NONE)
        # Match historical output type
        return int64_to_long(out)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        """
        hypergeometric(ngood, nbad, nsample, size=None)

        Draw samples from a Hypergeometric distribution.

        Samples are drawn from a hypergeometric distribution with specified
        parameters, `ngood` (ways to make a good selection), `nbad` (ways to make
        a bad selection), and `nsample` (number of items sampled, which is less
        than or equal to the sum ``ngood + nbad``).

        .. note::
            New code should use the
            `~numpy.random.Generator.hypergeometric`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        ngood : int or array_like of ints
            Number of ways to make a good selection.  Must be nonnegative.
        nbad : int or array_like of ints
            Number of ways to make a bad selection.  Must be nonnegative.
        nsample : int or array_like of ints
            Number of items sampled.  Must be at least 1 and at most
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
        scipy.stats.hypergeom : probability density function, distribution or
            cumulative density function, etc.
        random.Generator.hypergeometric: which should be used for new code.

        Notes
        -----
        The probability density for the Hypergeometric distribution is

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

        References
        ----------
        .. [1] Lentner, Marvin, "Elementary Applied Statistics", Bogden
               and Quigley, 1972.
        .. [2] Weisstein, Eric W. "Hypergeometric Distribution." From
               MathWorld--A Wolfram Web Resource.
               http://mathworld.wolfram.com/HypergeometricDistribution.html
        .. [3] Wikipedia, "Hypergeometric distribution",
               https://en.wikipedia.org/wiki/Hypergeometric_distribution

        Examples
        --------
        Draw samples from the distribution:

        >>> ngood, nbad, nsamp = 100, 2, 10
        # number of good, number of bad, and number of samples
        >>> s = np.random.hypergeometric(ngood, nbad, nsamp, 1000)
        >>> from matplotlib.pyplot import hist
        >>> hist(s)
        #   note that it is very unlikely to grab both bad items

        Suppose you have an urn with 15 white and 15 black marbles.
        If you pull 15 marbles at random, how likely is it that
        12 or more of them are one color?

        >>> s = np.random.hypergeometric(15, 15, 15, 100000)
        >>> sum(s>=12)/100000. + sum(s<=3)/100000.
        #   answer = 0.003 ... pretty unlikely!

        """
        cdef bint is_scalar = True
        cdef np.ndarray ongood, onbad, onsample
        cdef int64_t lngood, lnbad, lnsample

        # This cast to long is required to ensure that the values are inbounds
        ongood = <np.ndarray>np.PyArray_FROM_OTF(ngood, np.NPY_LONG, np.NPY_ALIGNED)
        onbad = <np.ndarray>np.PyArray_FROM_OTF(nbad, np.NPY_LONG, np.NPY_ALIGNED)
        onsample = <np.ndarray>np.PyArray_FROM_OTF(nsample, np.NPY_LONG, np.NPY_ALIGNED)

        if np.PyArray_NDIM(ongood) == np.PyArray_NDIM(onbad) == np.PyArray_NDIM(onsample) == 0:

            lngood = <int64_t>ngood
            lnbad = <int64_t>nbad
            lnsample = <int64_t>nsample

            if lngood + lnbad < lnsample:
                raise ValueError("ngood + nbad < nsample")
            out = disc(&legacy_random_hypergeometric, &self._bitgen, size, self.lock, 0, 3,
                       lngood, 'ngood', CONS_NON_NEGATIVE,
                       lnbad, 'nbad', CONS_NON_NEGATIVE,
                       lnsample, 'nsample', CONS_GTE_1)
            # Match historical output type
            return int64_to_long(out)

        if np.any(np.less(np.add(ongood, onbad), onsample)):
            raise ValueError("ngood + nbad < nsample")
        # Convert to int64, if necessary, to use int64 infrastructure
        ongood = ongood.astype(np.int64)
        onbad = onbad.astype(np.int64)
        onsample = onsample.astype(np.int64)
        out = discrete_broadcast_iii(&legacy_random_hypergeometric,&self._bitgen, size, self.lock,
                                     ongood, 'ngood', CONS_NON_NEGATIVE,
                                     onbad, 'nbad', CONS_NON_NEGATIVE,
                                     onsample, 'nsample', CONS_GTE_1)
        # Match historical output type
        return int64_to_long(out)

    def logseries(self, p, size=None):
        """
        logseries(p, size=None)

        Draw samples from a logarithmic series distribution.

        Samples are drawn from a log series distribution with specified
        shape parameter, 0 <= ``p`` < 1.

        .. note::
            New code should use the `~numpy.random.Generator.logseries`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
        random.Generator.logseries: which should be used for new code.

        Notes
        -----
        The probability density for the Log Series distribution is

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
        >>> s = np.random.logseries(a, 10000)
        >>> import matplotlib.pyplot as plt
        >>> count, bins, ignored = plt.hist(s)

        #   plot against distribution

        >>> def logseries(k, p):
        ...     return -p**k/(k*np.log(1-p))
        >>> plt.plot(bins, logseries(bins, a)*count.max()/
        ...          logseries(bins, a).max(), 'r')
        >>> plt.show()

        """
        out = disc(&legacy_logseries, &self._bitgen, size, self.lock, 1, 0,
                   p, 'p', CONS_BOUNDED_LT_0_1,
                   0.0, '', CONS_NONE,
                   0.0, '', CONS_NONE)
        # Match historical output type
        return int64_to_long(out)

    # Multivariate distributions:
    def multivariate_normal(self, mean, cov, size=None, check_valid='warn',
                            tol=1e-8):
        """
        multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8)

        Draw random samples from a multivariate normal distribution.

        The multivariate normal, multinormal or Gaussian distribution is a
        generalization of the one-dimensional normal distribution to higher
        dimensions.  Such a distribution is specified by its mean and
        covariance matrix.  These parameters are analogous to the mean
        (average or "center") and variance (standard deviation, or "width,"
        squared) of the one-dimensional normal distribution.

        .. note::
            New code should use the
            `~numpy.random.Generator.multivariate_normal`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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

        Returns
        -------
        out : ndarray
            The drawn samples, of shape *size*, if that was provided.  If not,
            the shape is ``(N,)``.

            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
            value drawn from the distribution.

        See Also
        --------
        random.Generator.multivariate_normal: which should be used for new code.

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
        >>> x, y = np.random.multivariate_normal(mean, cov, 5000).T
        >>> plt.plot(x, y, 'x')
        >>> plt.axis('equal')
        >>> plt.show()

        Note that the covariance matrix must be positive semidefinite (a.k.a.
        nonnegative-definite). Otherwise, the behavior of this method is
        undefined and backwards compatibility is not guaranteed.

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
        >>> x = np.random.multivariate_normal(mean, cov, (3, 3))
        >>> x.shape
        (3, 3, 2)

        Here we generate 800 samples from the bivariate normal distribution
        with mean [0, 0] and covariance matrix [[6, -3], [-3, 3.5]].  The
        expected variances of the first and second components of the sample
        are 6 and 3.5, respectively, and the expected correlation
        coefficient is -3/sqrt(6*3.5) ≈ -0.65465.

        >>> cov = np.array([[6, -3], [-3, 3.5]])
        >>> pts = np.random.multivariate_normal([0, 0], cov, size=800)

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
        from numpy.linalg import svd

        # Check preconditions on arguments
        mean = np.array(mean)
        cov = np.array(cov)
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
        (u, s, v) = svd(cov)

        if check_valid != 'ignore':
            if check_valid != 'warn' and check_valid != 'raise':
                raise ValueError(
                    "check_valid must equal 'warn', 'raise', or 'ignore'")

            psd = np.allclose(np.dot(v.T * s, v), cov, rtol=tol, atol=tol)
            if not psd:
                if check_valid == 'warn':
                    warnings.warn("covariance is not symmetric positive-semidefinite.",
                        RuntimeWarning)
                else:
                    raise ValueError(
                        "covariance is not symmetric positive-semidefinite.")

        x = np.dot(x, np.sqrt(s)[:, None] * v)
        x += mean
        x.shape = tuple(final_shape)
        return x

    def multinomial(self, long n, object pvals, size=None):
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

        .. note::
            New code should use the `~numpy.random.Generator.multinomial`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        n : int
            Number of experiments.
        pvals : sequence of floats, length p
            Probabilities of each of the ``p`` different outcomes.  These
            must sum to 1 (however, the last element is always assumed to
            account for the remaining probability, as long as
            ``sum(pvals[:-1]) <= 1)``.
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.

        Returns
        -------
        out : ndarray
            The drawn samples, of shape *size*, if that was provided.  If not,
            the shape is ``(N,)``.

            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
            value drawn from the distribution.

        See Also
        --------
        random.Generator.multinomial: which should be used for new code.

        Examples
        --------
        Throw a dice 20 times:

        >>> np.random.multinomial(20, [1/6.]*6, size=1)
        array([[4, 1, 7, 5, 2, 1]]) # random

        It landed 4 times on 1, once on 2, etc.

        Now, throw the dice 20 times, and 20 times again:

        >>> np.random.multinomial(20, [1/6.]*6, size=2)
        array([[3, 4, 3, 3, 4, 3], # random
               [2, 4, 3, 4, 0, 7]])

        For the first run, we threw 3 times 1, 4 times 2, etc.  For the second,
        we threw 2 times 1, 4 times 2, etc.

        A loaded die is more likely to land on number 6:

        >>> np.random.multinomial(100, [1/7.]*5 + [2/7.])
        array([11, 16, 14, 17, 16, 26]) # random

        The probability inputs should be normalized. As an implementation
        detail, the value of the last entry is ignored and assumed to take
        up any leftover probability mass, but this should not be relied on.
        A biased coin which has twice as much weight on one side as on the
        other should be sampled like so:

        >>> np.random.multinomial(100, [1.0 / 3, 2.0 / 3])  # RIGHT
        array([38, 62]) # random

        not like:

        >>> np.random.multinomial(100, [1.0, 2.0])  # WRONG
        Traceback (most recent call last):
        ValueError: pvals < 0, pvals > 1 or pvals contains NaNs

        """
        cdef np.npy_intp d, i, sz, offset, niter
        cdef np.ndarray parr, mnarr
        cdef double *pix
        cdef long *mnix
        cdef long ni

        parr = <np.ndarray>np.PyArray_FROMANY(
            pvals, np.NPY_DOUBLE, 0, 1, np.NPY_ARRAY_ALIGNED | np.NPY_ARRAY_C_CONTIGUOUS)
        if np.PyArray_NDIM(parr) == 0:
            raise TypeError("pvals must be a 1-d sequence")
        d = np.PyArray_SIZE(parr)
        pix = <double*>np.PyArray_DATA(parr)
        check_array_constraint(parr, 'pvals', CONS_BOUNDED_0_1)
        # Only check if pvals is non-empty due no checks in kahan_sum
        if d and kahan_sum(pix, d-1) > (1.0 + 1e-12):
            # When floating, but not float dtype, and close, improve the error
            # 1.0001 works for float16 and float32
            if (isinstance(pvals, np.ndarray)
                    and np.issubdtype(pvals.dtype, np.floating)
                    and pvals.dtype != float
                    and pvals.sum() < 1.0001):
                msg = ("sum(pvals[:-1].astype(np.float64)) > 1.0. The pvals "
                       "array is cast to 64-bit floating point prior to "
                       "checking the sum. Precision changes when casting may "
                       "cause problems even if the sum of the original pvals "
                       "is valid.")
            else:
                msg = "sum(pvals[:-1]) > 1.0"
            raise ValueError(msg)
        if size is None:
            shape = (d,)
        else:
            try:
                shape = (operator.index(size), d)
            except:
                shape = tuple(size) + (d,)
        multin = np.zeros(shape, dtype=int)
        mnarr = <np.ndarray>multin
        mnix = <long*>np.PyArray_DATA(mnarr)
        sz = np.PyArray_SIZE(mnarr)
        ni = n
        check_constraint(ni, 'n', CONS_NON_NEGATIVE)
        offset = 0
        # gh-20483: Avoids divide by 0
        niter = sz // d if d else 0
        with self.lock, nogil:
            for i in range(niter):
                legacy_random_multinomial(&self._bitgen, ni, &mnix[offset], pix, d, &self._binomial)
                offset += d

        return multin

    def dirichlet(self, object alpha, size=None):
        """
        dirichlet(alpha, size=None)

        Draw samples from the Dirichlet distribution.

        Draw `size` samples of dimension k from a Dirichlet distribution. A
        Dirichlet-distributed random variable can be seen as a multivariate
        generalization of a Beta distribution. The Dirichlet distribution
        is a conjugate prior of a multinomial distribution in Bayesian
        inference.

        .. note::
            New code should use the `~numpy.random.Generator.dirichlet`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

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
            If any value in ``alpha`` is less than or equal to zero

        See Also
        --------
        random.Generator.dirichlet: which should be used for new code.

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
               http://www.inference.org.uk/mackay/itila/
        .. [2] Wikipedia, "Dirichlet distribution",
               https://en.wikipedia.org/wiki/Dirichlet_distribution

        Examples
        --------
        Taking an example cited in Wikipedia, this distribution can be used if
        one wanted to cut strings (each of initial length 1.0) into K pieces
        with different lengths, where each piece had, on average, a designated
        average length, but allowing some variation in the relative sizes of
        the pieces.

        >>> s = np.random.dirichlet((10, 5, 3), 20).transpose()

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
        cdef np.ndarray alpha_arr, val_arr
        cdef double *alpha_data
        cdef double *val_data
        cdef double  acc, invacc

        k = len(alpha)
        alpha_arr = <np.ndarray>np.PyArray_FROMANY(
            alpha, np.NPY_DOUBLE, 1, 1,
            np.NPY_ARRAY_ALIGNED | np.NPY_ARRAY_C_CONTIGUOUS)
        if np.any(np.less_equal(alpha_arr, 0)):
            raise ValueError('alpha <= 0')
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
        val_data = <double*>np.PyArray_DATA(val_arr)

        i = 0
        totsize = np.PyArray_SIZE(val_arr)
        with self.lock, nogil:
            while i < totsize:
                acc = 0.0
                for j in range(k):
                    val_data[i+j] = legacy_standard_gamma(&self._aug_state,
                                                          alpha_data[j])
                    acc = acc + val_data[i + j]
                invacc = 1/acc
                for j in range(k):
                    val_data[i + j] = val_data[i + j] * invacc
                i = i + k

        return diric

    # Shuffling and permutations:
    def shuffle(self, object x):
        """
        shuffle(x)

        Modify a sequence in-place by shuffling its contents.

        This function only shuffles the array along the first axis of a
        multi-dimensional array. The order of sub-arrays is changed but
        their contents remains the same.

        .. note::
            New code should use the `~numpy.random.Generator.shuffle`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        x : ndarray or MutableSequence
            The array, list or mutable sequence to be shuffled.

        Returns
        -------
        None

        See Also
        --------
        random.Generator.shuffle: which should be used for new code.

        Examples
        --------
        >>> arr = np.arange(10)
        >>> np.random.shuffle(arr)
        >>> arr
        [1 7 5 2 9 4 3 6 0 8] # random

        Multi-dimensional arrays are only shuffled along the first axis:

        >>> arr = np.arange(9).reshape((3, 3))
        >>> np.random.shuffle(arr)
        >>> arr
        array([[3, 4, 5], # random
               [6, 7, 8],
               [0, 1, 2]])

        """
        cdef:
            np.npy_intp i, j, n = len(x), stride, itemsize
            char* x_ptr
            char* buf_ptr

        if isinstance(x, np.ndarray) and not x.flags.writeable:
            raise ValueError('array is read-only')

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
            with self.lock:
                # We trick gcc into providing a specialized implementation for
                # the most common case, yielding a ~33% performance improvement.
                # Note that apparently, only one branch can ever be specialized.
                if itemsize == sizeof(np.npy_intp):
                    self._shuffle_raw(n, sizeof(np.npy_intp), stride, x_ptr, buf_ptr)
                else:
                    self._shuffle_raw(n, itemsize, stride, x_ptr, buf_ptr)
        elif isinstance(x, np.ndarray):
            if x.size == 0:
                # shuffling is a no-op
                return

            if x.ndim == 1 and x.dtype.type is np.object_:
                warnings.warn(
                        "Shuffling a one dimensional array subclass containing "
                        "objects gives incorrect results for most array "
                        "subclasses.  "
                        "Please use the new random number API instead: "
                        "https://numpy.org/doc/stable/reference/random/index.html\n"
                        "The new API fixes this issue. This version will not "
                        "be fixed due to stability guarantees of the API.",
                        UserWarning, stacklevel=1)  # Cython adds no stacklevel

            buf = np.empty_like(x[0, ...])
            with self.lock:
                for i in reversed(range(1, n)):
                    j = random_interval(&self._bitgen, i)
                    if i == j:
                        continue  # i == j is not needed and memcpy is undefined.
                    buf[...] = x[j]
                    x[j] = x[i]
                    x[i] = buf
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

            with self.lock:
                for i in reversed(range(1, n)):
                    j = random_interval(&self._bitgen, i)
                    x[i], x[j] = x[j], x[i]

    cdef inline _shuffle_raw(self, np.npy_intp n, np.npy_intp itemsize,
                             np.npy_intp stride, char* data, char* buf):
        cdef np.npy_intp i, j
        for i in reversed(range(1, n)):
            j = random_interval(&self._bitgen, i)
            string.memcpy(buf, data + j * stride, itemsize)
            string.memcpy(data + j * stride, data + i * stride, itemsize)
            string.memcpy(data + i * stride, buf, itemsize)

    def permutation(self, object x):
        """
        permutation(x)

        Randomly permute a sequence, or return a permuted range.

        If `x` is a multi-dimensional array, it is only shuffled along its
        first index.

        .. note::
            New code should use the
            `~numpy.random.Generator.permutation`
            method of a `~numpy.random.Generator` instance instead;
            please see the :ref:`random-quick-start`.

        Parameters
        ----------
        x : int or array_like
            If `x` is an integer, randomly permute ``np.arange(x)``.
            If `x` is an array, make a copy and shuffle the elements
            randomly.

        Returns
        -------
        out : ndarray
            Permuted sequence or array range.

        See Also
        --------
        random.Generator.permutation: which should be used for new code.

        Examples
        --------
        >>> np.random.permutation(10)
        array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random

        >>> np.random.permutation([1, 4, 9, 12, 15])
        array([15,  1,  9,  4, 12]) # random

        >>> arr = np.arange(9).reshape((3, 3))
        >>> np.random.permutation(arr)
        array([[6, 7, 8], # random
               [0, 1, 2],
               [3, 4, 5]])

        """

        if isinstance(x, (int, np.integer)):
            arr = np.arange(x)
            self.shuffle(arr)
            return arr

        arr = np.asarray(x)
        if arr.ndim < 1:
            raise IndexError("x must be an integer or at least 1-dimensional")

        # shuffle has fast-path for 1-d
        if arr.ndim == 1:
            # Return a copy if same memory
            if np.may_share_memory(arr, x):
                arr = np.array(arr)
            self.shuffle(arr)
            return arr

        # Shuffle index array, dtype to ensure fast path
        idx = np.arange(arr.shape[0], dtype=np.intp)
        self.shuffle(idx)
        return arr[idx]

_rand = RandomState()

beta = _rand.beta
binomial = _rand.binomial
bytes = _rand.bytes
chisquare = _rand.chisquare
choice = _rand.choice
dirichlet = _rand.dirichlet
exponential = _rand.exponential
f = _rand.f
gamma = _rand.gamma
get_state = _rand.get_state
geometric = _rand.geometric
gumbel = _rand.gumbel
hypergeometric = _rand.hypergeometric
laplace = _rand.laplace
logistic = _rand.logistic
lognormal = _rand.lognormal
logseries = _rand.logseries
multinomial = _rand.multinomial
multivariate_normal = _rand.multivariate_normal
negative_binomial = _rand.negative_binomial
noncentral_chisquare = _rand.noncentral_chisquare
noncentral_f = _rand.noncentral_f
normal = _rand.normal
pareto = _rand.pareto
permutation = _rand.permutation
poisson = _rand.poisson
power = _rand.power
rand = _rand.rand
randint = _rand.randint
randn = _rand.randn
random = _rand.random
random_integers = _rand.random_integers
random_sample = _rand.random_sample
rayleigh = _rand.rayleigh
set_state = _rand.set_state
shuffle = _rand.shuffle
standard_cauchy = _rand.standard_cauchy
standard_exponential = _rand.standard_exponential
standard_gamma = _rand.standard_gamma
standard_normal = _rand.standard_normal
standard_t = _rand.standard_t
triangular = _rand.triangular
uniform = _rand.uniform
vonmises = _rand.vonmises
wald = _rand.wald
weibull = _rand.weibull
zipf = _rand.zipf

def seed(seed=None):
    """
    seed(seed=None)

    Reseed the singleton RandomState instance.

    Notes
    -----
    This is a convenience, legacy function that exists to support
    older code that uses the singleton RandomState. Best practice
    is to use a dedicated ``Generator`` instance rather than
    the random variate generation methods exposed directly in
    the random module.

    See Also
    --------
    numpy.random.Generator
    """
    if isinstance(_rand._bit_generator, _MT19937):
        return _rand.seed(seed)
    else:
        bg_type = type(_rand._bit_generator)
        _rand._bit_generator.state = bg_type(seed).state

def get_bit_generator():
    """
    Returns the singleton RandomState's bit generator

    Returns
    -------
    BitGenerator
        The bit generator that underlies the singleton RandomState instance

    Notes
    -----
    The singleton RandomState provides the random variate generators in the
    ``numpy.random`` namespace. This function, and its counterpart set method,
    provides a path to hot-swap the default MT19937 bit generator with a
    user provided alternative. These function are intended to provide
    a continuous path where a single underlying bit generator can be
    used both with an instance of ``Generator`` and with the singleton
    instance of RandomState.

    See Also
    --------
    set_bit_generator
    numpy.random.Generator
    """
    return _rand._bit_generator

def set_bit_generator(bitgen):
    """
    Sets the singleton RandomState's bit generator

    Parameters
    ----------
    bitgen
        A bit generator instance

    Notes
    -----
    The singleton RandomState provides the random variate generators in the
    ``numpy.random``namespace. This function, and its counterpart get method,
    provides a path to hot-swap the default MT19937 bit generator with a
    user provided alternative. These function are intended to provide
    a continuous path where a single underlying bit generator can be
    used both with an instance of ``Generator`` and with the singleton
    instance of RandomState.

    See Also
    --------
    get_bit_generator
    numpy.random.Generator
    """
    cdef RandomState singleton
    singleton = _rand
    singleton._initialize_bit_generator(bitgen)


# Old aliases that should not be removed
def sample(*args, **kwargs):
    """
    This is an alias of `random_sample`. See `random_sample`  for the complete
    documentation.
    """
    return _rand.random_sample(*args, **kwargs)

def ranf(*args, **kwargs):
    """
    This is an alias of `random_sample`. See `random_sample`  for the complete
    documentation.
    """
    return _rand.random_sample(*args, **kwargs)

__all__ = [
    'beta',
    'binomial',
    'bytes',
    'chisquare',
    'choice',
    'dirichlet',
    'exponential',
    'f',
    'gamma',
    'geometric',
    'get_bit_generator',
    'get_state',
    'gumbel',
    'hypergeometric',
    'laplace',
    'logistic',
    'lognormal',
    'logseries',
    'multinomial',
    'multivariate_normal',
    'negative_binomial',
    'noncentral_chisquare',
    'noncentral_f',
    'normal',
    'pareto',
    'permutation',
    'poisson',
    'power',
    'rand',
    'randint',
    'randn',
    'random',
    'random_integers',
    'random_sample',
    'ranf',
    'rayleigh',
    'sample',
    'seed',
    'set_bit_generator',
    'set_state',
    'shuffle',
    'standard_cauchy',
    'standard_exponential',
    'standard_gamma',
    'standard_normal',
    'standard_t',
    'triangular',
    'uniform',
    'vonmises',
    'wald',
    'weibull',
    'zipf',
    'RandomState',
]
