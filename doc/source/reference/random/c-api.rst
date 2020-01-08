Cython API for random
---------------------

.. currentmodule:: numpy.random

Typed versions of many of the `Generator` and `BitGenerator` methods as well as
the classes themselves can be accessed directly from Cython via

.. code-block:: cython

    cimport numpy.random

C API for random
----------------

Access to various distributions is available via Cython or C-wrapper libraries
like CFFI. All the functions accept a :c:type:`bitgen_t` as their first argument.

.. c:type:: bitgen_t

    The :c:type:`bitgen_t` holds the current state of the BitGenerator and
    pointers to functions that return standard C types while advancing the
    state.

    .. code-block:: c

        struct bitgen:
            void *state
            npy_uint64 (*next_uint64)(void *st) nogil
            uint32_t (*next_uint32)(void *st) nogil
            double (*next_double)(void *st) nogil
            npy_uint64 (*next_raw)(void *st) nogil

        ctypedef bitgen bitgen_t

See :doc:`extending` for examples of using these functions.

The functions are named with the following conventions:

- "standard" refers to the reference values for any parameters. For instance
  "standard_uniform" means a uniform distribution on the interval ``0.0`` to
  ``1.0``

- "fill" functions will fill the provided ``out`` with ``cnt`` values.

- The functions without "standard" in their name require additional parameters
  to describe the distributions.

- ``zig`` in the name are based on a ziggurat lookup algorithm is used instead
  of calculating the ``log``, which is significantly faster. The non-ziggurat
  variants are used in corner cases and for legacy compatibility.


.. c:function:: double random_standard_uniform(bitgen_t *bitgen_state)

.. c:function:: void random_standard_uniform_fill(bitgen_t* bitgen_state, npy_intp cnt, double *out)

.. c:function:: double random_standard_exponential(bitgen_t *bitgen_state)

.. c:function:: void random_standard_exponential_fill(bitgen_t *bitgen_state, npy_intp cnt, double *out)

.. c:function:: double random_standard_normal(bitgen_t* bitgen_state)

.. c:function:: void random_standard_normal_fill(bitgen_t *bitgen_state, npy_intp count, double *out)

.. c:function:: void random_standard_normal_fill_f(bitgen_t *bitgen_state, npy_intp count, float *out)

.. c:function:: double random_standard_gamma(bitgen_t *bitgen_state, double shape)

.. c:function:: float random_standard_uniform_f(bitgen_t *bitgen_state)

.. c:function:: void random_standard_uniform_fill_f(bitgen_t* bitgen_state, npy_intp cnt, float *out)

.. c:function:: float random_standard_exponential_f(bitgen_t *bitgen_state)

.. c:function:: void random_standard_exponential_fill_f(bitgen_t *bitgen_state, npy_intp cnt, float *out)

.. c:function:: float random_standard_normal_f(bitgen_t* bitgen_state)

.. c:function:: float random_standard_gamma_f(bitgen_t *bitgen_state, float shape)

.. c:function:: double random_normal(bitgen_t *bitgen_state, double loc, double scale)

.. c:function:: double random_gamma(bitgen_t *bitgen_state, double shape, double scale)

.. c:function:: float random_gamma_f(bitgen_t *bitgen_state, float shape, float scale)

.. c:function:: double random_exponential(bitgen_t *bitgen_state, double scale)

.. c:function:: double random_uniform(bitgen_t *bitgen_state, double lower, double range)

.. c:function:: double random_beta(bitgen_t *bitgen_state, double a, double b)

.. c:function:: double random_chisquare(bitgen_t *bitgen_state, double df)

.. c:function:: double random_f(bitgen_t *bitgen_state, double dfnum, double dfden)

.. c:function:: double random_standard_cauchy(bitgen_t *bitgen_state)

.. c:function:: double random_pareto(bitgen_t *bitgen_state, double a)

.. c:function:: double random_weibull(bitgen_t *bitgen_state, double a)

.. c:function:: double random_power(bitgen_t *bitgen_state, double a)

.. c:function:: double random_laplace(bitgen_t *bitgen_state, double loc, double scale)

.. c:function:: double random_gumbel(bitgen_t *bitgen_state, double loc, double scale)

.. c:function:: double random_logistic(bitgen_t *bitgen_state, double loc, double scale)

.. c:function:: double random_lognormal(bitgen_t *bitgen_state, double mean, double sigma)

.. c:function:: double random_rayleigh(bitgen_t *bitgen_state, double mode)

.. c:function:: double random_standard_t(bitgen_t *bitgen_state, double df)

.. c:function:: double random_noncentral_chisquare(bitgen_t *bitgen_state, double df, double nonc)
.. c:function:: double random_noncentral_f(bitgen_t *bitgen_state, double dfnum, double dfden, double nonc)
.. c:function:: double random_wald(bitgen_t *bitgen_state, double mean, double scale)

.. c:function:: double random_vonmises(bitgen_t *bitgen_state, double mu, double kappa)

.. c:function:: double random_triangular(bitgen_t *bitgen_state, double left, double mode, double right)

.. c:function:: npy_int64 random_poisson(bitgen_t *bitgen_state, double lam)

.. c:function:: npy_int64 random_negative_binomial(bitgen_t *bitgen_state, double n, double p)

.. c:type:: binomial_t

    .. code-block:: c

        typedef struct s_binomial_t {
          int has_binomial; /* !=0: following parameters initialized for binomial */
          double psave;
          RAND_INT_TYPE nsave;
          double r;
          double q;
          double fm;
          RAND_INT_TYPE m;
          double p1;
          double xm;
          double xl;
          double xr;
          double c;
          double laml;
          double lamr;
          double p2;
          double p3;
          double p4;
        } binomial_t;
     

.. c:function:: npy_int64 random_binomial(bitgen_t *bitgen_state, double p, npy_int64 n, binomial_t *binomial)

.. c:function:: npy_int64 random_logseries(bitgen_t *bitgen_state, double p)

.. c:function:: npy_int64 random_geometric_search(bitgen_t *bitgen_state, double p)

.. c:function:: npy_int64 random_geometric_inversion(bitgen_t *bitgen_state, double p)

.. c:function:: npy_int64 random_geometric(bitgen_t *bitgen_state, double p)

.. c:function:: npy_int64 random_zipf(bitgen_t *bitgen_state, double a)

.. c:function:: npy_int64 random_hypergeometric(bitgen_t *bitgen_state, npy_int64 good, npy_int64 bad, npy_int64 sample)

.. c:function:: npy_uint64 random_interval(bitgen_t *bitgen_state, npy_uint64 max)

.. c:function:: void random_multinomial(bitgen_t *bitgen_state, npy_int64 n, npy_int64 *mnix, double *pix, npy_intp d, binomial_t *binomial)

.. c:function:: int random_multivariate_hypergeometric_count(bitgen_t *bitgen_state, npy_int64 total, size_t num_colors, npy_int64 *colors, npy_int64 nsample, size_t num_variates, npy_int64 *variates)

.. c:function:: void random_multivariate_hypergeometric_marginals(bitgen_t *bitgen_state, npy_int64 total, size_t num_colors, npy_int64 *colors, npy_int64 nsample, size_t num_variates, npy_int64 *variates)

Generate a single integer

.. c:function:: npy_int64 random_positive_int64(bitgen_t *bitgen_state)

.. c:function:: npy_int32 random_positive_int32(bitgen_t *bitgen_state)

.. c:function:: npy_int64 random_positive_int(bitgen_t *bitgen_state)

.. c:function:: npy_uint64 random_uint(bitgen_t *bitgen_state)


Generate random uint64 numbers in closed interval [off, off + rng].

.. c:function:: npy_uint64 random_bounded_uint64(bitgen_t *bitgen_state, npy_uint64 off, npy_uint64 rng, npy_uint64 mask, bint use_masked)


