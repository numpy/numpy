Cython API for random
---------------------

.. currentmodule:: numpy.random

Typed versions of many of the `Generator` and `BitGenerator` can be accessed
directly from Cython: the complete list is given below.

The ``_bit_generator`` module is usable via::

    cimport numpy.random._bit_generator

It provides function pointers for quickly accessing the next bytes in the
`BitGenerator`::

    struct bitgen:
        void *state
        uint64_t (*next_uint64)(void *st) nogil
        uint32_t (*next_uint32)(void *st) nogil
        double (*next_double)(void *st) nogil
        uint64_t (*next_raw)(void *st) nogil

    ctypedef bitgen bitgen_t

See `extending` for examples of using these functions.

The ``_generator`` module is usable via::

    cimport numpy.random._generator

It provides low-level functions for various distributions. All the functions require a ``bitgen_t`` BitGenerator structure. The functions are named with the followig cconventions:

- "standard" refers to the reference values for any parameters. For instance
  "standard_uniform" means a uniform distribution on the interval ``0.0`` to
  ``1.0``

- "fill" functions will fill the provided ``out`` with ``cnt`` values.

- The functions without "standard" in their name require additional parameters
  to describe the distributions.

.. c:function:: double random_standard_uniform(bitgen_t *bitgen_state)

.. c:function:: void random_standard_uniform_fill(bitgen_t* bitgen_state, np.npy_intp cnt, double *out)

.. c:function:: double random_standard_exponential(bitgen_t *bitgen_state)

.. c:function:: void random_standard_exponential_fill(bitgen_t *bitgen_state, np.npy_intp cnt, double *out)

.. c:function:: double random_standard_exponential_zig(bitgen_t *bitgen_state)

.. c:function:: void random_standard_exponential_zig_fill(bitgen_t *bitgen_state, np.npy_intp cnt, double *out)

.. c:function:: double random_standard_normal(bitgen_t* bitgen_state)

.. c:function:: void random_standard_normal_fill(bitgen_t *bitgen_state, np.npy_intp count, double *out)

.. c:function:: void random_standard_normal_fill_f(bitgen_t *bitgen_state, np.npy_intp count, float *out)

.. c:function:: double random_standard_gamma(bitgen_t *bitgen_state, double shape)

.. c:function:: float random_standard_uniform_f(bitgen_t *bitgen_state)

.. c:function:: void random_standard_uniform_fill_f(bitgen_t* bitgen_state, np.npy_intp cnt, float *out)

.. c:function:: float random_standard_exponential_f(bitgen_t *bitgen_state)

.. c:function:: float random_standard_exponential_zig_f(bitgen_t *bitgen_state)

.. c:function:: void random_standard_exponential_fill_f(bitgen_t *bitgen_state, np.npy_intp cnt, float *out)

.. c:function:: void random_standard_exponential_zig_fill_f(bitgen_t *bitgen_state, np.npy_intp cnt, float *out)

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

.. c:function:: double random_noncentral_chisquare(bitgen_t *bitgen_state, double df,
                                       double nonc)
.. c:function:: double random_noncentral_f(bitgen_t *bitgen_state, double dfnum,
                               double dfden, double nonc)
.. c:function:: double random_wald(bitgen_t *bitgen_state, double mean, double scale)

.. c:function:: double random_vonmises(bitgen_t *bitgen_state, double mu, double kappa)

.. c:function:: double random_triangular(bitgen_t *bitgen_state, double left, double mode,
                             double right)

.. c:function:: int64_t random_poisson(bitgen_t *bitgen_state, double lam)

.. c:function:: int64_t random_negative_binomial(bitgen_t *bitgen_state, double n, double p)

.. c:function:: int64_t random_binomial(bitgen_t *bitgen_state, double p, int64_t n, binomial_t *binomial)

.. c:function:: int64_t random_logseries(bitgen_t *bitgen_state, double p)

.. c:function:: int64_t random_geometric_search(bitgen_t *bitgen_state, double p)

.. c:function:: int64_t random_geometric_inversion(bitgen_t *bitgen_state, double p)

.. c:function:: int64_t random_geometric(bitgen_t *bitgen_state, double p)

.. c:function:: int64_t random_zipf(bitgen_t *bitgen_state, double a)

.. c:function:: int64_t random_hypergeometric(bitgen_t *bitgen_state, int64_t good, int64_t bad,
                                    int64_t sample)

.. c:function:: uint64_t random_interval(bitgen_t *bitgen_state, uint64_t max)

.. c:function:: void random_multinomial(bitgen_t *bitgen_state, int64_t n, int64_t *mnix,
                            double *pix, np.npy_intp d, binomial_t *binomial)

.. c:function:: int random_mvhg_count(bitgen_t *bitgen_state,
                          int64_t total,
                          size_t num_colors, int64_t *colors,
                          int64_t nsample,
                          size_t num_variates, int64_t *variates)

.. c:function:: void random_mvhg_marginals(bitgen_t *bitgen_state,
                               int64_t total,
                               size_t num_colors, int64_t *colors,
                               int64_t nsample,
                               size_t num_variates, int64_t *variates)

Generate a single integer

.. c:function:: int64_t random_positive_int64(bitgen_t *bitgen_state)

.. c:function:: int32_t random_positive_int32(bitgen_t *bitgen_state)

.. c:function:: int64_t random_positive_int(bitgen_t *bitgen_state)

.. c:function:: uint64_t random_uint(bitgen_t *bitgen_state)


Generate random uint64 numbers in closed interval [off, off + rng].

.. c:function:: uint64_t random_bounded_uint64(bitgen_t *bitgen_state,
                                   uint64_t off, uint64_t rng,
                                   uint64_t mask, bint use_masked)


