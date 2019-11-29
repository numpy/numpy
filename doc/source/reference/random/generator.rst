.. currentmodule:: numpy.random

Random Generator
----------------
The `~Generator` provides access to
a wide range of distributions, and served as a replacement for
:class:`~numpy.random.RandomState`.  The main difference between
the two is that ``Generator`` relies on an additional BitGenerator to
manage state and generate the random bits, which are then transformed into
random values from useful distributions. The default BitGenerator used by
``Generator`` is `~PCG64`.  The BitGenerator
can be changed by passing an instantized BitGenerator to ``Generator``.


.. autofunction:: default_rng

.. autoclass:: Generator
	:exclude-members:

Accessing the BitGenerator
==========================
.. autosummary::
   :toctree: generated/

   ~numpy.random.Generator.bit_generator

Simple random data
==================
.. autosummary::
   :toctree: generated/

   ~numpy.random.Generator.integers
   ~numpy.random.Generator.random
   ~numpy.random.Generator.choice
   ~numpy.random.Generator.bytes

Permutations
============
.. autosummary::
   :toctree: generated/

   ~numpy.random.Generator.shuffle
   ~numpy.random.Generator.permutation

Distributions
=============
.. autosummary::
   :toctree: generated/

   ~numpy.random.Generator.beta
   ~numpy.random.Generator.binomial
   ~numpy.random.Generator.chisquare
   ~numpy.random.Generator.dirichlet
   ~numpy.random.Generator.exponential
   ~numpy.random.Generator.f
   ~numpy.random.Generator.gamma
   ~numpy.random.Generator.geometric
   ~numpy.random.Generator.gumbel
   ~numpy.random.Generator.hypergeometric
   ~numpy.random.Generator.laplace
   ~numpy.random.Generator.logistic
   ~numpy.random.Generator.lognormal
   ~numpy.random.Generator.logseries
   ~numpy.random.Generator.multinomial
   ~numpy.random.Generator.multivariate_hypergeometric
   ~numpy.random.Generator.multivariate_normal
   ~numpy.random.Generator.negative_binomial
   ~numpy.random.Generator.noncentral_chisquare
   ~numpy.random.Generator.noncentral_f
   ~numpy.random.Generator.normal
   ~numpy.random.Generator.pareto
   ~numpy.random.Generator.poisson
   ~numpy.random.Generator.power
   ~numpy.random.Generator.rayleigh
   ~numpy.random.Generator.standard_cauchy
   ~numpy.random.Generator.standard_exponential
   ~numpy.random.Generator.standard_gamma
   ~numpy.random.Generator.standard_normal
   ~numpy.random.Generator.standard_t
   ~numpy.random.Generator.triangular
   ~numpy.random.Generator.uniform
   ~numpy.random.Generator.vonmises
   ~numpy.random.Generator.wald
   ~numpy.random.Generator.weibull
   ~numpy.random.Generator.zipf
