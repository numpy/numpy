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

   ~Generator.bit_generator

Simple random data
==================
.. autosummary::
   :toctree: generated/

   ~Generator.integers
   ~Generator.random
   ~Generator.choice
   ~Generator.bytes

Permutations
============
.. autosummary::
   :toctree: generated/

   ~Generator.shuffle
   ~Generator.permutation

Distributions
=============
.. autosummary::
   :toctree: generated/

   ~Generator.beta
   ~Generator.binomial
   ~Generator.chisquare
   ~Generator.dirichlet
   ~Generator.exponential
   ~Generator.f
   ~Generator.gamma
   ~Generator.geometric
   ~Generator.gumbel
   ~Generator.hypergeometric
   ~Generator.laplace
   ~Generator.logistic
   ~Generator.lognormal
   ~Generator.logseries
   ~Generator.multinomial
   ~Generator.multivariate_normal
   ~Generator.negative_binomial
   ~Generator.noncentral_chisquare
   ~Generator.noncentral_f
   ~Generator.normal
   ~Generator.pareto
   ~Generator.poisson
   ~Generator.power
   ~Generator.rayleigh
   ~Generator.standard_cauchy
   ~Generator.standard_exponential
   ~Generator.standard_gamma
   ~Generator.standard_normal
   ~Generator.standard_t
   ~Generator.triangular
   ~Generator.uniform
   ~Generator.vonmises
   ~Generator.wald
   ~Generator.weibull
   ~Generator.zipf
