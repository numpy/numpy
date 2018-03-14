Random Generator
----------------
The :class:`~randomgen.generator.RandomGenerator` provides access to 
a wide range of distributions, and served as a replacement for 
:class:`~numpy.random.RandomState`.  The main difference between 
the two is that :class:`~randomgen.generator.RandomGenerator` relies
on an additional basic RNG to manage state and generate the random 
bits which are then transformed into random values from useful 
distributions. The default basic RNG used by 
:class:`~randomgen.generator.RandomGenerator` is 
:class:`~randomgen.xoroshiro128.Xoroshiro128`.  The basic RNG can be 
changed by passing an instantized basic RNG to 
:class:`~randomgen.generator.RandomGenerator`. 

.. currentmodule:: randomgen.generator

.. autoclass::
   RandomGenerator

Simple random data
==================
.. autosummary::
   :toctree: generated/

   ~RandomGenerator.rand
   ~RandomGenerator.randn
   ~RandomGenerator.randint
   ~RandomGenerator.random_integers
   ~RandomGenerator.random_sample
   ~RandomGenerator.choice
   ~RandomGenerator.bytes
   ~RandomGenerator.random_uintegers
   ~RandomGenerator.random_raw

Permutations
============
.. autosummary::
   :toctree: generated/

   ~RandomGenerator.shuffle
   ~RandomGenerator.permutation

Distributions
=============
.. autosummary::
   :toctree: generated/

   ~RandomGenerator.beta
   ~RandomGenerator.binomial
   ~RandomGenerator.chisquare
   ~RandomGenerator.complex_normal
   ~RandomGenerator.dirichlet
   ~RandomGenerator.exponential
   ~RandomGenerator.f
   ~RandomGenerator.gamma
   ~RandomGenerator.geometric
   ~RandomGenerator.gumbel
   ~RandomGenerator.hypergeometric
   ~RandomGenerator.laplace
   ~RandomGenerator.logistic
   ~RandomGenerator.lognormal
   ~RandomGenerator.logseries
   ~RandomGenerator.multinomial
   ~RandomGenerator.multivariate_normal
   ~RandomGenerator.negative_binomial
   ~RandomGenerator.noncentral_chisquare
   ~RandomGenerator.noncentral_f
   ~RandomGenerator.normal
   ~RandomGenerator.pareto
   ~RandomGenerator.poisson
   ~RandomGenerator.power
   ~RandomGenerator.rayleigh
   ~RandomGenerator.standard_cauchy
   ~RandomGenerator.standard_exponential
   ~RandomGenerator.standard_gamma
   ~RandomGenerator.standard_normal
   ~RandomGenerator.standard_t
   ~RandomGenerator.triangular
   ~RandomGenerator.uniform
   ~RandomGenerator.vonmises
   ~RandomGenerator.wald
   ~RandomGenerator.weibull
   ~RandomGenerator.zipf