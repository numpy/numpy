.. _legacy:

Legacy Random Generation
------------------------
The `~mtrand.RandomState` provides access to
legacy generators. This generator is considered frozen and will have
no further improvements.  It is guaranteed to produce the same values
as the final point release of NumPy v1.16. These all depend on Box-Muller
normals or inverse CDF exponentials or gammas. This class should only be used
if it is essential to have randoms that are identical to what
would have been produced by NumPy.

`~mtrand.RandomState` adds additional information
to the state which is required when using Box-Muller normals since these
are produced in pairs. It is important to use
`~mtrand.RandomState.get_state`, and not the underlying bit generators
`state`, when accessing the state so that these extra values are saved.

.. warning::

  :class:`~randomgen.legacy.LegacyGenerator` only contains functions
  that have changed.  Since it does not contain other functions, it
  is not directly possible to replace :class:`~numpy.random.RandomState`.
  In order to full replace :class:`~numpy.random.RandomState`, it is
  necessary to use both :class:`~randomgen.legacy.LegacyGenerator`
  and :class:`~randomgen.generator.RandomGenerator` both driven
  by the same basic RNG. Methods present in :class:`~randomgen.legacy.LegacyGenerator`
  must be called from :class:`~randomgen.legacy.LegacyGenerator`.  Other Methods
  should be called from :class:`~randomgen.generator.RandomGenerator`.


.. code-block:: python

   from numpy.random import MT19937
   from numpy.random import RandomState

   # Use same seed
   rs = RandomState(12345)
   mt19937 = MT19937(12345)
   lg = RandomState(mt19937)

   # Identical output
   rs.standard_normal()
   lg.standard_normal()

   rs.random()
   lg.random()

   rs.standard_exponential()
   lg.standard_exponential()


.. currentmodule:: numpy.random.mtrand

.. autoclass:: RandomState
	:exclude-members:

Seeding and State
=================

.. autosummary::
   :toctree: generated/

   ~RandomState.get_state
   ~RandomState.set_state
   ~RandomState.seed

Simple random data
==================
.. autosummary::
   :toctree: generated/

   ~RandomState.rand
   ~RandomState.randn
   ~RandomState.randint
   ~RandomState.random_integers
   ~RandomState.random_sample
   ~RandomState.choice
   ~RandomState.bytes

Permutations
============
.. autosummary::
   :toctree: generated/

   ~RandomState.shuffle
   ~RandomState.permutation

Distributions
=============
.. autosummary::
   :toctree: generated/

   ~RandomState.beta
   ~RandomState.binomial
   ~RandomState.chisquare
   ~RandomState.dirichlet
   ~RandomState.exponential
   ~RandomState.f
   ~RandomState.gamma
   ~RandomState.geometric
   ~RandomState.gumbel
   ~RandomState.hypergeometric
   ~RandomState.laplace
   ~RandomState.logistic
   ~RandomState.lognormal
   ~RandomState.logseries
   ~RandomState.multinomial
   ~RandomState.multivariate_normal
   ~RandomState.negative_binomial
   ~RandomState.noncentral_chisquare
   ~RandomState.noncentral_f
   ~RandomState.normal
   ~RandomState.pareto
   ~RandomState.poisson
   ~RandomState.power
   ~RandomState.rayleigh
   ~RandomState.standard_cauchy
   ~RandomState.standard_exponential
   ~RandomState.standard_gamma
   ~RandomState.standard_normal
   ~RandomState.standard_t
   ~RandomState.triangular
   ~RandomState.uniform
   ~RandomState.vonmises
   ~RandomState.wald
   ~RandomState.weibull
   ~RandomState.zipf
