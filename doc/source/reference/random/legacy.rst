.. currentmodule:: numpy.random

.. _legacy:

Legacy Random Generation
------------------------
The `~mtrand.RandomState` provides access to
legacy generators. This generator is considered frozen and will have
no further improvements.  It is guaranteed to produce the same values
as the final point release of NumPy v1.16. These all depend on Box-Muller
normals or inverse CDF exponentials or gammas. This class should only be used
if it is essential to have randoms that are identical to what
would have been produced by previous versions of NumPy.

`~mtrand.RandomState` adds additional information
to the state which is required when using Box-Muller normals since these
are produced in pairs. It is important to use
`~mtrand.RandomState.get_state`, and not the underlying bit generators
`state`, when accessing the state so that these extra values are saved.

Although we provide the `~mt19937.MT19937` BitGenerator for use independent of
`~mtrand.RandomState`, note that its default seeding uses `~SeedSequence`
rather than the legacy seeding algorithm. `~mtrand.RandomState` will use the
legacy seeding algorithm. The methods to use the legacy seeding algorithm are
currently private as the main reason to use them is just to implement
`~mtrand.RandomState`. However, one can reset the state of `~mt19937.MT19937`
using the state of the `~mtrand.RandomState`:

.. code-block:: python

   from numpy.random import MT19937
   from numpy.random import RandomState

   rs = RandomState(12345)
   mt19937 = MT19937()
   mt19937.state = rs.get_state()
   rs2 = RandomState(mt19937)

   # Same output
   rs.standard_normal()
   rs2.standard_normal()

   rs.random()
   rs2.random()

   rs.standard_exponential()
   rs2.standard_exponential()


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
