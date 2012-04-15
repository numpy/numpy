"""
========================
Random Number Generation
========================

==================== =========================================================
Utility functions
==============================================================================
random               Uniformly distributed values of a given shape.
bytes                Uniformly distributed random bytes.
random_integers      Uniformly distributed integers in a given range.
random_sample        Uniformly distributed floats in a given range.
random               Alias for random_sample
ranf                 Alias for random_sample
sample               Alias for random_sample
choice               Generate a weighted random sample from a given array-like
permutation          Randomly permute a sequence / generate a random sequence.
shuffle              Randomly permute a sequence in place.
seed                 Seed the random number generator.
==================== =========================================================

==================== =========================================================
Compatibility functions
==============================================================================
rand                 Uniformly distributed values.
randn                Normally distributed values.
ranf                 Uniformly distributed floating point numbers.
randint              Uniformly distributed integers in a given range.
==================== =========================================================

==================== =========================================================
Univariate distributions
==============================================================================
beta                 Beta distribution over ``[0, 1]``.
binomial             Binomial distribution.
chisquare            :math:`\\chi^2` distribution.
exponential          Exponential distribution.
f                    F (Fisher-Snedecor) distribution.
gamma                Gamma distribution.
geometric            Geometric distribution.
gumbel               Gumbel distribution.
hypergeometric       Hypergeometric distribution.
laplace              Laplace distribution.
logistic             Logistic distribution.
lognormal            Log-normal distribution.
logseries            Logarithmic series distribution.
negative_binomial    Negative binomial distribution.
noncentral_chisquare Non-central chi-square distribution.
noncentral_f         Non-central F distribution.
normal               Normal / Gaussian distribution.
pareto               Pareto distribution.
poisson              Poisson distribution.
power                Power distribution.
rayleigh             Rayleigh distribution.
triangular           Triangular distribution.
uniform              Uniform distribution.
vonmises             Von Mises circular distribution.
wald                 Wald (inverse Gaussian) distribution.
weibull              Weibull distribution.
zipf                 Zipf's distribution over ranked data.
==================== =========================================================

==================== =========================================================
Multivariate distributions
==============================================================================
dirichlet            Multivariate generalization of Beta distribution.
multinomial          Multivariate generalization of the binomial distribution.
multivariate_normal  Multivariate generalization of the normal distribution.
==================== =========================================================

==================== =========================================================
Standard distributions
==============================================================================
standard_cauchy      Standard Cauchy-Lorentz distribution.
standard_exponential Standard exponential distribution.
standard_gamma       Standard Gamma distribution.
standard_normal      Standard normal distribution.
standard_t           Standard Student's t-distribution.
==================== =========================================================

==================== =========================================================
Internal functions
==============================================================================
get_state            Get tuple representing internal state of generator.
set_state            Set state of generator.
==================== =========================================================

"""
# To get sub-modules
from info import __doc__, __all__

import warnings
from numpy.testing.utils import WarningManager

warn_ctx = WarningManager()
warn_ctx.__enter__()
try:
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
    from mtrand import *
finally:
    warn_ctx.__exit__()

# Some aliases:
ranf = random = sample = random_sample
__all__.extend(['ranf','random','sample'])

def __RandomState_ctor():
    """Return a RandomState instance.

    This function exists solely to assist (un)pickling.
    """
    return RandomState()

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
