"""Re-create the RNG interface from Numeric.

Replace import RNG with import numpy.oldnumeric.rng as RNG.
It is for backwards compatibility only.

"""
from __future__ import division, absolute_import, print_function

__all__ = ['CreateGenerator', 'ExponentialDistribution', 'LogNormalDistribution',
           'NormalDistribution', 'UniformDistribution', 'error', 'ranf',
           'default_distribution', 'random_sample', 'standard_generator']

import numpy.random.mtrand as mt
import math

class error(Exception):
    pass

class Distribution(object):
    def __init__(self, meth, *args):
        self._meth = meth
        self._args = args

    def density(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.density(x)

    def _onesample(self, rng):
        return getattr(rng, self._meth)(*self._args)

    def _sample(self, rng, n):
        kwds = {'size' : n}
        return getattr(rng, self._meth)(*self._args, **kwds)


class ExponentialDistribution(Distribution):
    def __init__(self, lambda_):
        if (lambda_ <= 0):
            raise error("parameter must be positive")
        Distribution.__init__(self, 'exponential', lambda_)

    def density(x):
        if x < 0:
            return 0.0
        else:
            lambda_ = self._args[0]
            return lambda_*math.exp(-lambda_*x)

class LogNormalDistribution(Distribution):
    def __init__(self, m, s):
        m = float(m)
        s = float(s)
        if (s <= 0):
            raise error("standard deviation must be positive")
        Distribution.__init__(self, 'lognormal', m, s)
        sn = math.log(1.0+s*s/(m*m));
        self._mn = math.log(m)-0.5*sn
        self._sn = math.sqrt(sn)
        self._fac = 1.0/math.sqrt(2*math.pi)/self._sn

    def density(x):
        m, s = self._args
        y = (math.log(x)-self._mn)/self._sn
        return self._fac*math.exp(-0.5*y*y)/x


class NormalDistribution(Distribution):
    def __init__(self, m, s):
        m = float(m)
        s = float(s)
        if (s <= 0):
            raise error("standard deviation must be positive")
        Distribution.__init__(self, 'normal', m, s)
        self._fac = 1.0/math.sqrt(2*math.pi)/s

    def density(x):
        m, s = self._args
        y = (x-m)/s
        return self._fac*math.exp(-0.5*y*y)

class UniformDistribution(Distribution):
    def __init__(self, a, b):
        a = float(a)
        b = float(b)
        width = b-a
        if (width <=0):
            raise error("width of uniform distribution must be > 0")
        Distribution.__init__(self, 'uniform', a, b)
        self._fac = 1.0/width

    def density(x):
        a, b = self._args
        if (x < a) or (x >= b):
            return 0.0
        else:
            return self._fac

default_distribution = UniformDistribution(0.0, 1.0)

class CreateGenerator(object):
    def __init__(self, seed, dist=None):
        if seed <= 0:
            self._rng = mt.RandomState()
        elif seed > 0:
            self._rng = mt.RandomState(seed)
        if dist is None:
            dist = default_distribution
        if not isinstance(dist, Distribution):
            raise error("Not a distribution object")
        self._dist = dist

    def ranf(self):
        return self._dist._onesample(self._rng)

    def sample(self, n):
        return self._dist._sample(self._rng, n)


standard_generator = CreateGenerator(-1)

def ranf():
    "ranf() = a random number from the standard generator."
    return standard_generator.ranf()

def random_sample(*n):
    """random_sample(n) = array of n random numbers;

    random_sample(n1, n2, ...)= random array of shape (n1, n2, ..)"""

    if not n:
        return standard_generator.ranf()
    m = 1
    for i in n:
        m = m * i
    return standard_generator.sample(m).reshape(*n)
