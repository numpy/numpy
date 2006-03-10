# To get sub-modules
from info import __doc__, __all__
from mtrand import *

# Some aliases:
ranf = random = sample = random_sample
__all__.extend(['ranf','random','sample'])

def __RandomState_ctor():
    """Return a RandomState instance.

    This function exists solely to assist (un)pickling.
    """
    return RandomState()

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
