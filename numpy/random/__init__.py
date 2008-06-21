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

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench
