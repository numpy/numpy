# To get sub-modules
from info import __doc__

from linalg import *

try:
    import scipy.linalg
    inv = scipy.linalg.inv
    svd = scipy.linalg.svd
except ImportError:
    pass

from scipy.test.testing import ScipyTest 
test = ScipyTest().test
