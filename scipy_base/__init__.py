
import Numeric
from Numeric import *
import fastumath
import limits

from type_check import *
from index_tricks import *
from function_base import *
from shape_base import *
from matrix_base import *
from transform_base import *

from polynomial import *
from scimath import *

# needs fastumath
Inf = inf = fastumath.PINF
try:
    NAN = NaN = nan = fastumath.NAN
except AttributeError:
    NaN = NAN = nan = fastumath.PINF - fastumath.PINF


#---- testing ----#

def test(level=10):
    import unittest
    runner = unittest.TextTestRunner()
    runner.run(test_suite())
    return runner

def test_suite(level=1):
    import scipy_base.testing
    import scipy_base
    this_mod = scipy_base
    # testing is the module that actually does all the testing...
    ignore = ['testing']
    return scipy_base.testing.harvest_test_suites(this_mod,ignore = ignore,
                                                  level=level)


