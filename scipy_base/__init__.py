
import Numeric
from Numeric import *
import fastumath
import limits

#from utility import *
#from convenience import *

from type_check import *
from index_tricks import *
from function_base import *
from shape_base import *
from matrix_base import *

from polynomial import *
from scimath import *

# needs fastumath
Inf = inf = Numeric.array(1e308)**10
NaN = nan = Numeric.array(0.0) / Numeric.array(0.0)


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
    # ieee_754 gets tested in the type_check module.
    # testing is the module that actually does all the testing...
    ignore = ['ieee_754','testing']
    return scipy_base.testing.harvest_test_suites(this_mod,ignore = ignore,
                                                  level=level)


