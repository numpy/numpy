
import Numeric
from Numeric import *
import fastumath
import limits

from utility import *
from convenience import *
from polynomial import *
from scimath import *
from helpmod import help, source
from Matrix import Matrix as mat
Mat = mat  # deprecated

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
    return scipy_base.testing.harvest_test_suites(this_mod,level=level)


