import unittest
from scipy_test.testing import assert_array_equal, assert_equal, rand
from scipy_test.testing import assert_almost_equal, assert_array_almost_equal    

import sys
from scipy_test.testing import set_package_path
from scipy_test.testing import ScipyTestCase
set_package_path()
import scipy_base;reload(scipy_base)
from scipy_base import fastumath;reload(fastumath)
del sys.path[0]


class test_maximum(ScipyTestCase):
    def check_reduce_complex(self):
        x = [1,2]
        assert_equal(fastumath.maximum.reduce([1,2j]),1)

class test_minimum(ScipyTestCase):
    def check_reduce_complex(self):
        x = [1,2]
        assert_equal(fastumath.minimum.reduce([1,2j]),2j)

def test_suite(level=1):
    suites = []
    if level > 0:
        suites.append( unittest.makeSuite(test_maximum,'check_') )
        suites.append( unittest.makeSuite(test_minimum,'check_') )

    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10,verbosity=2):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    if len(sys.argv)>1:
        level = eval(sys.argv[1])
    else:
        level = 1
    test(level)
