import sys
import unittest
from scipy_test.testing import assert_array_equal, assert_equal
from scipy_test.testing import assert_almost_equal, assert_array_almost_equal

from scipy_test.testing import set_package_path
set_package_path()
import scipy_base;reload(scipy_base)
from scipy_base import *
del sys.path[0]

class test_general_function(unittest.TestCase):

    def check_simple(self):
        def addsubtract(a,b):
            if a > b:
                return a - b
            else:
                return a + b
        f = general_function(addsubtract)
        r = f([0,3,6,9],[1,3,5,7])
        assert_array_equal(r,[1,6,1,2])

####### Testing ##############

def test_suite(level=1):
    suites = []
    if level > 0:
        suites.append( unittest.makeSuite(test_general_function,'check_') )
    if level > 5:
	pass
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10,verbosity=2):
    all_tests = test_suite(level=level)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
