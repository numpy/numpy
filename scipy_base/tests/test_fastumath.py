import unittest
import sys
from scipy_test.testing import *
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

if __name__ == "__main__":
    ScipyTest('scipy_base.fastumath').run()
