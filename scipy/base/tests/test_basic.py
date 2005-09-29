import unittest
import sys
from scipy.test.testing import *
set_package_path()
import scipy.base;reload(scipy.base)
#from scipy.base import fastumath;reload(fastumath)
del sys.path[0]


class test_maximum(ScipyTestCase):
    def check_reduce_complex(self):
        x = [1,2]
        assert_equal(maximum.reduce([1,2j]),1)

class test_minimum(ScipyTestCase):
    def check_reduce_complex(self):
        x = [1,2]
        assert_equal(minimum.reduce([1,2j]),2j)

if __name__ == "__main__":
    ScipyTest('scipy.base').run()
