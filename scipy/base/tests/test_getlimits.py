""" Test functions for limits module.

    Currently empty -- not sure how to test these values
    and routines as they are machine dependent.  Suggestions?
"""

import unittest
import sys

from scipy.test.testing import *
set_package_path()
import scipy.base;reload(scipy.base)
from scipy.base.getlimits import finfo
from scipy import single,double,longdouble
del sys.path[0]

##################################################

class test_python_float(unittest.TestCase):
    def check_singleton(self):
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype),id(ftype2))

class test_single(unittest.TestCase):
    def check_singleton(self):
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype),id(ftype2))

class test_double(unittest.TestCase):
    def check_singleton(self):
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype),id(ftype2))

class test_longdouble(unittest.TestCase):
    def check_singleton(self,level=2):
        ftype = finfo(longdouble)
        ftype2 = finfo(longdouble)
        assert_equal(id(ftype),id(ftype2))

if __name__ == "__main__":
    ScipyTest().run()
