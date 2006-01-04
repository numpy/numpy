""" Test functions for limits module.
"""

from numpy.testing import *
set_package_path()
import numpy.lib;reload(numpy.lib)
from numpy.lib.getlimits import finfo
from numpy import single,double,longdouble
restore_path()

##################################################

class test_python_float(ScipyTestCase):
    def check_singleton(self):
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype),id(ftype2))

class test_single(ScipyTestCase):
    def check_singleton(self):
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype),id(ftype2))

class test_double(ScipyTestCase):
    def check_singleton(self):
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype),id(ftype2))

class test_longdouble(ScipyTestCase):
    def check_singleton(self,level=2):
        ftype = finfo(longdouble)
        ftype2 = finfo(longdouble)
        assert_equal(id(ftype),id(ftype2))

if __name__ == "__main__":
    ScipyTest().run()
