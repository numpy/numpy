""" Test functions for limits module.
"""

from numpy.testing import *
set_package_path()
import numpy.lib;reload(numpy.lib)
from numpy.lib.getlimits import finfo, iinfo
from numpy import single,double,longdouble
import numpy as N
restore_path()

##################################################

class test_python_float(NumpyTestCase):
    def check_singleton(self):
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype),id(ftype2))

class test_single(NumpyTestCase):
    def check_singleton(self):
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype),id(ftype2))

class test_double(NumpyTestCase):
    def check_singleton(self):
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype),id(ftype2))

class test_longdouble(NumpyTestCase):
    def check_singleton(self,level=2):
        ftype = finfo(longdouble)
        ftype2 = finfo(longdouble)
        assert_equal(id(ftype),id(ftype2))

class test_iinfo(NumpyTestCase):
    def check_basic(self):
        dts = zip(['i1', 'i2', 'i4', 'i8',
                   'u1', 'u2', 'u4', 'u8'],
                  [N.int8, N.int16, N.int32, N.int64,
                   N.uint8, N.uint16, N.uint32, N.uint64])
        for dt1, dt2 in dts:
            assert_equal(iinfo(dt1).min, iinfo(dt2).min)
            assert_equal(iinfo(dt1).max, iinfo(dt2).max)
        self.assertRaises(ValueError, iinfo, 'f4')

    def check_unsigned_max(self):
        types = N.sctypes['uint']
        for T in types:
            assert_equal(iinfo(T).max, T(-1))

if __name__ == "__main__":
    NumpyTest().run()
