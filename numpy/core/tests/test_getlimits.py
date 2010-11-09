""" Test functions for limits module.
"""

from numpy.testing import *

from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
import numpy as np

##################################################

class TestPythonFloat(TestCase):
    def test_singleton(self):
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype),id(ftype2))

class TestHalf(TestCase):
    def test_singleton(self):
        ftype = finfo(half)
        ftype2 = finfo(half)
        assert_equal(id(ftype),id(ftype2))

class TestSingle(TestCase):
    def test_singleton(self):
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype),id(ftype2))

class TestDouble(TestCase):
    def test_singleton(self):
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype),id(ftype2))

class TestLongdouble(TestCase):
    def test_singleton(self,level=2):
        ftype = finfo(longdouble)
        ftype2 = finfo(longdouble)
        assert_equal(id(ftype),id(ftype2))

class TestIinfo(TestCase):
    def test_basic(self):
        dts = zip(['i1', 'i2', 'i4', 'i8',
                   'u1', 'u2', 'u4', 'u8'],
                  [np.int8, np.int16, np.int32, np.int64,
                   np.uint8, np.uint16, np.uint32, np.uint64])
        for dt1, dt2 in dts:
            assert_equal(iinfo(dt1).min, iinfo(dt2).min)
            assert_equal(iinfo(dt1).max, iinfo(dt2).max)
        self.assertRaises(ValueError, iinfo, 'f4')

    def test_unsigned_max(self):
        types = np.sctypes['uint']
        for T in types:
            assert_equal(iinfo(T).max, T(-1))


def test_instances():
    iinfo(10)
    finfo(3.0)

if __name__ == "__main__":
    run_module_suite()
