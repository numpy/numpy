""" Test functions for limits module.

"""
from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.core import finfo, iinfo
from numpy import half, single, double, longdouble
from numpy.testing import (
    TestCase, run_module_suite, assert_equal
)

##################################################

class TestPythonFloat(TestCase):
    def test_singleton(self):
        ftype = finfo(float)
        ftype2 = finfo(float)
        assert_equal(id(ftype), id(ftype2))

class TestHalf(TestCase):
    def test_singleton(self):
        ftype = finfo(half)
        ftype2 = finfo(half)
        assert_equal(id(ftype), id(ftype2))

class TestSingle(TestCase):
    def test_singleton(self):
        ftype = finfo(single)
        ftype2 = finfo(single)
        assert_equal(id(ftype), id(ftype2))

class TestDouble(TestCase):
    def test_singleton(self):
        ftype = finfo(double)
        ftype2 = finfo(double)
        assert_equal(id(ftype), id(ftype2))

class TestLongdouble(TestCase):
    def test_singleton(self,level=2):
        ftype = finfo(longdouble)
        ftype2 = finfo(longdouble)
        assert_equal(id(ftype), id(ftype2))

class TestFinfo(TestCase):
    def test_basic(self):
        dts = list(zip(['f2', 'f4', 'f8', 'c8', 'c16'],
                       [np.float16, np.float32, np.float64, np.complex64,
                        np.complex128]))
        for dt1, dt2 in dts:
            for attr in ('bits', 'eps', 'epsneg', 'iexp', 'machar', 'machep',
                         'max', 'maxexp', 'min', 'minexp', 'negep', 'nexp',
                         'nmant', 'precision', 'resolution', 'tiny'):
                assert_equal(getattr(finfo(dt1), attr),
                             getattr(finfo(dt2), attr), attr)
        self.assertRaises(ValueError, finfo, 'i4')

class TestIinfo(TestCase):
    def test_basic(self):
        dts = list(zip(['i1', 'i2', 'i4', 'i8',
                   'u1', 'u2', 'u4', 'u8'],
                  [np.int8, np.int16, np.int32, np.int64,
                   np.uint8, np.uint16, np.uint32, np.uint64]))
        for dt1, dt2 in dts:
            for attr in ('bits', 'min', 'max'):
                assert_equal(getattr(iinfo(dt1), attr),
                             getattr(iinfo(dt2), attr), attr)
        self.assertRaises(ValueError, iinfo, 'f4')

    def test_unsigned_max(self):
        types = np.sctypes['uint']
        for T in types:
            assert_equal(iinfo(T).max, T(-1))

class TestRepr(TestCase):
    def test_iinfo_repr(self):
        expected = "iinfo(min=-32768, max=32767, dtype=int16)"
        assert_equal(repr(np.iinfo(np.int16)), expected)

    def test_finfo_repr(self):
        expected = "finfo(resolution=1e-06, min=-3.4028235e+38," + \
                   " max=3.4028235e+38, dtype=float32)"
        assert_equal(repr(np.finfo(np.float32)), expected)


def test_instances():
    iinfo(10)
    finfo(3.0)

if __name__ == "__main__":
    run_module_suite()
