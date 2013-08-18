from __future__ import division, absolute_import, print_function

import unittest

from numpy.testing import *

from numpy import array
from numpy.oldnumeric import *
from numpy.core.numeric import float32, float64, complex64, complex128, int8, \
        int16, int32, int64, uint, uint8, uint16, uint32, uint64

class test_oldtypes(unittest.TestCase):
    def test_oldtypes(self, level=1):
        a1 = array([0, 1, 0], Float)
        a2 = array([0, 1, 0], float)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Float8)
        a2 = array([0, 1, 0], float)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Float16)
        a2 = array([0, 1, 0], float)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Float32)
        a2 = array([0, 1, 0], float32)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Float64)
        a2 = array([0, 1, 0], float64)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Complex)
        a2 = array([0, 1, 0], complex)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Complex8)
        a2 = array([0, 1, 0], complex)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Complex16)
        a2 = array([0, 1, 0], complex)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Complex32)
        a2 = array([0, 1, 0], complex64)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Complex64)
        a2 = array([0, 1, 0], complex128)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Int)
        a2 = array([0, 1, 0], int)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Int8)
        a2 = array([0, 1, 0], int8)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Int16)
        a2 = array([0, 1, 0], int16)
        assert_array_equal(a1, a2)
        a1 = array([0, 1, 0], Int32)
        a2 = array([0, 1, 0], int32)
        assert_array_equal(a1, a2)
        try:
            a1 = array([0, 1, 0], Int64)
            a2 = array([0, 1, 0], int64)
            assert_array_equal(a1, a2)
        except NameError:
            # Not all systems have 64-bit integers.
            pass
        a1 = array([0, 1, 0], UnsignedInt)
        a2 = array([0, 1, 0], UnsignedInteger)
        a3 = array([0, 1, 0], uint)
        assert_array_equal(a1, a3)
        assert_array_equal(a2, a3)
        a1 = array([0, 1, 0], UInt8)
        a2 = array([0, 1, 0], UnsignedInt8)
        a3 = array([0, 1, 0], uint8)
        assert_array_equal(a1, a3)
        assert_array_equal(a2, a3)
        a1 = array([0, 1, 0], UInt16)
        a2 = array([0, 1, 0], UnsignedInt16)
        a3 = array([0, 1, 0], uint16)
        assert_array_equal(a1, a3)
        assert_array_equal(a2, a3)
        a1 = array([0, 1, 0], UInt32)
        a2 = array([0, 1, 0], UnsignedInt32)
        a3 = array([0, 1, 0], uint32)
        assert_array_equal(a1, a3)
        assert_array_equal(a2, a3)
        try:
            a1 = array([0, 1, 0], UInt64)
            a2 = array([0, 1, 0], UnsignedInt64)
            a3 = array([0, 1, 0], uint64)
            assert_array_equal(a1, a3)
            assert_array_equal(a2, a3)
        except NameError:
            # Not all systems have 64-bit integers.
            pass


if __name__ == "__main__":
    import nose
    nose.main()
