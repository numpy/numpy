from numpy.testing import *
set_package_path()
import numpy.core.umath as ncu
from numpy import array
import numpy as N
restore_path()

types = [N.bool_, N.byte, N.ubyte, N.short, N.ushort, N.intc, N.uintc,
         N.int_, N.uint, N.longlong, N.ulonglong,
         N.single, N.double, N.longdouble, N.csingle,
         N.cdouble, N.clongdouble]

# This compares scalarmath against ufuncs.

class test_types(NumpyTestCase):
    def check_types(self, level=1):
        for atype in types:
            a = atype(1)
            assert a == 1, "error with %r: got %r" % (atype,a)

    def check_type_add(self, level=1):
        # list of types
        for k, atype in enumerate(types):
            vala = atype(3)
            val1 = array([3],dtype=atype)
            for l, btype in enumerate(types):
                valb = btype(1)
                val2 = array([1],dtype=btype)
                val = vala+valb
                valo = val1 + val2
                assert val.dtype.num == valo.dtype.num and \
                       val.dtype.char == valo.dtype.char, \
                       "error with (%d,%d)" % (k,l)

    def check_type_create(self, level=1):
        for k, atype in enumerate(types):
            a = array([1,2,3],atype)
            b = atype([1,2,3])
            assert_equal(a,b)

class test_power(NumpyTestCase):
    def check_small_types(self):
        for t in [N.int8, N.int16]:
            a = t(3)
            b = a ** 4
            assert b == 81, "error with %r: got %r" % (t,b)

    def check_large_types(self):
        for t in [N.int32, N.int64, N.float32, N.float64, N.longdouble]:
            a = t(51)
            b = a ** 4
            assert b == 6765201, "error with %r: got %r" % (t,b)

if __name__ == "__main__":
    NumpyTest().run()
