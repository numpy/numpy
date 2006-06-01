
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

class test_types(ScipyTestCase):
    def check_types(self, level=1):
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

if __name__ == "__main__":
    NumpyTest().run()
