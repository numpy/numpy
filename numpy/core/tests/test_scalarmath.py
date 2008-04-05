from numpy.testing import *
import numpy.core.umath as ncu
from numpy import array
import numpy as np

types = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
         np.int_, np.uint, np.longlong, np.ulonglong,
         np.single, np.double, np.longdouble, np.csingle,
         np.cdouble, np.clongdouble]

# This compares scalarmath against ufuncs.

class TestTypes(NumpyTestCase):
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

class TestPower(NumpyTestCase):
    def check_small_types(self):
        for t in [np.int8, np.int16]:
            a = t(3)
            b = a ** 4
            assert b == 81, "error with %r: got %r" % (t,b)

    def check_large_types(self):
        for t in [np.int32, np.int64, np.float32, np.float64, np.longdouble]:
            a = t(51)
            b = a ** 4
            msg = "error with %r: got %r" % (t,b)
            if np.issubdtype(t, np.integer):
                assert b == 6765201, msg
            else:
                assert_almost_equal(b, 6765201, err_msg=msg)

class TestConversion(NumpyTestCase):
    def test_int_from_long(self):
        l = [1e6, 1e12, 1e18, -1e6, -1e12, -1e18]
        li = [10**6, 10**12, 10**18, -10**6, -10**12, -10**18]
        for T in [None, np.float64, np.int64]:
            a = np.array(l,dtype=T)
            assert_equal(map(int,a), li)

        a = np.array(l[:3], dtype=np.uint64)
        assert_equal(map(int,a), li[:3])

#class TestRepr(NumpyTestCase):
#    def check_repr(self):
#        for t in types:
#            val = t(1197346475.0137341)
#            val_repr = repr(val)
#            val2 = eval(val_repr)
#            assert_equal( val, val2 )

class TestRepr(NumpyTestCase):
    def check_float_repr(self):
        from numpy import nan, inf
        for t in [np.float32, np.float64, np.longdouble]:
            if t is np.longdouble: # skip it for now.  
                continue
            finfo=np.finfo(t)
            last_fraction_bit_idx = finfo.nexp + finfo.nmant
            last_exponent_bit_idx = finfo.nexp
            storage_bytes = np.dtype(t).itemsize*8
            for which in ['small denorm','small norm']: # could add some more types here
                # Values from http://en.wikipedia.org/wiki/IEEE_754
                constr = array([0x00]*storage_bytes,dtype=np.uint8)
                if which == 'small denorm':
                    byte = last_fraction_bit_idx // 8
                    bytebit = 7-(last_fraction_bit_idx % 8)
                    constr[byte] = 1<<bytebit
                elif which == 'small norm':
                    byte = last_exponent_bit_idx // 8
                    bytebit = 7-(last_exponent_bit_idx % 8)
                    constr[byte] = 1<<bytebit
                else:
                    raise ValueError('hmm')
                val = constr.view(t)[0]
                val_repr = repr(val)
                val2 = t(eval(val_repr))
                if t == np.longdouble:
                    # Skip longdouble - the eval() statement goes
                    # through a Python float, which will lose
                    # precision
                    continue
                if not (val2 == 0 and val < 1e-100):
                    assert_equal(val, val2)

if __name__ == "__main__":
    NumpyTest().run()
