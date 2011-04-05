import sys
from numpy.testing import *
import numpy as np

types = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
         np.int_, np.uint, np.longlong, np.ulonglong,
         np.single, np.double, np.longdouble, np.csingle,
         np.cdouble, np.clongdouble]

# This compares scalarmath against ufuncs.

class TestTypes(TestCase):
    def test_types(self, level=1):
        for atype in types:
            a = atype(1)
            assert a == 1, "error with %r: got %r" % (atype,a)

    def test_type_add(self, level=1):
        # list of types
        for k, atype in enumerate(types):
            a_scalar = atype(3)
            a_array = np.array([3],dtype=atype)
            for l, btype in enumerate(types):
                b_scalar = btype(1)
                b_array = np.array([1],dtype=btype)
                c_scalar = a_scalar + b_scalar
                c_array = a_array + b_array
                # It was comparing the type numbers, but the new ufunc
                # function-finding mechanism finds the lowest function
                # to which both inputs can be cast - which produces 'l'
                # when you do 'q' + 'b'.  The old function finding mechanism
                # skipped ahead based on the first argument, but that
                # does not produce properly symmetric results...
                assert_equal(c_scalar.dtype, c_array.dtype,
                           "error with types (%d/'%c' + %d/'%c')" %
                            (k,np.dtype(atype).char,l,np.dtype(btype).char))

    def test_type_create(self, level=1):
        for k, atype in enumerate(types):
            a = np.array([1,2,3],atype)
            b = atype([1,2,3])
            assert_equal(a,b)


class TestPower(TestCase):
    def test_small_types(self):
        for t in [np.int8, np.int16]:
            a = t(3)
            b = a ** 4
            assert b == 81, "error with %r: got %r" % (t,b)

    def test_large_types(self):
        for t in [np.int32, np.int64, np.float32, np.float64, np.longdouble]:
            a = t(51)
            b = a ** 4
            msg = "error with %r: got %r" % (t,b)
            if np.issubdtype(t, np.integer):
                assert b == 6765201, msg
            else:
                assert_almost_equal(b, 6765201, err_msg=msg)


class TestConversion(TestCase):
    def test_int_from_long(self):
        l = [1e6, 1e12, 1e18, -1e6, -1e12, -1e18]
        li = [10**6, 10**12, 10**18, -10**6, -10**12, -10**18]
        for T in [None, np.float64, np.int64]:
            a = np.array(l,dtype=T)
            assert_equal(map(int,a), li)

        a = np.array(l[:3], dtype=np.uint64)
        assert_equal(map(int,a), li[:3])


#class TestRepr(TestCase):
#    def test_repr(self):
#        for t in types:
#            val = t(1197346475.0137341)
#            val_repr = repr(val)
#            val2 = eval(val_repr)
#            assert_equal( val, val2 )


class TestRepr:
    def _test_type_repr(self, t):
        finfo=np.finfo(t)
        last_fraction_bit_idx = finfo.nexp + finfo.nmant
        last_exponent_bit_idx = finfo.nexp
        storage_bytes = np.dtype(t).itemsize*8
        # could add some more types to the list below
        for which in ['small denorm','small norm']:
            # Values from http://en.wikipedia.org/wiki/IEEE_754
            constr = np.array([0x00]*storage_bytes,dtype=np.uint8)
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
            if not (val2 == 0 and val < 1e-100):
                assert_equal(val, val2)

    def test_float_repr(self):
        # long double test cannot work, because eval goes through a python
        # float
        for t in [np.float32, np.float64]:
            yield self._test_type_repr, t

if __name__ == "__main__":
    run_module_suite()
