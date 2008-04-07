# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for MaskedArray & subclassing.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
"""
__author__ = "Pierre GF Gerard-Marchant"

import types
import warnings

import numpy
import numpy.core.fromnumeric  as fromnumeric
from numpy.testing import NumpyTest, NumpyTestCase
from numpy.testing import set_local_path, restore_path
from numpy.testing.utils import build_err_msg
from numpy import array as narray

import numpy.ma.testutils
from numpy.ma.testutils import *

import numpy.ma.core as coremodule
from numpy.ma.core import *

pi = numpy.pi

set_local_path()
from test_old_ma import *
restore_path()

#..............................................................................
class TestMA(NumpyTestCase):
    "Base test class for MaskedArrays."
    def __init__(self, *args, **kwds):
        NumpyTestCase.__init__(self, *args, **kwds)
        self.setUp()

    def setUp (self):
        "Base data definition."
        x = narray([1.,1.,1.,-2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = narray([5.,0.,3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0 ,0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        z = narray([-.5, 0., .5, .8])
        zm = masked_array(z, mask=[0,1,0,0])
        xf = numpy.where(m1, 1.e+20, x)
        xm.set_fill_value(1.e+20)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)
    #........................
    def test_basic1d(self):
        "Test of basic array creation and properties in 1 dimension."
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert(not isMaskedArray(x))
        assert(isMaskedArray(xm))
        assert((xm-ym).filled(0).any())
        fail_if_equal(xm.mask.astype(int_), ym.mask.astype(int_))
        s = x.shape
        assert_equal(numpy.shape(xm), s)
        assert_equal(xm.shape, s)
        assert_equal(xm.dtype, x.dtype)
        assert_equal(zm.dtype, z.dtype)
        assert_equal(xm.size , reduce(lambda x,y:x*y, s))
        assert_equal(count(xm) , len(m1) - reduce(lambda x,y:x+y, m1))
        assert_array_equal(xm, xf)
        assert_array_equal(filled(xm, 1.e20), xf)
        assert_array_equal(x, xm)
    #........................
    def test_basic2d(self):
        "Test of basic array creation and properties in 2 dimensions."
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        for s in [(4,3), (6,2)]:
            x.shape = s
            y.shape = s
            xm.shape = s
            ym.shape = s
            xf.shape = s

            assert(not isMaskedArray(x))
            assert(isMaskedArray(xm))
            assert_equal(shape(xm), s)
            assert_equal(xm.shape, s)
            assert_equal( xm.size , reduce(lambda x,y:x*y, s))
            assert_equal( count(xm) , len(m1) - reduce(lambda x,y:x+y, m1))
            assert_equal(xm, xf)
            assert_equal(filled(xm, 1.e20), xf)
            assert_equal(x, xm)
    #........................
    def test_basic_arithmetic (self):
        "Test of basic arithmetic."
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        a2d = array([[1,2],[0,4]])
        a2dm = masked_array(a2d, [[0,0],[1,0]])
        assert_equal(a2d * a2d, a2d * a2dm)
        assert_equal(a2d + a2d, a2d + a2dm)
        assert_equal(a2d - a2d, a2d - a2dm)
        for s in [(12,), (4,3), (2,6)]:
            x = x.reshape(s)
            y = y.reshape(s)
            xm = xm.reshape(s)
            ym = ym.reshape(s)
            xf = xf.reshape(s)
            assert_equal(-x, -xm)
            assert_equal(x + y, xm + ym)
            assert_equal(x - y, xm - ym)
            assert_equal(x * y, xm * ym)
            assert_equal(x / y, xm / ym)
            assert_equal(a10 + y, a10 + ym)
            assert_equal(a10 - y, a10 - ym)
            assert_equal(a10 * y, a10 * ym)
            assert_equal(a10 / y, a10 / ym)
            assert_equal(x + a10, xm + a10)
            assert_equal(x - a10, xm - a10)
            assert_equal(x * a10, xm * a10)
            assert_equal(x / a10, xm / a10)
            assert_equal(x**2, xm**2)
            assert_equal(abs(x)**2.5, abs(xm) **2.5)
            assert_equal(x**y, xm**ym)
            assert_equal(numpy.add(x,y), add(xm, ym))
            assert_equal(numpy.subtract(x,y), subtract(xm, ym))
            assert_equal(numpy.multiply(x,y), multiply(xm, ym))
            assert_equal(numpy.divide(x,y), divide(xm, ym))
    #........................
    def test_mixed_arithmetic(self):
        "Tests mixed arithmetics."
        na = narray([1])
        ma = array([1])
        self.failUnless(isinstance(na + ma, MaskedArray))
        self.failUnless(isinstance(ma + na, MaskedArray))
    #........................
    def test_inplace_arithmetic(self):
        """Test of inplace operations and rich comparisons"""
        # addition
        x = arange(10)
        y = arange(10)
        xm = arange(10)
        xm[2] = masked
        x += 1
        assert_equal(x, y+1)
        xm += 1
        assert_equal(xm, y+1)
        # subtraction
        x = arange(10)
        xm = arange(10)
        xm[2] = masked
        x -= 1
        assert_equal(x, y-1)
        xm -= 1
        assert_equal(xm, y-1)
        # multiplication
        x = arange(10)*1.0
        xm = arange(10)*1.0
        xm[2] = masked
        x *= 2.0
        assert_equal(x, y*2)
        xm *= 2.0
        assert_equal(xm, y*2)
        # division
        x = arange(10)*2
        xm = arange(10)*2
        xm[2] = masked
        x /= 2
        assert_equal(x, y)
        xm /= 2
        assert_equal(xm, y)
        # division, pt 2
        x = arange(10)*1.0
        xm = arange(10)*1.0
        xm[2] = masked
        x /= 2.0
        assert_equal(x, y/2.0)
        xm /= arange(10)
        assert_equal(xm, ones((10,)))

        warnings.simplefilter('ignore', DeprecationWarning)
        x = arange(10).astype(float_)
        xm = arange(10)
        xm[2] = masked
        id1 = x.raw_data().ctypes.data
        x += 1.
        assert (id1 == x.raw_data().ctypes.data)
        assert_equal(x, y+1.)
        warnings.simplefilter('default', DeprecationWarning)

        # addition w/ array
        x = arange(10, dtype=float_)
        xm = arange(10, dtype=float_)
        xm[2] = masked
        m = xm.mask
        a = arange(10, dtype=float_)
        a[-1] = masked
        x += a
        xm += a
        assert_equal(x,y+a)
        assert_equal(xm,y+a)
        assert_equal(xm.mask, mask_or(m,a.mask))
        # subtraction w/ array
        x = arange(10, dtype=float_)
        xm = arange(10, dtype=float_)
        xm[2] = masked
        m = xm.mask
        a = arange(10, dtype=float_)
        a[-1] = masked
        x -= a
        xm -= a
        assert_equal(x,y-a)
        assert_equal(xm,y-a)
        assert_equal(xm.mask, mask_or(m,a.mask))
        # multiplication w/ array
        x = arange(10, dtype=float_)
        xm = arange(10, dtype=float_)
        xm[2] = masked
        m = xm.mask
        a = arange(10, dtype=float_)
        a[-1] = masked
        x *= a
        xm *= a
        assert_equal(x,y*a)
        assert_equal(xm,y*a)
        assert_equal(xm.mask, mask_or(m,a.mask))
        # division w/ array
        x = arange(10, dtype=float_)
        xm = arange(10, dtype=float_)
        xm[2] = masked
        m = xm.mask
        a = arange(10, dtype=float_)
        a[-1] = masked
        x /= a
        xm /= a
        assert_equal(x,y/a)
        assert_equal(xm,y/a)
        assert_equal(xm.mask, mask_or(mask_or(m,a.mask), (a==0)))
        #
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        z = xm/ym
        assert_equal(z._mask, [1,1,1,0,0,1,1,0,0,0,1,1])
        assert_equal(z._data, [0.2,1.,1./3.,-1.,-pi/2.,-1.,5.,1.,1.,1.,2.,1.])
        xm = xm.copy()
        xm /= ym
        assert_equal(xm._mask, [1,1,1,0,0,1,1,0,0,0,1,1])
        assert_equal(xm._data, [1/5.,1.,1./3.,-1.,-pi/2.,-1.,5.,1.,1.,1.,2.,1.])


    #..........................
    def test_scalararithmetic(self):
        "Tests some scalar arithmetics on MaskedArrays."
        xm = array(0, mask=1)
        assert((1/array(0)).mask)
        assert((1 + xm).mask)
        assert((-xm).mask)
        assert((-xm).mask)
        assert(maximum(xm, xm).mask)
        assert(minimum(xm, xm).mask)
        assert(xm.filled().dtype is xm.data.dtype)
        x = array(0, mask=0)
        assert_equal(x.filled().ctypes.data, x.ctypes.data)
        assert_equal(str(xm), str(masked_print_option))
    #.........................
    def test_basic_ufuncs (self):
        "Test various functions such as sin, cos."
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_equal(numpy.cos(x), cos(xm))
        assert_equal(numpy.cosh(x), cosh(xm))
        assert_equal(numpy.sin(x), sin(xm))
        assert_equal(numpy.sinh(x), sinh(xm))
        assert_equal(numpy.tan(x), tan(xm))
        assert_equal(numpy.tanh(x), tanh(xm))
        assert_equal(numpy.sqrt(abs(x)), sqrt(xm))
        assert_equal(numpy.log(abs(x)), log(xm))
        assert_equal(numpy.log10(abs(x)), log10(xm))
        assert_equal(numpy.exp(x), exp(xm))
        assert_equal(numpy.arcsin(z), arcsin(zm))
        assert_equal(numpy.arccos(z), arccos(zm))
        assert_equal(numpy.arctan(z), arctan(zm))
        assert_equal(numpy.arctan2(x, y), arctan2(xm, ym))
        assert_equal(numpy.absolute(x), absolute(xm))
        assert_equal(numpy.equal(x,y), equal(xm, ym))
        assert_equal(numpy.not_equal(x,y), not_equal(xm, ym))
        assert_equal(numpy.less(x,y), less(xm, ym))
        assert_equal(numpy.greater(x,y), greater(xm, ym))
        assert_equal(numpy.less_equal(x,y), less_equal(xm, ym))
        assert_equal(numpy.greater_equal(x,y), greater_equal(xm, ym))
        assert_equal(numpy.conjugate(x), conjugate(xm))
    #........................
    def test_count_func (self):
        "Tests count"
        ott = array([0.,1.,2.,3.], mask=[1,0,0,0])
        assert( isinstance(count(ott), int))
        assert_equal(3, count(ott))
        assert_equal(1, count(1))
        assert_equal(0, array(1,mask=[1]))
        ott = ott.reshape((2,2))
        assert isinstance(count(ott,0), ndarray)
        assert isinstance(count(ott), types.IntType)
        assert_equal(3, count(ott))
        assert getmask(count(ott,0)) is nomask
        assert_equal([1,2],count(ott,0))
    #........................
    def test_minmax_func (self):
        "Tests minimum and maximum."
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        xr = numpy.ravel(x) #max doesn't work if shaped
        xmr = ravel(xm)
        assert_equal(max(xr), maximum(xmr)) #true because of careful selection of data
        assert_equal(min(xr), minimum(xmr)) #true because of careful selection of data
        #
        assert_equal(minimum([1,2,3],[4,0,9]), [1,0,3])
        assert_equal(maximum([1,2,3],[4,0,9]), [4,2,9])
        x = arange(5)
        y = arange(5) - 2
        x[3] = masked
        y[0] = masked
        assert_equal(minimum(x,y), where(less(x,y), x, y))
        assert_equal(maximum(x,y), where(greater(x,y), x, y))
        assert minimum(x) == 0
        assert maximum(x) == 4
        #
        x = arange(4).reshape(2,2)
        x[-1,-1] = masked
        assert_equal(maximum(x), 2)

    def test_minmax_methods(self):
        "Additional tests on max/min"
        (_, _, _, _, _, xm, _, _, _, _) = self.d
        xm.shape = (xm.size,)
        assert_equal(xm.max(), 10)
        assert(xm[0].max() is masked)
        assert(xm[0].max(0) is masked)
        assert(xm[0].max(-1) is masked)
        assert_equal(xm.min(), -10.)
        assert(xm[0].min() is masked)
        assert(xm[0].min(0) is masked)
        assert(xm[0].min(-1) is masked)
        assert_equal(xm.ptp(), 20.)
        assert(xm[0].ptp() is masked)
        assert(xm[0].ptp(0) is masked)
        assert(xm[0].ptp(-1) is masked)
        #
        x = array([1,2,3], mask=True)
        assert(x.min() is masked)
        assert(x.max() is masked)
        assert(x.ptp() is masked)
    #........................
    def test_addsumprod (self):
        "Tests add, sum, product."
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_equal(numpy.add.reduce(x), add.reduce(x))
        assert_equal(numpy.add.accumulate(x), add.accumulate(x))
        assert_equal(4, sum(array(4),axis=0))
        assert_equal(4, sum(array(4), axis=0))
        assert_equal(numpy.sum(x,axis=0), sum(x,axis=0))
        assert_equal(numpy.sum(filled(xm,0),axis=0), sum(xm,axis=0))
        assert_equal(numpy.sum(x,0), sum(x,0))
        assert_equal(numpy.product(x,axis=0), product(x,axis=0))
        assert_equal(numpy.product(x,0), product(x,0))
        assert_equal(numpy.product(filled(xm,1),axis=0), product(xm,axis=0))
        s = (3,4)
        x.shape = y.shape = xm.shape = ym.shape = s
        if len(s) > 1:
            assert_equal(numpy.concatenate((x,y),1), concatenate((xm,ym),1))
            assert_equal(numpy.add.reduce(x,1), add.reduce(x,1))
            assert_equal(numpy.sum(x,1), sum(x,1))
            assert_equal(numpy.product(x,1), product(x,1))
    #.........................
    def test_concat(self):
        "Tests concatenations."
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # basic concatenation
        assert_equal(numpy.concatenate((x,y)), concatenate((xm,ym)))
        assert_equal(numpy.concatenate((x,y)), concatenate((x,y)))
        assert_equal(numpy.concatenate((x,y)), concatenate((xm,y)))
        assert_equal(numpy.concatenate((x,y,x)), concatenate((x,ym,x)))
        # Concatenation along an axis
        s = (3,4)
        x.shape = y.shape = xm.shape = ym.shape = s
        assert_equal(xm.mask, numpy.reshape(m1, s))
        assert_equal(ym.mask, numpy.reshape(m2, s))
        xmym = concatenate((xm,ym),1)
        assert_equal(numpy.concatenate((x,y),1), xmym)
        assert_equal(numpy.concatenate((xm.mask,ym.mask),1), xmym._mask)
        #
        x=zeros(2)
        y=array(ones(2),mask=[False,True])
        z = concatenate((x,y))
        assert_array_equal(z,[0,0,1,1])
        assert_array_equal(z.mask,[False,False,False,True])
        z = concatenate((y,x))
        assert_array_equal(z,[1,1,0,0])
        assert_array_equal(z.mask,[False,True,False,False])

    #........................
    def test_indexing(self):
        "Tests conversions and indexing"
        x1 = numpy.array([1,2,4,3])
        x2 = array(x1, mask=[1,0,0,0])
        x3 = array(x1, mask=[0,1,0,1])
        x4 = array(x1)
    # test conversion to strings
        junk, garbage = str(x2), repr(x2)
        assert_equal(numpy.sort(x1),sort(x2,endwith=False))
    # tests of indexing
        assert type(x2[1]) is type(x1[1])
        assert x1[1] == x2[1]
        assert x2[0] is masked
        assert_equal(x1[2],x2[2])
        assert_equal(x1[2:5],x2[2:5])
        assert_equal(x1[:],x2[:])
        assert_equal(x1[1:], x3[1:])
        x1[2] = 9
        x2[2] = 9
        assert_equal(x1,x2)
        x1[1:3] = 99
        x2[1:3] = 99
        assert_equal(x1,x2)
        x2[1] = masked
        assert_equal(x1,x2)
        x2[1:3] = masked
        assert_equal(x1,x2)
        x2[:] = x1
        x2[1] = masked
        assert allequal(getmask(x2),array([0,1,0,0]))
        x3[:] = masked_array([1,2,3,4],[0,1,1,0])
        assert allequal(getmask(x3), array([0,1,1,0]))
        x4[:] = masked_array([1,2,3,4],[0,1,1,0])
        assert allequal(getmask(x4), array([0,1,1,0]))
        assert allequal(x4, array([1,2,3,4]))
        x1 = numpy.arange(5)*1.0
        x2 = masked_values(x1, 3.0)
        assert_equal(x1,x2)
        assert allequal(array([0,0,0,1,0],MaskType), x2.mask)
#FIXME: Well, eh, fill_value is now a property        assert_equal(3.0, x2.fill_value())
        assert_equal(3.0, x2.fill_value)
        x1 = array([1,'hello',2,3],object)
        x2 = numpy.array([1,'hello',2,3],object)
        s1 = x1[1]
        s2 = x2[1]
        assert_equal(type(s2), str)
        assert_equal(type(s1), str)
        assert_equal(s1, s2)
        assert x1[1:1].shape == (0,)
    #........................
    def test_copy(self):
        "Tests of some subtle points of copying and sizing."
        n = [0,0,1,0,0]
        m = make_mask(n)
        m2 = make_mask(m)
        assert(m is m2)
        m3 = make_mask(m, copy=1)
        assert(m is not m3)

        warnings.simplefilter('ignore', DeprecationWarning)
        x1 = numpy.arange(5)
        y1 = array(x1, mask=m)
        #assert( y1._data is x1)
        assert_equal(y1._data.__array_interface__, x1.__array_interface__)
        assert( allequal(x1,y1.raw_data()))
        #assert( y1.mask is m)
        assert_equal(y1._mask.__array_interface__, m.__array_interface__)
        warnings.simplefilter('default', DeprecationWarning)

        y1a = array(y1)
        #assert( y1a.raw_data() is y1.raw_data())
        assert( y1a._data.__array_interface__ == y1._data.__array_interface__)
        assert( y1a.mask is y1.mask)

        y2 = array(x1, mask=m)
        #assert( y2.raw_data() is x1)
        assert (y2._data.__array_interface__ == x1.__array_interface__)
        #assert( y2.mask is m)
        assert (y2._mask.__array_interface__ == m.__array_interface__)
        assert( y2[2] is masked)
        y2[2] = 9
        assert( y2[2] is not masked)
        #assert( y2.mask is not m)
        assert (y2._mask.__array_interface__ != m.__array_interface__)
        assert( allequal(y2.mask, 0))

        y3 = array(x1*1.0, mask=m)
        assert(filled(y3).dtype is (x1*1.0).dtype)

        x4 = arange(4)
        x4[2] = masked
        y4 = resize(x4, (8,))
        assert_equal(concatenate([x4,x4]), y4)
        assert_equal(getmask(y4),[0,0,1,0,0,0,1,0])
        y5 = repeat(x4, (2,2,2,2), axis=0)
        assert_equal(y5, [0,0,1,1,2,2,3,3])
        y6 = repeat(x4, 2, axis=0)
        assert_equal(y5, y6)
        y7 = x4.repeat((2,2,2,2), axis=0)
        assert_equal(y5,y7)
        y8 = x4.repeat(2,0)
        assert_equal(y5,y8)

        y9 = x4.copy()
        assert_equal(y9._data, x4._data)
        assert_equal(y9._mask, x4._mask)
        #
        x = masked_array([1,2,3], mask=[0,1,0])
        # Copy is False by default
        y = masked_array(x)
        assert_equal(y._data.ctypes.data, x._data.ctypes.data)
        assert_equal(y._mask.ctypes.data, x._mask.ctypes.data)
        y = masked_array(x, copy=True)
        assert_not_equal(y._data.ctypes.data, x._data.ctypes.data)
        assert_not_equal(y._mask.ctypes.data, x._mask.ctypes.data)
    #........................
    def test_where(self):
        "Test the where function"
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        d = where(xm>2,xm,-9)
        assert_equal(d, [-9.,-9.,-9.,-9., -9., 4., -9., -9., 10., -9., -9., 3.])
        assert_equal(d._mask, xm._mask)
        d = where(xm>2,-9,ym)
        assert_equal(d, [5.,0.,3., 2., -1.,-9.,-9., -10., -9., 1., 0., -9.])
        assert_equal(d._mask, [1,0,1,0,0,0,1,0,0,0,0,0])
        d = where(xm>2, xm, masked)
        assert_equal(d, [-9.,-9.,-9.,-9., -9., 4., -9., -9., 10., -9., -9., 3.])
        tmp = xm._mask.copy()
        tmp[(xm<=2).filled(True)] = True
        assert_equal(d._mask, tmp)
        #
        ixm = xm.astype(int_)
        d = where(ixm>2, ixm, masked)
        assert_equal(d, [-9,-9,-9,-9, -9, 4, -9, -9, 10, -9, -9, 3])
        assert_equal(d.dtype, ixm.dtype)
        #
        x = arange(10)
        x[3] = masked
        c = x >= 8
        z = where(c , x, masked)
        assert z.dtype is x.dtype
        assert z[3] is masked
        assert z[4] is masked
        assert z[7] is masked
        assert z[8] is not masked
        assert z[9] is not masked
        assert_equal(x,z)
        #
        z = where(c , masked, x)
        assert z.dtype is x.dtype
        assert z[3] is masked
        assert z[4] is not masked
        assert z[7] is not masked
        assert z[8] is masked
        assert z[9] is masked

    #........................
    def test_oddfeatures_1(self):
        "Test of other odd features"
        x = arange(20)
        x = x.reshape(4,5)
        x.flat[5] = 12
        assert x[1,0] == 12
        z = x + 10j * x
        assert_equal(z.real, x)
        assert_equal(z.imag, 10*x)
        assert_equal((z*conjugate(z)).real, 101*x*x)
        z.imag[...] = 0.0

        x = arange(10)
        x[3] = masked
        assert str(x[3]) == str(masked)
        c = x >= 8
        assert count(where(c,masked,masked)) == 0
        assert shape(where(c,masked,masked)) == c.shape
        #
        z = masked_where(c, x)
        assert z.dtype is x.dtype
        assert z[3] is masked
        assert z[4] is not masked
        assert z[7] is not masked
        assert z[8] is masked
        assert z[9] is masked
        assert_equal(x,z)
        #
    #........................
    def test_oddfeatures_2(self):
        "Tests some more features."
        x = array([1.,2.,3.,4.,5.])
        c = array([1,1,1,0,0])
        x[2] = masked
        z = where(c, x, -x)
        assert_equal(z, [1.,2.,0., -4., -5])
        c[0] = masked
        z = where(c, x, -x)
        assert_equal(z, [1.,2.,0., -4., -5])
        assert z[0] is masked
        assert z[1] is not masked
        assert z[2] is masked
        #
        x = arange(6)
        x[5] = masked
        y = arange(6)*10
        y[2] = masked
        c = array([1,1,1,0,0,0], mask=[1,0,0,0,0,0])
        cm = c.filled(1)
        z = where(c,x,y)
        zm = where(cm,x,y)
        assert_equal(z, zm)
        assert getmask(zm) is nomask
        assert_equal(zm, [0,1,2,30,40,50])
        z = where(c, masked, 1)
        assert_equal(z, [99,99,99,1,1,1])
        z = where(c, 1, masked)
        assert_equal(z, [99, 1, 1, 99, 99, 99])
    #........................
    def test_oddfeatures_3(self):
        """Tests some generic features."""
        atest = ones((10,10,10), dtype=float_)
        btest = zeros(atest.shape, MaskType)
        ctest = masked_where(btest,atest)
        assert_equal(atest,ctest)
    #........................
    def test_maskingfunctions(self):
        "Tests masking functions."
        x = array([1.,2.,3.,4.,5.])
        x[2] = masked
        assert_equal(masked_where(greater(x, 2), x), masked_greater(x,2))
        assert_equal(masked_where(greater_equal(x, 2), x), masked_greater_equal(x,2))
        assert_equal(masked_where(less(x, 2), x), masked_less(x,2))
        assert_equal(masked_where(less_equal(x, 2), x), masked_less_equal(x,2))
        assert_equal(masked_where(not_equal(x, 2), x), masked_not_equal(x,2))
        assert_equal(masked_where(equal(x, 2), x), masked_equal(x,2))
        assert_equal(masked_where(not_equal(x,2), x), masked_not_equal(x,2))
        assert_equal(masked_inside(range(5), 1, 3), [0, 199, 199, 199, 4])
        assert_equal(masked_outside(range(5), 1, 3),[199,1,2,3,199])
        assert_equal(masked_inside(array(range(5), mask=[1,0,0,0,0]), 1, 3).mask, [1,1,1,1,0])
        assert_equal(masked_outside(array(range(5), mask=[0,1,0,0,0]), 1, 3).mask, [1,1,0,0,1])
        assert_equal(masked_equal(array(range(5), mask=[1,0,0,0,0]), 2).mask, [1,0,1,0,0])
        assert_equal(masked_not_equal(array([2,2,1,2,1], mask=[1,0,0,0,0]), 2).mask, [1,0,1,0,1])
        assert_equal(masked_where([1,1,0,0,0], [1,2,3,4,5]), [99,99,3,4,5])
    #........................
    def test_TakeTransposeInnerOuter(self):
        "Test of take, transpose, inner, outer products"
        x = arange(24)
        y = numpy.arange(24)
        x[5:6] = masked
        x = x.reshape(2,3,4)
        y = y.reshape(2,3,4)
        assert_equal(numpy.transpose(y,(2,0,1)), transpose(x,(2,0,1)))
        assert_equal(numpy.take(y, (2,0,1), 1), take(x, (2,0,1), 1))
        assert_equal(numpy.inner(filled(x,0),filled(y,0)),
                            inner(x, y))
        assert_equal(numpy.outer(filled(x,0),filled(y,0)),
                            outer(x, y))
        y = array(['abc', 1, 'def', 2, 3], object)
        y[2] = masked
        t = take(y,[0,3,4])
        assert t[0] == 'abc'
        assert t[1] == 2
        assert t[2] == 3
    #.......................
    def test_maskedelement(self):
        "Test of masked element"
        x = arange(6)
        x[1] = masked
        assert(str(masked) ==  '--')
        assert(x[1] is masked)
        assert_equal(filled(x[1], 0), 0)
        # don't know why these should raise an exception...
        #self.failUnlessRaises(Exception, lambda x,y: x+y, masked, masked)
        #self.failUnlessRaises(Exception, lambda x,y: x+y, masked, 2)
        #self.failUnlessRaises(Exception, lambda x,y: x+y, masked, xx)
        #self.failUnlessRaises(Exception, lambda x,y: x+y, xx, masked)
    #........................
    def test_scalar(self):
        "Checks masking a scalar"
        x = masked_array(0)
        assert_equal(str(x), '0')
        x = masked_array(0,mask=True)
        assert_equal(str(x), str(masked_print_option))
        x = masked_array(0, mask=False)
        assert_equal(str(x), '0')
    #........................
    def test_usingmasked(self):
        "Checks that there's no collapsing to masked"
        x = masked_array([1,2])
        y = x * masked
        assert_equal(y.shape, x.shape)
        assert_equal(y._mask, [True, True])
        y = x[0] * masked
        assert y is masked
        y = x + masked
        assert_equal(y.shape, x.shape)
        assert_equal(y._mask, [True, True])

    #........................
    def test_topython(self):
        "Tests some communication issues with Python."
        assert_equal(1, int(array(1)))
        assert_equal(1.0, float(array(1)))
        assert_equal(1, int(array([[[1]]])))
        assert_equal(1.0, float(array([[1]])))
        self.failUnlessRaises(ValueError, float, array([1,1]))

        warnings.simplefilter('ignore',UserWarning)
        assert numpy.isnan(float(array([1],mask=[1])))
        warnings.simplefilter('default',UserWarning)
#TODO: Check how bool works...
#TODO:        self.failUnless(bool(array([0,1])))
#TODO:        self.failUnless(bool(array([0,0],mask=[0,1])))
#TODO:        self.failIf(bool(array([0,0])))
#TODO:        self.failIf(bool(array([0,0],mask=[0,0])))
    #........................
    def test_arraymethods(self):
        "Tests some MaskedArray methods."
        a = array([1,3,2])
        b = array([1,3,2], mask=[1,0,1])
        assert_equal(a.any(), a.data.any())
        assert_equal(a.all(), a.data.all())
        assert_equal(a.argmax(), a.data.argmax())
        assert_equal(a.argmin(), a.data.argmin())
        assert_equal(a.choose(0,1,2,3,4), a.data.choose(0,1,2,3,4))
        assert_equal(a.compress([1,0,1]), a.data.compress([1,0,1]))
        assert_equal(a.conj(), a.data.conj())
        assert_equal(a.conjugate(), a.data.conjugate())
        #
        m = array([[1,2],[3,4]])
        assert_equal(m.diagonal(), m.data.diagonal())
        assert_equal(a.sum(), a.data.sum())
        assert_equal(a.take([1,2]), a.data.take([1,2]))
        assert_equal(m.transpose(), m.data.transpose())
    #........................
    def test_basicattributes(self):
        "Tests some basic array attributes."
        a = array([1,3,2])
        b = array([1,3,2], mask=[1,0,1])
        assert_equal(a.ndim, 1)
        assert_equal(b.ndim, 1)
        assert_equal(a.size, 3)
        assert_equal(b.size, 3)
        assert_equal(a.shape, (3,))
        assert_equal(b.shape, (3,))
    #........................
    def test_single_element_subscript(self):
        "Tests single element subscripts of Maskedarrays."
        a = array([1,3,2])
        b = array([1,3,2], mask=[1,0,1])
        assert_equal(a[0].shape, ())
        assert_equal(b[0].shape, ())
        assert_equal(b[1].shape, ())
    #........................
    def test_maskcreation(self):
        "Tests how masks are initialized at the creation of Maskedarrays."
        data = arange(24, dtype=float_)
        data[[3,6,15]] = masked
        dma_1 = MaskedArray(data)
        assert_equal(dma_1.mask, data.mask)
        dma_2 = MaskedArray(dma_1)
        assert_equal(dma_2.mask, dma_1.mask)
        dma_3 = MaskedArray(dma_1, mask=[1,0,0,0]*6)
        fail_if_equal(dma_3.mask, dma_1.mask)

    def test_pickling(self):
        "Tests pickling"
        import cPickle
        a = arange(10)
        a[::3] = masked
        a.fill_value = 999
        a_pickled = cPickle.loads(a.dumps())
        assert_equal(a_pickled._mask, a._mask)
        assert_equal(a_pickled._data, a._data)
        assert_equal(a_pickled.fill_value, 999)
        #
        a = array(numpy.matrix(range(10)), mask=[1,0,1,0,0]*2)
        a_pickled = cPickle.loads(a.dumps())
        assert_equal(a_pickled._mask, a._mask)
        assert_equal(a_pickled, a)
        assert(isinstance(a_pickled._data,numpy.matrix))
    #
    def test_fillvalue(self):
        "Having fun with the fill_value"
        data = masked_array([1,2,3],fill_value=-999)
        series = data[[0,2,1]]
        assert_equal(series._fill_value, data._fill_value)
        #
        mtype = [('f',float_),('s','|S3')]
        x = array([(1,'a'),(2,'b'),(numpy.pi,'pi')], dtype=mtype)
        x.fill_value=999
        assert_equal(x.fill_value,[999.,'999'])
        assert_equal(x['f'].fill_value, 999)
        assert_equal(x['s'].fill_value, '999')
        #
        x.fill_value=(9,'???')
        assert_equal(x.fill_value, (9,'???'))
        assert_equal(x['f'].fill_value, 9)
        assert_equal(x['s'].fill_value, '???')
        #
        x = array([1,2,3.1])
        x.fill_value = 999
        assert_equal(numpy.asarray(x.fill_value).dtype, float_)
        assert_equal(x.fill_value, 999.)
    #
    def test_asarray(self):
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        xmm = asarray(xm)
        assert_equal(xmm._data, xm._data)
        assert_equal(xmm._mask, xm._mask)
    #
    def test_fix_invalid(self):
        "Checks fix_invalid."
        data = masked_array(numpy.sqrt([-1., 0., 1.]), mask=[0,0,1])
        data_fixed = fix_invalid(data)
        assert_equal(data_fixed._data, [data.fill_value, 0., 1.])
        assert_equal(data_fixed._mask, [1., 0., 1.])
    #
    def test_imag_real(self):
        "Check complex"
        xx = array([1+10j,20+2j], mask=[1,0])
        assert_equal(xx.imag,[10,2])
        assert_equal(xx.imag.filled(), [1e+20,2])
        assert_equal(xx.imag.dtype, xx._data.imag.dtype)
        assert_equal(xx.real,[1,20])
        assert_equal(xx.real.filled(), [1e+20,20])
        assert_equal(xx.real.dtype, xx._data.real.dtype)
    #
    def test_ndmin(self):
        "Check the use of ndmin"
        x = array([1,2,3],mask=[1,0,0], ndmin=2)
        assert_equal(x.shape,(1,3))
        assert_equal(x._data,[[1,2,3]])
        assert_equal(x._mask,[[1,0,0]])
    #
    def test_record(self):
        "Check record access"
        mtype = [('f',float_),('s','|S3')]
        x = array([(1,'a'),(2,'b'),(numpy.pi,'pi')], dtype=mtype)
        x[1] = masked
        #
        (xf, xs) = (x['f'], x['s'])
        assert_equal(xf.data, [1,2,numpy.pi])
        assert_equal(xf.mask, [0,1,0])
        assert_equal(xf.dtype, float_)
        assert_equal(xs.data, ['a', 'b', 'pi'])
        assert_equal(xs.mask, [0,1,0])
        assert_equal(xs.dtype, '|S3')
    #
    def test_set_records(self):
        "Check setting an element of a record)"
        mtype = [('f',float_),('s','|S3')]
        x = array([(1,'a'),(2,'b'),(numpy.pi,'pi')], dtype=mtype)
        x[0] = (10,'A')
        (xf, xs) = (x['f'], x['s'])
        assert_equal(xf.data, [10,2,numpy.pi])
        assert_equal(xf.dtype, float_)
        assert_equal(xs.data, ['A', 'b', 'pi'])
        assert_equal(xs.dtype, '|S3')
        


#...............................................................................

class TestUfuncs(NumpyTestCase):
    "Test class for the application of ufuncs on MaskedArrays."
    def setUp(self):
        "Base data definition."
        self.d = (array([1.0, 0, -1, pi/2]*2, mask=[0,1]+[0]*6),
                  array([1.0, 0, -1, pi/2]*2, mask=[1,0]+[0]*6),)

    def test_testUfuncRegression(self):
        "Tests new ufuncs on MaskedArrays."
        for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate',
                  'sin', 'cos', 'tan',
                  'arcsin', 'arccos', 'arctan',
                  'sinh', 'cosh', 'tanh',
                  'arcsinh',
                  'arccosh',
                  'arctanh',
                  'absolute', 'fabs', 'negative',
                  # 'nonzero', 'around',
                  'floor', 'ceil',
                  # 'sometrue', 'alltrue',
                  'logical_not',
                  'add', 'subtract', 'multiply',
                  'divide', 'true_divide', 'floor_divide',
                  'remainder', 'fmod', 'hypot', 'arctan2',
                  'equal', 'not_equal', 'less_equal', 'greater_equal',
                  'less', 'greater',
                  'logical_and', 'logical_or', 'logical_xor',
                  ]:
            #print f
            try:
                uf = getattr(umath, f)
            except AttributeError:
                uf = getattr(fromnumeric, f)
            mf = getattr(coremodule, f)
            args = self.d[:uf.nin]
            ur = uf(*args)
            mr = mf(*args)
            assert_equal(ur.filled(0), mr.filled(0), f)
            assert_mask_equal(ur.mask, mr.mask)
    #........................
    def test_reduce(self):
        "Tests reduce on MaskedArrays."
        a = self.d[0]
        assert(not alltrue(a,axis=0))
        assert(sometrue(a,axis=0))
        assert_equal(sum(a[:3],axis=0), 0)
        assert_equal(product(a,axis=0), 0)
        assert_equal(add.reduce(a), pi)
    #........................
    def test_minmax(self):
        "Tests extrema on MaskedArrays."
        a = arange(1,13).reshape(3,4)
        amask = masked_where(a < 5,a)
        assert_equal(amask.max(), a.max())
        assert_equal(amask.min(), 5)
        assert_equal(amask.max(0), a.max(0))
        assert_equal(amask.min(0), [5,6,7,8])
        assert(amask.max(1)[0].mask)
        assert(amask.min(1)[0].mask)

#...............................................................................

class TestArrayMethods(NumpyTestCase):
    "Test class for miscellaneous MaskedArrays methods."
    def setUp(self):
        "Base data definition."
        x = numpy.array([ 8.375,  7.545,  8.828,  8.5  ,  1.757,  5.928,
                      8.43 ,  7.78 ,  9.865,  5.878,  8.979,  4.732,
                      3.012,  6.022,  5.095,  3.116,  5.238,  3.957,
                      6.04 ,  9.63 ,  7.712,  3.382,  4.489,  6.479,
                      7.189,  9.645,  5.395,  4.961,  9.894,  2.893,
                      7.357,  9.828,  6.272,  3.758,  6.693,  0.993])
        X = x.reshape(6,6)
        XX = x.reshape(3,2,2,3)

        m = numpy.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        mx = array(data=x,mask=m)
        mX = array(data=X,mask=m.reshape(X.shape))
        mXX = array(data=XX,mask=m.reshape(XX.shape))

        m2 = numpy.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        m2x = array(data=x,mask=m2)
        m2X = array(data=X,mask=m2.reshape(X.shape))
        m2XX = array(data=XX,mask=m2.reshape(XX.shape))
        self.d =  (x,X,XX,m,mx,mX,mXX,m2x,m2X,m2XX)

    #------------------------------------------------------
    def test_trace(self):
        "Tests trace on MaskedArrays."
        (x,X,XX,m,mx,mX,mXX,m2x,m2X,m2XX) = self.d
        mXdiag = mX.diagonal()
        assert_equal(mX.trace(), mX.diagonal().compressed().sum())
        assert_almost_equal(mX.trace(),
                            X.trace() - sum(mXdiag.mask*X.diagonal(),axis=0))

    def test_clip(self):
        "Tests clip on MaskedArrays."
        (x,X,XX,m,mx,mX,mXX,m2x,m2X,m2XX) = self.d
        clipped = mx.clip(2,8)
        assert_equal(clipped.mask,mx.mask)
        assert_equal(clipped.data,x.clip(2,8))
        assert_equal(clipped.data,mx.data.clip(2,8))

    def test_ptp(self):
        "Tests ptp on MaskedArrays."
        (x,X,XX,m,mx,mX,mXX,m2x,m2X,m2XX) = self.d
        (n,m) = X.shape
        assert_equal(mx.ptp(),mx.compressed().ptp())
        rows = numpy.zeros(n,numpy.float_)
        cols = numpy.zeros(m,numpy.float_)
        for k in range(m):
            cols[k] = mX[:,k].compressed().ptp()
        for k in range(n):
            rows[k] = mX[k].compressed().ptp()
        assert_equal(mX.ptp(0),cols)
        assert_equal(mX.ptp(1),rows)

    def test_swapaxes(self):
        "Tests swapaxes on MaskedArrays."
        (x,X,XX,m,mx,mX,mXX,m2x,m2X,m2XX) = self.d
        mXswapped = mX.swapaxes(0,1)
        assert_equal(mXswapped[-1],mX[:,-1])
        mXXswapped = mXX.swapaxes(0,2)
        assert_equal(mXXswapped.shape,(2,2,3,3))

    def test_cumsumprod(self):
        "Tests cumsum & cumprod on MaskedArrays."
        (x,X,XX,m,mx,mX,mXX,m2x,m2X,m2XX) = self.d
        mXcp = mX.cumsum(0)
        assert_equal(mXcp.data,mX.filled(0).cumsum(0))
        mXcp = mX.cumsum(1)
        assert_equal(mXcp.data,mX.filled(0).cumsum(1))
        #
        mXcp = mX.cumprod(0)
        assert_equal(mXcp.data,mX.filled(1).cumprod(0))
        mXcp = mX.cumprod(1)
        assert_equal(mXcp.data,mX.filled(1).cumprod(1))

    def test_varstd(self):
        "Tests var & std on MaskedArrays."
        (x,X,XX,m,mx,mX,mXX,m2x,m2X,m2XX) = self.d
        assert_almost_equal(mX.var(axis=None),mX.compressed().var())
        assert_almost_equal(mX.std(axis=None),mX.compressed().std())
        assert_equal(mXX.var(axis=3).shape,XX.var(axis=3).shape)
        assert_equal(mX.var().shape,X.var().shape)
        (mXvar0,mXvar1) = (mX.var(axis=0), mX.var(axis=1))
        assert_almost_equal(mX.var(axis=None,ddof=2),mX.compressed().var(ddof=2))
        assert_almost_equal(mX.std(axis=None,ddof=2),mX.compressed().std(ddof=2))
        for k in range(6):
            assert_almost_equal(mXvar1[k],mX[k].compressed().var())
            assert_almost_equal(mXvar0[k],mX[:,k].compressed().var())
            assert_almost_equal(numpy.sqrt(mXvar0[k]), mX[:,k].compressed().std())

    def test_argmin(self):
        "Tests argmin & argmax on MaskedArrays."
        (x,X,XX,m,mx,mX,mXX,m2x,m2X,m2XX) = self.d
        #
        assert_equal(mx.argmin(),35)
        assert_equal(mX.argmin(),35)
        assert_equal(m2x.argmin(),4)
        assert_equal(m2X.argmin(),4)
        assert_equal(mx.argmax(),28)
        assert_equal(mX.argmax(),28)
        assert_equal(m2x.argmax(),31)
        assert_equal(m2X.argmax(),31)
        #
        assert_equal(mX.argmin(0), [2,2,2,5,0,5])
        assert_equal(m2X.argmin(0), [2,2,4,5,0,4])
        assert_equal(mX.argmax(0), [0,5,0,5,4,0])
        assert_equal(m2X.argmax(0), [5,5,0,5,1,0])
        #
        assert_equal(mX.argmin(1), [4,1,0,0,5,5,])
        assert_equal(m2X.argmin(1), [4,4,0,0,5,3])
        assert_equal(mX.argmax(1), [2,4,1,1,4,1])
        assert_equal(m2X.argmax(1), [2,4,1,1,1,1])

    def test_put(self):
        "Tests put."
        d = arange(5)
        n = [0,0,0,1,1]
        m = make_mask(n)
        x = array(d, mask = m)
        assert( x[3] is masked)
        assert( x[4] is masked)
        x[[1,4]] = [10,40]
#        assert( x.mask is not m)
        assert( x[3] is masked)
        assert( x[4] is not masked)
        assert_equal(x, [0,10,2,-1,40])
        #
        x = masked_array(arange(10), mask=[1,0,0,0,0]*2)
        i = [0,2,4,6]
        x.put(i, [6,4,2,0])
        assert_equal(x, asarray([6,1,4,3,2,5,0,7,8,9,]))
        assert_equal(x.mask, [0,0,0,0,0,1,0,0,0,0])
        x.put(i, masked_array([0,2,4,6],[1,0,1,0]))
        assert_array_equal(x, [0,1,2,3,4,5,6,7,8,9,])
        assert_equal(x.mask, [1,0,0,0,1,1,0,0,0,0])
        #
        x = masked_array(arange(10), mask=[1,0,0,0,0]*2)
        put(x, i, [6,4,2,0])
        assert_equal(x, asarray([6,1,4,3,2,5,0,7,8,9,]))
        assert_equal(x.mask, [0,0,0,0,0,1,0,0,0,0])
        put(x, i, masked_array([0,2,4,6],[1,0,1,0]))
        assert_array_equal(x, [0,1,2,3,4,5,6,7,8,9,])
        assert_equal(x.mask, [1,0,0,0,1,1,0,0,0,0])

    def test_put_hardmask(self):
        "Tests put on hardmask"
        d = arange(5)
        n = [0,0,0,1,1]
        m = make_mask(n)
        xh = array(d+1, mask = m, hard_mask=True, copy=True)
        xh.put([4,2,0,1,3],[1,2,3,4,5])
        assert_equal(xh._data, [3,4,2,4,5])

    def test_take(self):
        "Tests take"
        x = masked_array([10,20,30,40],[0,1,0,1])
        assert_equal(x.take([0,0,3]), masked_array([10, 10, 40], [0,0,1]) )
        assert_equal(x.take([0,0,3]), x[[0,0,3]])
        assert_equal(x.take([[0,1],[0,1]]),
                     masked_array([[10,20],[10,20]], [[0,1],[0,1]]) )
        #
        x = array([[10,20,30],[40,50,60]], mask=[[0,0,1],[1,0,0,]])
        assert_equal(x.take([0,2], axis=1),
                     array([[10,30],[40,60]], mask=[[0,1],[1,0]]))
        assert_equal(take(x, [0,2], axis=1),
                      array([[10,30],[40,60]], mask=[[0,1],[1,0]]))
        #........................
    def test_anyall(self):
        """Checks the any/all methods/functions."""
        x = numpy.array([[ 0.13,  0.26,  0.90],
                     [ 0.28,  0.33,  0.63],
                     [ 0.31,  0.87,  0.70]])
        m = numpy.array([[ True, False, False],
                     [False, False, False],
                     [True,  True, False]], dtype=numpy.bool_)
        mx = masked_array(x, mask=m)
        xbig = numpy.array([[False, False,  True],
                        [False, False,  True],
                        [False,  True,  True]], dtype=numpy.bool_)
        mxbig = (mx > 0.5)
        mxsmall = (mx < 0.5)
        #
        assert (mxbig.all()==False)
        assert (mxbig.any()==True)
        assert_equal(mxbig.all(0),[False, False, True])
        assert_equal(mxbig.all(1), [False, False, True])
        assert_equal(mxbig.any(0),[False, False, True])
        assert_equal(mxbig.any(1), [True, True, True])
        #
        assert (mxsmall.all()==False)
        assert (mxsmall.any()==True)
        assert_equal(mxsmall.all(0), [True,   True, False])
        assert_equal(mxsmall.all(1), [False, False, False])
        assert_equal(mxsmall.any(0), [True,   True, False])
        assert_equal(mxsmall.any(1), [True,   True, False])
        #
        X = numpy.matrix(x)
        mX = masked_array(X, mask=m)
        mXbig = (mX > 0.5)
        mXsmall = (mX < 0.5)
        #
        assert (mXbig.all()==False)
        assert (mXbig.any()==True)
        assert_equal(mXbig.all(0), numpy.matrix([False, False, True]))
        assert_equal(mXbig.all(1), numpy.matrix([False, False, True]).T)
        assert_equal(mXbig.any(0), numpy.matrix([False, False, True]))
        assert_equal(mXbig.any(1), numpy.matrix([ True,  True, True]).T)
        #
        assert (mXsmall.all()==False)
        assert (mXsmall.any()==True)
        assert_equal(mXsmall.all(0), numpy.matrix([True,   True, False]))
        assert_equal(mXsmall.all(1), numpy.matrix([False, False, False]).T)
        assert_equal(mXsmall.any(0), numpy.matrix([True,   True, False]))
        assert_equal(mXsmall.any(1), numpy.matrix([True,   True, False]).T)

    def test_keepmask(self):
        "Tests the keep mask flag"
        x = masked_array([1,2,3], mask=[1,0,0])
        mx = masked_array(x)
        assert_equal(mx.mask, x.mask)
        mx = masked_array(x, mask=[0,1,0], keep_mask=False)
        assert_equal(mx.mask, [0,1,0])
        mx = masked_array(x, mask=[0,1,0], keep_mask=True)
        assert_equal(mx.mask, [1,1,0])
        # We default to true
        mx = masked_array(x, mask=[0,1,0])
        assert_equal(mx.mask, [1,1,0])

    def test_hardmask(self):
        "Test hard_mask"
        d = arange(5)
        n = [0,0,0,1,1]
        m = make_mask(n)
        xh = array(d, mask = m, hard_mask=True)
        # We need to copy, to avoid updating d in xh!
        xs = array(d, mask = m, hard_mask=False, copy=True)
        xh[[1,4]] = [10,40]
        xs[[1,4]] = [10,40]
        assert_equal(xh._data, [0,10,2,3,4])
        assert_equal(xs._data, [0,10,2,3,40])
        #assert_equal(xh.mask.ctypes.data, m.ctypes.data)
        assert_equal(xs.mask, [0,0,0,1,0])
        assert(xh._hardmask)
        assert(not xs._hardmask)
        xh[1:4] = [10,20,30]
        xs[1:4] = [10,20,30]
        assert_equal(xh._data, [0,10,20,3,4])
        assert_equal(xs._data, [0,10,20,30,40])
        #assert_equal(xh.mask.ctypes.data, m.ctypes.data)
        assert_equal(xs.mask, nomask)
        xh[0] = masked
        xs[0] = masked
        assert_equal(xh.mask, [1,0,0,1,1])
        assert_equal(xs.mask, [1,0,0,0,0])
        xh[:] = 1
        xs[:] = 1
        assert_equal(xh._data, [0,1,1,3,4])
        assert_equal(xs._data, [1,1,1,1,1])
        assert_equal(xh.mask, [1,0,0,1,1])
        assert_equal(xs.mask, nomask)
        # Switch to soft mask
        xh.soften_mask()
        xh[:] = arange(5)
        assert_equal(xh._data, [0,1,2,3,4])
        assert_equal(xh.mask, nomask)
        # Switch back to hard mask
        xh.harden_mask()
        xh[xh<3] = masked
        assert_equal(xh._data, [0,1,2,3,4])
        assert_equal(xh._mask, [1,1,1,0,0])
        xh[filled(xh>1,False)] = 5
        assert_equal(xh._data, [0,1,2,5,5])
        assert_equal(xh._mask, [1,1,1,0,0])
        #
        xh = array([[1,2],[3,4]], mask = [[1,0],[0,0]], hard_mask=True)
        xh[0] = 0
        assert_equal(xh._data, [[1,0],[3,4]])
        assert_equal(xh._mask, [[1,0],[0,0]])
        xh[-1,-1] = 5
        assert_equal(xh._data, [[1,0],[3,5]])
        assert_equal(xh._mask, [[1,0],[0,0]])
        xh[filled(xh<5,False)] = 2
        assert_equal(xh._data, [[1,2],[2,5]])
        assert_equal(xh._mask, [[1,0],[0,0]])
        #
        "Another test of hardmask"
        d = arange(5)
        n = [0,0,0,1,1]
        m = make_mask(n)
        xh = array(d, mask = m, hard_mask=True)
        xh[4:5] = 999
        #assert_equal(xh.mask.ctypes.data, m.ctypes.data)
        xh[0:1] = 999
        assert_equal(xh._data,[999,1,2,3,4])

    def test_smallmask(self):
        "Checks the behaviour of _smallmask"
        a = arange(10)
        a[1] = masked
        a[1] = 1
        assert_equal(a._mask, nomask)
        a = arange(10)
        a._smallmask = False
        a[1] = masked
        a[1] = 1
        assert_equal(a._mask, zeros(10))


    def test_sort(self):
        "Test sort"
        x = array([1,4,2,3],mask=[0,1,0,0],dtype=numpy.uint8)
        #
        sortedx = sort(x)
        assert_equal(sortedx._data,[1,2,3,4])
        assert_equal(sortedx._mask,[0,0,0,1])
        #
        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [4,1,2,3])
        assert_equal(sortedx._mask, [1,0,0,0])
        #
        x.sort()
        assert_equal(x._data,[1,2,3,4])
        assert_equal(x._mask,[0,0,0,1])
        #
        x = array([1,4,2,3],mask=[0,1,0,0],dtype=numpy.uint8)
        x.sort(endwith=False)
        assert_equal(x._data, [4,1,2,3])
        assert_equal(x._mask, [1,0,0,0])
        #
        x = [1,4,2,3]
        sortedx = sort(x)
        assert(not isinstance(sorted, MaskedArray))
        #
        x = array([0,1,-1,-2,2], mask=nomask, dtype=numpy.int8)
        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [-2,-1,0,1,2])
        x = array([0,1,-1,-2,2], mask=[0,1,0,0,1], dtype=numpy.int8)
        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [1,2,-2,-1,0])
        assert_equal(sortedx._mask, [1,1,0,0,0])

    def test_sort_2d(self):
        "Check sort of 2D array."
        # 2D array w/o mask
        a = masked_array([[8,4,1],[2,0,9]])
        a.sort(0)
        assert_equal(a, [[2,0,1],[8,4,9]])
        a = masked_array([[8,4,1],[2,0,9]])
        a.sort(1)
        assert_equal(a, [[1,4,8],[0,2,9]])
        # 2D array w/mask
        a = masked_array([[8,4,1],[2,0,9]], mask=[[1,0,0],[0,0,1]])
        a.sort(0)
        assert_equal(a, [[2,0,1],[8,4,9]])
        assert_equal(a._mask, [[0,0,0],[1,0,1]])
        a = masked_array([[8,4,1],[2,0,9]], mask=[[1,0,0],[0,0,1]])
        a.sort(1)
        assert_equal(a, [[1,4,8],[0,2,9]])
        assert_equal(a._mask, [[0,0,1],[0,0,1]])
        # 3D
        a = masked_array([[[7, 8, 9],[4, 5, 6],[1, 2, 3]],
                          [[1, 2, 3],[7, 8, 9],[4, 5, 6]],
                          [[7, 8, 9],[1, 2, 3],[4, 5, 6]],
                          [[4, 5, 6],[1, 2, 3],[7, 8, 9]]])
        a[a%4==0] = masked
        am = a.copy()
        an = a.filled(99)
        am.sort(0)
        an.sort(0)
        assert_equal(am, an)
        am = a.copy()
        an = a.filled(99)
        am.sort(1)
        an.sort(1)
        assert_equal(am, an)
        am = a.copy()
        an = a.filled(99)
        am.sort(2)
        an.sort(2)
        assert_equal(am, an)


    def test_ravel(self):
        "Tests ravel"
        a = array([[1,2,3,4,5]], mask=[[0,1,0,0,0]])
        aravel = a.ravel()
        assert_equal(a._mask.shape, a.shape)
        a = array([0,0], mask=[1,1])
        aravel = a.ravel()
        assert_equal(a._mask.shape, a.shape)
        a = array(numpy.matrix([1,2,3,4,5]), mask=[[0,1,0,0,0]])
        aravel = a.ravel()
        assert_equal(a.shape,(1,5))
        assert_equal(a._mask.shape, a.shape)
        # Checs that small_mask is preserved
        a = array([1,2,3,4],mask=[0,0,0,0],shrink=False)
        assert_equal(a.ravel()._mask, [0,0,0,0])

    def test_reshape(self):
        "Tests reshape"
        x = arange(4)
        x[0] = masked
        y = x.reshape(2,2)
        assert_equal(y.shape, (2,2,))
        assert_equal(y._mask.shape, (2,2,))
        assert_equal(x.shape, (4,))
        assert_equal(x._mask.shape, (4,))

    def test_compressed(self):
        "Tests compressed"
        a = array([1,2,3,4],mask=[0,0,0,0])
        b = a.compressed()
        assert_equal(b, a)
        assert_equal(b._mask, nomask)
        a[0] = masked
        b = a.compressed()
        assert_equal(b._data, [2,3,4])
        assert_equal(b._mask, nomask)

    def test_tolist(self):
        "Tests to list"
        x = array(numpy.arange(12))
        x[[1,-2]] = masked
        xlist = x.tolist()
        assert(xlist[1] is None)
        assert(xlist[-2] is None)
        #
        x.shape = (3,4)
        xlist = x.tolist()
        #
        assert_equal(xlist[0],[0,None,2,3])
        assert_equal(xlist[1],[4,5,6,7])
        assert_equal(xlist[2],[8,9,None,11])
        # Make sure a masked record is output as a tuple of None
        x = array(zip([1,2,3],
                      [1.1,2.2,3.3],
                      ['one','two','thr']),
                  dtype=[('a',int_),('b',float_),('c','|S8')])
        x[-1] = masked
        assert_equal(x.tolist(), [(1,1.1,'one'),(2,2.2,'two'),(None,None,None)])


    def test_squeeze(self):
        "Check squeeze"
        data = masked_array([[1,2,3]])
        assert_equal(data.squeeze(), [1,2,3])
        data = masked_array([[1,2,3]], mask=[[1,1,1]])
        assert_equal(data.squeeze(), [1,2,3])
        assert_equal(data.squeeze()._mask, [1,1,1])
        data = masked_array([[1]], mask=True)
        assert(data.squeeze() is masked)

    def test_putmask(self):
        x = arange(6)+1
        mx = array(x, mask=[0,0,0,1,1,1])
        mask = [0,0,1,0,0,1]
        # w/o mask, w/o masked values
        xx = x.copy()
        putmask(xx, mask, 99)
        assert_equal(xx, [1,2,99,4,5,99])
        # w/ mask, w/o masked values
        mxx = mx.copy()
        putmask(mxx, mask, 99)
        assert_equal(mxx._data, [1,2,99,4,5,99])
        assert_equal(mxx._mask, [0,0,0,1,1,0])
        # w/o mask, w/ masked values
        values = array([10,20,30,40,50,60],mask=[1,1,1,0,0,0])
        xx = x.copy()
        putmask(xx, mask, values)
        assert_equal(xx._data, [1,2,30,4,5,60])
        assert_equal(xx._mask, [0,0,1,0,0,0])
        # w/ mask, w/ masked values
        mxx = mx.copy()
        putmask(mxx, mask, values)
        assert_equal(mxx._data, [1,2,30,4,5,60])
        assert_equal(mxx._mask, [0,0,1,1,1,0])
        # w/ mask, w/ masked values + hardmask
        mxx = mx.copy()
        mxx.harden_mask()
        putmask(mxx, mask, values)
        assert_equal(mxx, [1,2,30,4,5,60])

    def test_compress(self):
        "test compress"
        a = masked_array([1., 2., 3., 4., 5.], fill_value=9999)
        condition = (a > 1.5) & (a < 3.5)
        assert_equal(a.compress(condition),[2.,3.])
        #
        a[[2,3]] = masked
        b = a.compress(condition)
        assert_equal(b._data,[2.,3.])
        assert_equal(b._mask,[0,1])
        assert_equal(b.fill_value,9999)
        assert_equal(b,a[condition])
        #
        condition = (a<4.)
        b = a.compress(condition)
        assert_equal(b._data,[1.,2.,3.])
        assert_equal(b._mask,[0,0,1])
        assert_equal(b.fill_value,9999)
        assert_equal(b,a[condition])
        #
        a = masked_array([[10,20,30],[40,50,60]], mask=[[0,0,1],[1,0,0]])
        b = a.compress(a.ravel() >= 22)
        assert_equal(b._data, [30, 40, 50, 60])
        assert_equal(b._mask, [1,1,0,0])
        #
        x = numpy.array([3,1,2])
        b = a.compress(x >= 2, axis=1)
        assert_equal(b._data, [[10,30],[40,60]])
        assert_equal(b._mask, [[0,1],[1,0]])
    #
    def test_empty(self):
        "Tests empty/like"
        datatype = [('a',int_),('b',float_),('c','|S8')]
        a = masked_array([(1,1.1,'1.1'),(2,2.2,'2.2'),(3,3.3,'3.3')],
                         dtype=datatype)
        assert_equal(len(a.fill_value), len(datatype))
        #
        b = empty_like(a)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)
        #
        b = empty(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

#..............................................................................

class TestMiscFunctions(NumpyTestCase):
    "Test class for miscellaneous functions."
    #
    def test_masked_where(self):
        x = [1,2]
        y = masked_where(False,x)
        assert_equal(y,[1,2])
        assert_equal(y[1],2)
    #
    def test_round(self):
        a = array([1.23456, 2.34567, 3.45678, 4.56789, 5.67890],
                  mask=[0,1,0,0,0])
        assert_equal(a.round(), [1., 2., 3., 5., 6.])
        assert_equal(a.round(1), [1.2, 2.3, 3.5, 4.6, 5.7])
        assert_equal(a.round(3), [1.235, 2.346, 3.457, 4.568, 5.679])
        b = empty_like(a)
        a.round(out=b)
        assert_equal(b, [1., 2., 3., 5., 6.])
    #
    def test_identity(self):
        a = identity(5)
        assert(isinstance(a, MaskedArray))
        assert_equal(a, numpy.identity(5))


###############################################################################
#------------------------------------------------------------------------------
if __name__ == "__main__":
    NumpyTest().run()
