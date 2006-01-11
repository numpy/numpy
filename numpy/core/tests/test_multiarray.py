
from numpy.testing import *
from numpy.core import *

class test_flags(ScipyTestCase):
    def setUp(self):
        self.a = arange(10)

    def check_writeable(self):
        mydict = locals()
        self.a.flags.writeable = False
        self.assertRaises(RuntimeError, runstring, 'self.a[0] = 3', mydict)
        self.a.flags.writeable = True
        self.a[0] = 5
        self.a[0] = 0

    def check_otherflags(self):
        assert_equal(self.a.flags.carray, True)
        assert_equal(self.a.flags.farray, False)
        assert_equal(self.a.flags.behaved, True)
        assert_equal(self.a.flags.fnc, False)
        assert_equal(self.a.flags.forc, True)
        assert_equal(self.a.flags.owndata, True)
        assert_equal(self.a.flags.writeable, True)
        assert_equal(self.a.flags.aligned, True)
        assert_equal(self.a.flags.updateifcopy, False)
        

class test_attributes(ScipyTestCase):
    def setUp(self):
        self.one = arange(10)
        self.two = arange(20).reshape(4,5)
        self.three = arange(60,dtype=float64).reshape(2,5,6)

    def check_attributes(self):
        assert_equal(self.one.shape, (10,))
        assert_equal(self.two.shape, (4,5))
        assert_equal(self.three.shape, (2,5,6))
        self.three.shape = (10,3,2)
        assert_equal(self.three.shape, (10,3,2))
        self.three.shape = (2,5,6)
        assert_equal(self.one.strides, (self.one.itemsize,))
        num = self.two.itemsize
        assert_equal(self.two.strides, (5*num, num))
        num = self.three.itemsize
        assert_equal(self.three.strides, (30*num, 6*num, num))
        assert_equal(self.one.ndim, 1)
        assert_equal(self.two.ndim, 2)
        assert_equal(self.three.ndim, 3)
        num = self.two.itemsize        
        assert_equal(self.two.size, 20)
        assert_equal(self.two.nbytes, 20*num)
        assert_equal(self.two.itemsize, self.two.dtypedescr.itemsize)
        assert_equal(self.two.base, arange(20))

    def check_dtypeattr(self):
        assert_equal(self.one.dtype, int_)
        assert_equal(self.three.dtype, float_)
        assert_equal(self.one.dtypechar, 'l')
        assert_equal(self.three.dtypechar, 'd')
        self.failUnless(self.three.dtypestr[0] in '<>')
        assert_equal(self.one.dtypestr[1], 'i')
        assert_equal(self.three.dtypestr[1], 'f')

class test_dtypedescr(ScipyTestCase):
    def check_construction(self):
        d1 = dtypedescr('i4')
        assert_equal(d1, dtypedescr(int32))
        d2 = dtypedescr('f8')
        assert_equal(d2, dtypedescr(float64))
        
class test_zero_rank(ScipyTestCase):
    def setUp(self):
        self.d = array(0), array('x', object)
        
    def check_ellipsis_subscript(self):
        a,b = self.d
        self.failUnlessEqual(a[...], 0)
        self.failUnlessEqual(b[...].item(), 'x')
        self.failUnless(type(a[...]) is a.dtype)
        self.failUnless(type(b[...]) is b.dtype)
        
    def check_empty_subscript(self):
        a,b = self.d
        self.failUnlessEqual(a[()], 0)
        self.failUnlessEqual(b[()].item(), 'x')
        self.failUnless(type(a[()]) is a.dtype)
        self.failUnless(type(b[()]) is b.dtype)

    def check_invalid_subscript(self):
        a,b = self.d
        self.failUnlessRaises(IndexError, lambda x: x[0], a)
        self.failUnlessRaises(IndexError, lambda x: x[0], b)
        self.failUnlessRaises(IndexError, lambda x: x[array([], int)], a)
        self.failUnlessRaises(IndexError, lambda x: x[array([], int)], b)

    def check_ellipsis_subscript_assignment(self):
        a,b = self.d
        a[...] = 42
        self.failUnlessEqual(a, 42)
        b[...] = ''
        self.failUnlessEqual(b.item(), '')
        
    def check_empty_subscript_assignment(self):
        a,b = self.d
        a[()] = 42
        self.failUnlessEqual(a, 42)
        b[()] = ''
        self.failUnlessEqual(b.item(), '')

    def check_invalid_subscript_assignment(self):
        a,b = self.d
        def assign(x, i, v):
            x[i] = v
        self.failUnlessRaises(IndexError, assign, a, 0, 42)
        self.failUnlessRaises(IndexError, assign, b, 0, '')
        self.failUnlessRaises(TypeError, assign, a, (), '')

    def check_newaxis(self):
        a,b = self.d
        self.failUnlessEqual(a[newaxis].shape, (1,))
        self.failUnlessEqual(a[..., newaxis].shape, (1,))
        self.failUnlessEqual(a[newaxis, ...].shape, (1,))
        self.failUnlessEqual(a[..., newaxis].shape, (1,))
        self.failUnlessEqual(a[newaxis, ..., newaxis].shape, (1,1))
        self.failUnlessEqual(a[..., newaxis, newaxis].shape, (1,1))
        self.failUnlessEqual(a[newaxis, newaxis, ...].shape, (1,1))
        self.failUnlessEqual(a[(newaxis,)*10].shape, (1,)*10)

    def check_invalid_newaxis(self):
        a,b = self.d
        def subscript(x, i): x[i]
        self.failUnlessRaises(IndexError, subscript, a, (newaxis, 0))
        self.failUnlessRaises(IndexError, subscript, a, (newaxis,)*50)

if __name__ == "__main__":
        ScipyTest('numpy.core.multiarray').run()
