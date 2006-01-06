
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
        
        
    
