from numpy.testing import *
from numpy.core import *
from numpy import random
import numpy as N

class test_flags(NumpyTestCase):
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


class test_attributes(NumpyTestCase):
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
        assert_equal(self.two.itemsize, self.two.dtype.itemsize)
        assert_equal(self.two.base, arange(20))

    def check_dtypeattr(self):
        assert_equal(self.one.dtype, dtype(int_))
        assert_equal(self.three.dtype, dtype(float_))
        assert_equal(self.one.dtype.char, 'l')
        assert_equal(self.three.dtype.char, 'd')
        self.failUnless(self.three.dtype.str[0] in '<>')
        assert_equal(self.one.dtype.str[1], 'i')
        assert_equal(self.three.dtype.str[1], 'f')

    def check_stridesattr(self):
        x = self.one
        def make_array(size, offset, strides):
            return ndarray([size], buffer=x, dtype=int,
                           offset=offset*x.itemsize,
                           strides=strides*x.itemsize)
        assert_equal(make_array(4, 4, -1), array([4, 3, 2, 1]))
        self.failUnlessRaises(ValueError, make_array, 4, 4, -2)
        self.failUnlessRaises(ValueError, make_array, 4, 2, -1)
        self.failUnlessRaises(ValueError, make_array, 8, 3, 1)
        #self.failUnlessRaises(ValueError, make_array, 8, 3, 0)
        #self.failUnlessRaises(ValueError, lambda: ndarray([1], strides=4))


    def check_set_stridesattr(self):
        x = self.one
        def make_array(size, offset, strides):
            try:
                r = ndarray([size], dtype=int, buffer=x, offset=offset*x.itemsize)
            except:
                pass
            r.strides = strides=strides*x.itemsize
            return r
        assert_equal(make_array(4, 4, -1), array([4, 3, 2, 1]))
        self.failUnlessRaises(ValueError, make_array, 4, 4, -2)
        self.failUnlessRaises(ValueError, make_array, 4, 2, -1)
        self.failUnlessRaises(ValueError, make_array, 8, 3, 1)
        #self.failUnlessRaises(ValueError, make_array, 8, 3, 0)

    def check_fill(self):
        for t in "?bhilqpBHILQPfdgFDGO":
            x = empty((3,2,1), t)
            y = empty((3,2,1), t)
            x.fill(1)
            y[...] = 1
            assert_equal(x,y)

        x = array([(0,0.0), (1,1.0)], dtype='i4,f8')
        x.fill(x[0])
        assert_equal(x['f1'][1], x['f1'][0])

class test_dtypedescr(NumpyTestCase):
    def check_construction(self):
        d1 = dtype('i4')
        assert_equal(d1, dtype(int32))
        d2 = dtype('f8')
        assert_equal(d2, dtype(float64))

class test_fromstring(NumpyTestCase):
    def check_binary(self):
        a = fromstring('\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@',dtype='<f4')
        assert_array_equal(a, array([1,2,3,4]))

    def check_string(self):
        a = fromstring('1,2,3,4', sep=',')
        assert_array_equal(a, [1., 2., 3., 4.])

    def check_counted_string(self):
        a = fromstring('1,2,3,4', count=4, sep=',')
        assert_array_equal(a, [1., 2., 3., 4.])
        a = fromstring('1,2,3,4', count=3, sep=',')
        assert_array_equal(a, [1., 2., 3.])

    def check_string_with_ws(self):
        a = fromstring('1 2  3     4   ', dtype=int, sep=' ')
        assert_array_equal(a, [1, 2, 3, 4])

    def check_counted_string_with_ws(self):
        a = fromstring('1 2  3     4   ', count=3, dtype=int, sep=' ')
        assert_array_equal(a, [1, 2, 3])

    def check_ascii(self):
        a = fromstring('1 , 2 , 3 , 4', sep=',')
        b = fromstring('1,2,3,4', dtype=float, sep=',')
        assert_array_equal(a, [1.,2.,3.,4.])
        assert_array_equal(a,b)

class test_zero_rank(NumpyTestCase):
    def setUp(self):
        self.d = array(0), array('x', object)

    def check_ellipsis_subscript(self):
        a,b = self.d
        self.failUnlessEqual(a[...], 0)
        self.failUnlessEqual(b[...], 'x')
        self.failUnless(a[...] is a)
        self.failUnless(b[...] is b)

    def check_empty_subscript(self):
        a,b = self.d
        self.failUnlessEqual(a[()], 0)
        self.failUnlessEqual(b[()], 'x')
        self.failUnless(type(a[()]) is a.dtype.type)
        self.failUnless(type(b[()]) is str)

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
        self.failUnlessRaises(ValueError, assign, a, (), '')

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

    def check_constructor(self):
        x = ndarray(())
        x[()] = 5
        self.failUnlessEqual(x[()], 5)
        y = ndarray((),buffer=x)
        y[()] = 6
        self.failUnlessEqual(x[()], 6)

    def check_output(self):
        x = array(2)
        self.failUnlessRaises(ValueError, add, x, [1], x)

class test_creation(NumpyTestCase):
    def check_from_attribute(self):
        class x(object):
            def __array__(self, dtype=None):
                pass
        self.failUnlessRaises(ValueError, array, x())

class test_bool(NumpyTestCase):
    def check_test_interning(self):
        a0 = bool_(0)
        b0 = bool_(False)
        self.failUnless(a0 is b0)
        a1 = bool_(1)
        b1 = bool_(True)
        self.failUnless(a1 is b1)
        self.failUnless(array([True])[0] is a1)
        self.failUnless(array(True)[()] is a1)


class test_methods(NumpyTestCase):
    def check_test_round(self):
        assert_equal(array([1.2,1.5]).round(), [1,2])
        assert_equal(array(1.5).round(), 2)
        assert_equal(array([12.2,15.5]).round(-1), [10,20])
        assert_equal(array([12.15,15.51]).round(1), [12.2,15.5])

    def check_transpose(self):
        a = array([[1,2],[3,4]])
        assert_equal(a.transpose(), [[1,3],[2,4]])
        self.failUnlessRaises(ValueError, lambda: a.transpose(0))
        self.failUnlessRaises(ValueError, lambda: a.transpose(0,0))
        self.failUnlessRaises(ValueError, lambda: a.transpose(0,1,2))

class test_subscripting(NumpyTestCase):
    def check_test_zero_rank(self):
        x = array([1,2,3])
        self.failUnless(isinstance(x[0], int))
        self.failUnless(type(x[0, ...]) is ndarray)

class test_pickling(NumpyTestCase):
    def check_both(self):
        import pickle
        carray = array([[2,9],[7,0],[3,8]])
        tarray = transpose(carray)
        assert_equal(carray, pickle.loads(carray.dumps()))
        assert_equal(tarray, pickle.loads(tarray.dumps()))

    # version 0 pickles, using protocol=2 to pickle
    # version 0 doesn't have a version field
    def check_version0_int8(self):
        s = '\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x04\x85cnumpy\ndtype\nq\x04U\x02i1K\x00K\x01\x87Rq\x05(U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x04\x01\x02\x03\x04tb.'
        a = array([1,2,3,4], dtype=int8)
        p = loads(s)
        assert_equal(a, p)

    def check_version0_float32(self):
        s = '\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x04\x85cnumpy\ndtype\nq\x04U\x02f4K\x00K\x01\x87Rq\x05(U\x01<NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x10\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@tb.'
        a = array([1.0, 2.0, 3.0, 4.0], dtype=float32)
        p = loads(s)
        assert_equal(a, p)

    def check_version0_object(self):
        s = '\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x02\x85cnumpy\ndtype\nq\x04U\x02O8K\x00K\x01\x87Rq\x05(U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89]q\x06(}q\x07U\x01aK\x01s}q\x08U\x01bK\x02setb.'
        a = array([{'a':1}, {'b':2}])
        p = loads(s)
        assert_equal(a, p)

    # version 1 pickles, using protocol=2 to pickle
    def check_version1_int8(self):
        s = '\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x04\x85cnumpy\ndtype\nq\x04U\x02i1K\x00K\x01\x87Rq\x05(K\x01U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x04\x01\x02\x03\x04tb.'
        a = array([1,2,3,4], dtype=int8)
        p = loads(s)
        assert_equal(a, p)

    def check_version1_float32(self):
        s = '\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x04\x85cnumpy\ndtype\nq\x04U\x02f4K\x00K\x01\x87Rq\x05(K\x01U\x01<NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89U\x10\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@tb.'
        a = array([1.0, 2.0, 3.0, 4.0], dtype=float32)
        p = loads(s)
        assert_equal(a, p)

    def check_version1_object(self):
        s = '\x80\x02cnumpy.core._internal\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x02\x85cnumpy\ndtype\nq\x04U\x02O8K\x00K\x01\x87Rq\x05(K\x01U\x01|NNJ\xff\xff\xff\xffJ\xff\xff\xff\xfftb\x89]q\x06(}q\x07U\x01aK\x01s}q\x08U\x01bK\x02setb.'
        a = array([{'a':1}, {'b':2}])
        p = loads(s)
        assert_equal(a, p)

class test_fancy_indexing(NumpyTestCase):
    def check_list(self):
        x = ones((1,1))
        x[:,[0]] = 2.0
        assert_array_equal(x, array([[2.0]]))

        x = ones((1,1,1))
        x[:,:,[0]] = 2.0
        assert_array_equal(x, array([[[2.0]]]))

    def check_tuple(self):
        x = ones((1,1))
        x[:,(0,)] = 2.0
        assert_array_equal(x, array([[2.0]]))
        x = ones((1,1,1))
        x[:,:,(0,)] = 2.0
        assert_array_equal(x, array([[[2.0]]]))

class test_string_compare(NumpyTestCase):
    def check_string(self):
        g1 = array(["This","is","example"])
        g2 = array(["This","was","example"])
        assert_array_equal(g1 == g2, [g1[i] == g2[i] for i in [0,1,2]])
        assert_array_equal(g1 != g2, [g1[i] != g2[i] for i in [0,1,2]])
        assert_array_equal(g1 <= g2, [g1[i] <= g2[i] for i in [0,1,2]])
        assert_array_equal(g1 >= g2, [g1[i] >= g2[i] for i in [0,1,2]])
        assert_array_equal(g1 < g2, [g1[i] < g2[i] for i in [0,1,2]])
        assert_array_equal(g1 > g2, [g1[i] > g2[i] for i in [0,1,2]])

    def check_mixed(self):
        g1 = array(["spam","spa","spammer","and eggs"])
        g2 = "spam"
        assert_array_equal(g1 == g2, [x == g2 for x in g1])
        assert_array_equal(g1 != g2, [x != g2 for x in g1])
        assert_array_equal(g1 < g2, [x < g2 for x in g1])
        assert_array_equal(g1 > g2, [x > g2 for x in g1])
        assert_array_equal(g1 <= g2, [x <= g2 for x in g1])
        assert_array_equal(g1 >= g2, [x >= g2 for x in g1])


    def check_unicode(self):
        g1 = array([u"This",u"is",u"example"])
        g2 = array([u"This",u"was",u"example"])
        assert_array_equal(g1 == g2, [g1[i] == g2[i] for i in [0,1,2]])
        assert_array_equal(g1 != g2, [g1[i] != g2[i] for i in [0,1,2]])
        assert_array_equal(g1 <= g2, [g1[i] <= g2[i] for i in [0,1,2]])
        assert_array_equal(g1 >= g2, [g1[i] >= g2[i] for i in [0,1,2]])
        assert_array_equal(g1 < g2,  [g1[i] < g2[i] for i in [0,1,2]])
        assert_array_equal(g1 > g2,  [g1[i] > g2[i] for i in [0,1,2]])


class test_argmax(NumpyTestCase):
    def check_all(self):
        a = random.normal(0,1,(4,5,6,7,8))
        for i in xrange(a.ndim):
            amax = a.max(i)
            aargmax = a.argmax(i)
            axes = range(a.ndim)
            axes.remove(i)
            assert all(amax == aargmax.choose(*a.transpose(i,*axes)))

class test_newaxis(NumpyTestCase):
    def check_basic(self):
        sk = array([0,-0.1,0.1])
        res = 250*sk[:,newaxis]
        assert_almost_equal(res.ravel(),250*sk)

class test_clip(NumpyTestCase):
    def _check_range(self,x,cmin,cmax):
        assert N.all(x >= cmin)
        assert N.all(x <= cmax)

    def _clip_type(self,type_group,array_max,
                   clip_min,clip_max,inplace=False,
                   expected_min=None,expected_max=None):
        if expected_min is None:
            expected_min = clip_min
        if expected_max is None:
            expected_max = clip_max

        for T in N.sctypes[type_group]:
            if sys.byteorder == 'little':
                byte_orders = ['=','>']
            else:
                byte_orders = ['<','=']

            for byteorder in byte_orders:
                dtype = N.dtype(T).newbyteorder(byteorder)

                x = (N.random.random(1000) * array_max).astype(dtype)
                if inplace:
                    x.clip(clip_min,clip_max,x)
                else:
                    x = x.clip(clip_min,clip_max)
                    byteorder = '='

                if x.dtype.byteorder == '|': byteorder = '|'
                assert_equal(x.dtype.byteorder,byteorder)
                self._check_range(x,expected_min,expected_max)
        return x

    def check_basic(self):
        for inplace in [False, True]:
            self._clip_type('float',1024,-12.8,100.2, inplace=inplace)
            self._clip_type('float',1024,0,0, inplace=inplace)

            self._clip_type('int',1024,-120,100.5, inplace=inplace)
            self._clip_type('int',1024,0,0, inplace=inplace)

            x = self._clip_type('uint',1024,-120,100,expected_min=0, inplace=inplace)
            x = self._clip_type('uint',1024,0,0, inplace=inplace)

    def check_record_array(self):
        rec = N.array([(-5, 2.0, 3.0), (5.0, 4.0, 3.0)],
                      dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
        y = rec['x'].clip(-0.3,0.5)
        self._check_range(y,-0.3,0.5)

class test_putmask(ParametricTestCase):
    def tst_basic(self,x,T,mask,val):
        N.putmask(x,mask,val)
        assert N.all(x[mask] == T(val))
        assert x.dtype == T

    def testip_types(self):
        unchecked_types = [str,unicode,N.void,object]

        x = N.random.random(1000)*100
        mask = x < 40

        tests = []
        for val in [-100,0,15]:
            for types in N.sctypes.itervalues():
                tests.extend([(self.tst_basic,x.copy().astype(T),T,mask,val)
                              for T in types if T not in unchecked_types])
        return tests

    def test_mask_size(self):
        self.failUnlessRaises(ValueError,N.putmask,
                              N.array([1,2,3]),[True],5)

    def tst_byteorder(self,dtype):
        x = N.array([1,2,3],dtype)
        N.putmask(x,[True,False,True],-1)
        assert_array_equal(x,[-1,2,-1])

    def testip_byteorder(self):
        return [(self.tst_byteorder,dtype) for dtype in ('>i4','<i4')]

    def test_record_array(self):
        # Note mixed byteorder.
        rec = N.array([(-5, 2.0, 3.0), (5.0, 4.0, 3.0)],
                      dtype=[('x', '<f8'), ('y', '>f8'), ('z', '<f8')])
        N.putmask(rec['x'],[True,False],10)
        assert_array_equal(rec['x'],[10,5])
        N.putmask(rec['y'],[True,False],10)
        assert_array_equal(rec['y'],[10,4])

    def test_masked_array(self):
        ## x = N.array([1,2,3])
        ## z = N.ma.array(x,mask=[True,False,False])
        ## N.putmask(z,[True,True,True],3)
        pass

# Import tests from unicode
set_local_path()
from test_unicode import *
from test_regression import *
restore_path()

if __name__ == "__main__":
    NumpyTest('numpy.core.multiarray').run()
