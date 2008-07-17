import tempfile

import numpy as np
from numpy.testing import *
from numpy.core import *


class TestFlags(NumpyTestCase):
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


class TestAttributes(NumpyTestCase):
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

class TestDtypedescr(NumpyTestCase):
    def check_construction(self):
        d1 = dtype('i4')
        assert_equal(d1, dtype(int32))
        d2 = dtype('f8')
        assert_equal(d2, dtype(float64))

class TestFromstring(NumpyTestCase):
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

    def test_malformed(self):
        a = fromstring('1.234 1,234', sep=' ')
        assert_array_equal(a, [1.234, 1.])

class TestZeroRank(NumpyTestCase):
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

class TestScalarIndexing(NumpyTestCase):
    def setUp(self):
        self.d = array([0,1])[0]

    def check_ellipsis_subscript(self):
        a = self.d
        self.failUnlessEqual(a[...], 0)
        self.failUnlessEqual(a[...].shape,())

    def check_empty_subscript(self):
        a = self.d
        self.failUnlessEqual(a[()], 0)
        self.failUnlessEqual(a[()].shape,())

    def check_invalid_subscript(self):
        a = self.d
        self.failUnlessRaises(IndexError, lambda x: x[0], a)
        self.failUnlessRaises(IndexError, lambda x: x[array([], int)], a)

    def check_invalid_subscript_assignment(self):
        a = self.d
        def assign(x, i, v):
            x[i] = v
        self.failUnlessRaises(TypeError, assign, a, 0, 42)

    def check_newaxis(self):
        a = self.d
        self.failUnlessEqual(a[newaxis].shape, (1,))
        self.failUnlessEqual(a[..., newaxis].shape, (1,))
        self.failUnlessEqual(a[newaxis, ...].shape, (1,))
        self.failUnlessEqual(a[..., newaxis].shape, (1,))
        self.failUnlessEqual(a[newaxis, ..., newaxis].shape, (1,1))
        self.failUnlessEqual(a[..., newaxis, newaxis].shape, (1,1))
        self.failUnlessEqual(a[newaxis, newaxis, ...].shape, (1,1))
        self.failUnlessEqual(a[(newaxis,)*10].shape, (1,)*10)

    def check_invalid_newaxis(self):
        a = self.d
        def subscript(x, i): x[i]
        self.failUnlessRaises(IndexError, subscript, a, (newaxis, 0))
        self.failUnlessRaises(IndexError, subscript, a, (newaxis,)*50)



class TestCreation(NumpyTestCase):
    def check_from_attribute(self):
        class x(object):
            def __array__(self, dtype=None):
                pass
        self.failUnlessRaises(ValueError, array, x())

    def check_from_string(self) :
        types = np.typecodes['AllInteger'] + np.typecodes['Float']
        nstr = ['123','123']
        result = array([123, 123], dtype=int)
        for type in types :
            msg = 'String conversion for %s' % type
            assert_equal(array(nstr, dtype=type), result, err_msg=msg)

class TestBool(NumpyTestCase):
    def check_test_interning(self):
        a0 = bool_(0)
        b0 = bool_(False)
        self.failUnless(a0 is b0)
        a1 = bool_(1)
        b1 = bool_(True)
        self.failUnless(a1 is b1)
        self.failUnless(array([True])[0] is a1)
        self.failUnless(array(True)[()] is a1)


class TestMethods(NumpyTestCase):
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

    def check_sort(self):
        # all c scalar sorts use the same code with different types
        # so it suffices to run a quick check with one type. The number
        # of sorted items must be greater than ~50 to check the actual
        # algorithm because quick and merge sort fall over to insertion
        # sort for small arrays.
        a = np.arange(100)
        b = a[::-1].copy()
        for kind in ['q','m','h'] :
            msg = "scalar sort, kind=%s" % kind
            c = a.copy();
            c.sort(kind=kind)
            assert_equal(c, a, msg)
            c = b.copy();
            c.sort(kind=kind)
            assert_equal(c, a, msg)

        # test complex sorts. These use the same code as the scalars
        # but the compare fuction differs.
        ai = a*1j + 1
        bi = b*1j + 1
        for kind in ['q','m','h'] :
            msg = "complex sort, real part == 1, kind=%s" % kind
            c = ai.copy();
            c.sort(kind=kind)
            assert_equal(c, ai, msg)
            c = bi.copy();
            c.sort(kind=kind)
            assert_equal(c, ai, msg)
        ai = a + 1j
        bi = b + 1j
        for kind in ['q','m','h'] :
            msg = "complex sort, imag part == 1, kind=%s" % kind
            c = ai.copy();
            c.sort(kind=kind)
            assert_equal(c, ai, msg)
            c = bi.copy();
            c.sort(kind=kind)
            assert_equal(c, ai, msg)

        # test string sorts.
        s = 'aaaaaaaa'
        a = np.array([s + chr(i) for i in range(100)])
        b = a[::-1].copy()
        for kind in ['q', 'm', 'h'] :
            msg = "string sort, kind=%s" % kind
            c = a.copy();
            c.sort(kind=kind)
            assert_equal(c, a, msg)
            c = b.copy();
            c.sort(kind=kind)
            assert_equal(c, a, msg)

        # test unicode sort.
        s = 'aaaaaaaa'
        a = np.array([s + chr(i) for i in range(100)], dtype=np.unicode)
        b = a[::-1].copy()
        for kind in ['q', 'm', 'h'] :
            msg = "unicode sort, kind=%s" % kind
            c = a.copy();
            c.sort(kind=kind)
            assert_equal(c, a, msg)
            c = b.copy();
            c.sort(kind=kind)
            assert_equal(c, a, msg)

        # todo, check object array sorts.

        # check axis handling. This should be the same for all type
        # specific sorts, so we only check it for one type and one kind
        a = np.array([[3,2],[1,0]])
        b = np.array([[1,0],[3,2]])
        c = np.array([[2,3],[0,1]])
        d = a.copy()
        d.sort(axis=0)
        assert_equal(d, b, "test sort with axis=0")
        d = a.copy()
        d.sort(axis=1)
        assert_equal(d, c, "test sort with axis=1")
        d = a.copy()
        d.sort()
        assert_equal(d, c, "test sort with default axis")
        # using None is known fail at this point
        # d = a.copy()
        # d.sort(axis=None)
        #assert_equal(d, c, "test sort with axis=None")

    def check_argsort(self):
        # all c scalar argsorts use the same code with different types
        # so it suffices to run a quick check with one type. The number
        # of sorted items must be greater than ~50 to check the actual
        # algorithm because quick and merge sort fall over to insertion
        # sort for small arrays.
        a = np.arange(100)
        b = a[::-1].copy()
        for kind in ['q','m','h'] :
            msg = "scalar argsort, kind=%s" % kind
            assert_equal(a.copy().argsort(kind=kind), a, msg)
            assert_equal(b.copy().argsort(kind=kind), b, msg)

        # test complex argsorts. These use the same code as the scalars
        # but the compare fuction differs.
        ai = a*1j + 1
        bi = b*1j + 1
        for kind in ['q','m','h'] :
            msg = "complex argsort, kind=%s" % kind
            assert_equal(ai.copy().argsort(kind=kind), a, msg)
            assert_equal(bi.copy().argsort(kind=kind), b, msg)
        ai = a + 1j
        bi = b + 1j
        for kind in ['q','m','h'] :
            msg = "complex argsort, kind=%s" % kind
            assert_equal(ai.copy().argsort(kind=kind), a, msg)
            assert_equal(bi.copy().argsort(kind=kind), b, msg)

        # test string argsorts.
        s = 'aaaaaaaa'
        a = np.array([s + chr(i) for i in range(100)])
        b = a[::-1].copy()
        r = arange(100)
        rr = r[::-1].copy()
        for kind in ['q', 'm', 'h'] :
            msg = "string argsort, kind=%s" % kind
            assert_equal(a.copy().argsort(kind=kind), r, msg)
            assert_equal(b.copy().argsort(kind=kind), rr, msg)

        # test unicode argsorts.
        s = 'aaaaaaaa'
        a = np.array([s + chr(i) for i in range(100)], dtype=np.unicode)
        b = a[::-1].copy()
        r = arange(100)
        rr = r[::-1].copy()
        for kind in ['q', 'm', 'h'] :
            msg = "unicode argsort, kind=%s" % kind
            assert_equal(a.copy().argsort(kind=kind), r, msg)
            assert_equal(b.copy().argsort(kind=kind), rr, msg)

        # todo, check object array argsorts.

        # check axis handling. This should be the same for all type
        # specific argsorts, so we only check it for one type and one kind
        a = np.array([[3,2],[1,0]])
        b = np.array([[1,1],[0,0]])
        c = np.array([[1,0],[1,0]])
        assert_equal(a.copy().argsort(axis=0), b)
        assert_equal(a.copy().argsort(axis=1), c)
        assert_equal(a.copy().argsort(), c)
        # using None is known fail at this point
        #assert_equal(a.copy().argsort(axis=None, c)

        # check that stable argsorts are stable
        r = np.arange(100)
        # scalars
        a = np.zeros(100)
        assert_equal(a.argsort(kind='m'), r)
        # complex
        a = np.zeros(100, dtype=np.complex)
        assert_equal(a.argsort(kind='m'), r)
        # string
        a = np.array(['aaaaaaaaa' for i in range(100)])
        assert_equal(a.argsort(kind='m'), r)
        # unicode
        a = np.array(['aaaaaaaaa' for i in range(100)], dtype=np.unicode)
        assert_equal(a.argsort(kind='m'), r)

    def check_flatten(self):
        x0 = np.array([[1,2,3],[4,5,6]], np.int32)
        x1 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], np.int32)
        y0 = np.array([1,2,3,4,5,6], np.int32)
        y0f = np.array([1,4,2,5,3,6], np.int32)
        y1 = np.array([1,2,3,4,5,6,7,8], np.int32)
        y1f = np.array([1,5,3,7,2,6,4,8], np.int32)
        assert_equal(x0.flatten(), y0)
        assert_equal(x0.flatten('F'), y0f)
        assert_equal(x0.flatten('F'), x0.T.flatten())
        assert_equal(x1.flatten(), y1)
        assert_equal(x1.flatten('F'), y1f)
        assert_equal(x1.flatten('F'), x1.T.flatten())

class TestSubscripting(NumpyTestCase):
    def check_test_zero_rank(self):
        x = array([1,2,3])
        self.failUnless(isinstance(x[0], int))
        self.failUnless(type(x[0, ...]) is ndarray)

class TestPickling(NumpyTestCase):
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

class TestFancyIndexing(NumpyTestCase):
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

class TestStringCompare(NumpyTestCase):
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


class TestArgmax(NumpyTestCase):
    def check_all(self):
        a = np.random.normal(0,1,(4,5,6,7,8))
        for i in xrange(a.ndim):
            amax = a.max(i)
            aargmax = a.argmax(i)
            axes = range(a.ndim)
            axes.remove(i)
            assert all(amax == aargmax.choose(*a.transpose(i,*axes)))

class TestNewaxis(NumpyTestCase):
    def check_basic(self):
        sk = array([0,-0.1,0.1])
        res = 250*sk[:,newaxis]
        assert_almost_equal(res.ravel(),250*sk)

class TestClip(NumpyTestCase):
    def _check_range(self,x,cmin,cmax):
        assert np.all(x >= cmin)
        assert np.all(x <= cmax)

    def _clip_type(self,type_group,array_max,
                   clip_min,clip_max,inplace=False,
                   expected_min=None,expected_max=None):
        if expected_min is None:
            expected_min = clip_min
        if expected_max is None:
            expected_max = clip_max

        for T in np.sctypes[type_group]:
            if sys.byteorder == 'little':
                byte_orders = ['=','>']
            else:
                byte_orders = ['<','=']

            for byteorder in byte_orders:
                dtype = np.dtype(T).newbyteorder(byteorder)

                x = (np.random.random(1000) * array_max).astype(dtype)
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
        rec = np.array([(-5, 2.0, 3.0), (5.0, 4.0, 3.0)],
                      dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
        y = rec['x'].clip(-0.3,0.5)
        self._check_range(y,-0.3,0.5)

    def check_max_or_min(self):
        val = np.array([0,1,2,3,4,5,6,7])
        x = val.clip(3)
        assert np.all(x >= 3)
        x = val.clip(min=3)
        assert np.all(x >= 3)
        x = val.clip(max=4)
        assert np.all(x <= 4)

class TestPutmask(ParametricTestCase):
    def tst_basic(self,x,T,mask,val):
        np.putmask(x,mask,val)
        assert np.all(x[mask] == T(val))
        assert x.dtype == T

    def testip_types(self):
        unchecked_types = [str, unicode, np.void, object]

        x = np.random.random(1000)*100
        mask = x < 40

        tests = []
        for val in [-100,0,15]:
            for types in np.sctypes.itervalues():
                tests.extend([(self.tst_basic,x.copy().astype(T),T,mask,val)
                              for T in types if T not in unchecked_types])
        return tests

    def test_mask_size(self):
        self.failUnlessRaises(ValueError, np.putmask,
                              np.array([1,2,3]), [True], 5)

    def tst_byteorder(self,dtype):
        x = np.array([1,2,3],dtype)
        np.putmask(x,[True,False,True],-1)
        assert_array_equal(x,[-1,2,-1])

    def testip_byteorder(self):
        return [(self.tst_byteorder,dtype) for dtype in ('>i4','<i4')]

    def test_record_array(self):
        # Note mixed byteorder.
        rec = np.array([(-5, 2.0, 3.0), (5.0, 4.0, 3.0)],
                      dtype=[('x', '<f8'), ('y', '>f8'), ('z', '<f8')])
        np.putmask(rec['x'],[True,False],10)
        assert_array_equal(rec['x'],[10,5])
        np.putmask(rec['y'],[True,False],10)
        assert_array_equal(rec['y'],[10,4])

    def test_masked_array(self):
        ## x = np.array([1,2,3])
        ## z = np.ma.array(x,mask=[True,False,False])
        ## np.putmask(z,[True,True,True],3)
        pass

class TestTake(ParametricTestCase):
    def tst_basic(self,x):
        ind = range(x.shape[0])
        assert_array_equal(x.take(ind, axis=0), x)

    def testip_types(self):
        unchecked_types = [str, unicode, np.void, object]

        x = np.random.random(24)*100
        x.shape = 2,3,4
        tests = []
        for types in np.sctypes.itervalues():
            tests.extend([(self.tst_basic,x.copy().astype(T))
                          for T in types if T not in unchecked_types])
        return tests

    def test_raise(self):
        x = np.random.random(24)*100
        x.shape = 2,3,4
        self.failUnlessRaises(IndexError, x.take, [0,1,2], axis=0)
        self.failUnlessRaises(IndexError, x.take, [-3], axis=0)
        assert_array_equal(x.take([-1], axis=0)[0], x[1])

    def test_clip(self):
        x = np.random.random(24)*100
        x.shape = 2,3,4
        assert_array_equal(x.take([-1], axis=0, mode='clip')[0], x[0])
        assert_array_equal(x.take([2], axis=0, mode='clip')[0], x[1])

    def test_wrap(self):
        x = np.random.random(24)*100
        x.shape = 2,3,4
        assert_array_equal(x.take([-1], axis=0, mode='wrap')[0], x[1])
        assert_array_equal(x.take([2], axis=0, mode='wrap')[0], x[0])
        assert_array_equal(x.take([3], axis=0, mode='wrap')[0], x[1])

    def tst_byteorder(self,dtype):
        x = np.array([1,2,3],dtype)
        assert_array_equal(x.take([0,2,1]),[1,3,2])

    def testip_byteorder(self):
        return [(self.tst_byteorder,dtype) for dtype in ('>i4','<i4')]

    def test_record_array(self):
        # Note mixed byteorder.
        rec = np.array([(-5, 2.0, 3.0), (5.0, 4.0, 3.0)],
                      dtype=[('x', '<f8'), ('y', '>f8'), ('z', '<f8')])
        rec1 = rec.take([1])
        assert rec1['x'] == 5.0 and rec1['y'] == 4.0

class TestLexsort(NumpyTestCase):
    def test_basic(self):
        a = [1,2,1,3,1,5]
        b = [0,4,5,6,2,3]
        idx = np.lexsort((b,a))
        expected_idx = np.array([0,4,2,1,3,5])
        assert_array_equal(idx,expected_idx)

        x = np.vstack((b,a))
        idx = np.lexsort(x)
        assert_array_equal(idx,expected_idx)

        assert_array_equal(x[1][idx],np.sort(x[1]))

class TestFromToFile(NumpyTestCase):
    def setUp(self):
        shape = (4,7)
        rand = np.random.random

        self.x = rand(shape) + rand(shape).astype(np.complex)*1j
        self.dtype = self.x.dtype

    def test_file(self):
        # Test disabled on Windows, since the tempfile does not flush
        # properly.  The test ensures that both filenames and file
        # objects are accepted in tofile and fromfile, so as long as
        # it runs on at least one platform, we should be ok.
        if not sys.platform.startswith('win'):
            tmp_file = tempfile.NamedTemporaryFile('wb',
                                                   prefix='numpy_tofromfile')
            self.x.tofile(tmp_file.file)
            tmp_file.flush()
            y = np.fromfile(tmp_file.name,dtype=self.dtype)
            assert_array_equal(y,self.x.flat)

    def test_filename(self):
        filename = tempfile.mktemp()
        f = open(filename,'wb')
        self.x.tofile(f)
        f.close()
        y = np.fromfile(filename,dtype=self.dtype)
        assert_array_equal(y,self.x.flat)

    def test_malformed(self):
        filename = tempfile.mktemp()
        f = open(filename,'w')
        f.write("1.234 1,234")
        f.close()
        y = np.fromfile(filename, sep=' ')
        assert_array_equal(y, [1.234, 1.])

class TestFromBuffer(ParametricTestCase):
    def tst_basic(self,buffer,expected,kwargs):
        assert_array_equal(np.frombuffer(buffer,**kwargs),expected)

    def testip_basic(self):
        tests = []
        for byteorder in ['<','>']:
            for dtype in [float,int,np.complex]:
                dt = np.dtype(dtype).newbyteorder(byteorder)
                x = (np.random.random((4,7))*5).astype(dt)
                buf = x.tostring()
                tests.append((self.tst_basic,buf,x.flat,{'dtype':dt}))
        return tests

class TestResize(NumpyTestCase):
    def test_basic(self):
        x = np.eye(3)
        x.resize((5,5))
        assert_array_equal(x.flat[:9],np.eye(3).flat)
        assert_array_equal(x[9:].flat,0)

    def test_check_reference(self):
        x = np.eye(3)
        y = x
        self.failUnlessRaises(ValueError,x.resize,(5,1))

class TestRecord(NumpyTestCase):
    def test_field_rename(self):
        dt = np.dtype([('f',float),('i',int)])
        dt.names = ['p','q']
        assert_equal(dt.names,['p','q'])

class TestView(NumpyTestCase):
    def test_basic(self):
        x = np.array([(1,2,3,4),(5,6,7,8)],dtype=[('r',np.int8),('g',np.int8),
                                                  ('b',np.int8),('a',np.int8)])
        # We must be specific about the endianness here:
        y = x.view(dtype='<i4')
        # ... and again without the keyword.
        z = x.view('<i4')
        assert_array_equal(y, z)
        assert_array_equal(y, [67305985, 134678021])

    def test_type(self):
        x = np.array([1,2,3])
        assert(isinstance(x.view(np.matrix),np.matrix))

    def test_keywords(self):
        x = np.array([(1,2)],dtype=[('a',np.int8),('b',np.int8)])
        # We must be specific about the endianness here:
        y = x.view(dtype='<i2', type=np.matrix)
        assert_array_equal(y,[[513]])

        assert(isinstance(y,np.matrix))
        assert_equal(y.dtype, np.dtype('<i2'))

class TestStats(NumpyTestCase):
    def test_subclass(self):
        class TestArray(np.ndarray):
            def __new__(cls, data, info):
                result = np.array(data)
                result = result.view(cls)
                result.info = info
                return result
            def __array_finalize__(self, obj):
                self.info = getattr(obj, "info", '')
        dat = TestArray([[1,2,3,4],[5,6,7,8]], 'jubba')
        res = dat.mean(1)
        assert res.info == dat.info
        res = dat.std(1)
        assert res.info == dat.info
        res = dat.var(1)
        assert res.info == dat.info

# Import tests without matching module names
set_local_path()
from test_unicode import *
from test_regression import *
from test_ufunc import *
restore_path()

if __name__ == "__main__":
    NumpyTest('numpy.core.multiarray').run()
