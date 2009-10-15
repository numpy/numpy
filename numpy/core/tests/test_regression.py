from StringIO import StringIO
import pickle
import sys
import gc
from os import path
from numpy.testing import *
from numpy.testing.utils import _assert_valid_refcount
import numpy as np

rlevel = 1

class TestRegression(TestCase):
    def test_invalid_round(self,level=rlevel):
        """Ticket #3"""
        v = 4.7599999999999998
        assert_array_equal(np.array([v]),np.array(v))

    def test_mem_empty(self,level=rlevel):
        """Ticket #7"""
        np.empty((1,),dtype=[('x',np.int64)])

    def test_pickle_transposed(self,level=rlevel):
        """Ticket #16"""
        a = np.transpose(np.array([[2,9],[7,0],[3,8]]))
        f = StringIO()
        pickle.dump(a,f)
        f.seek(0)
        b = pickle.load(f)
        f.close()
        assert_array_equal(a,b)

    def test_typeNA(self,level=rlevel):
        """Ticket #31"""
        assert_equal(np.typeNA[np.int64],'Int64')
        assert_equal(np.typeNA[np.uint64],'UInt64')

    def test_dtype_names(self,level=rlevel):
        """Ticket #35"""
        dt = np.dtype([(('name','label'),np.int32,3)])

    def test_reduce(self,level=rlevel):
        """Ticket #40"""
        assert_almost_equal(np.add.reduce([1.,.5],dtype=None), 1.5)

    def test_zeros_order(self,level=rlevel):
        """Ticket #43"""
        np.zeros([3], int, 'C')
        np.zeros([3], order='C')
        np.zeros([3], int, order='C')

    def test_sort_bigendian(self,level=rlevel):
        """Ticket #47"""
        a = np.linspace(0, 10, 11)
        c = a.astype(np.dtype('<f8'))
        c.sort()
        assert_array_almost_equal(c, a)

    def test_negative_nd_indexing(self,level=rlevel):
        """Ticket #49"""
        c = np.arange(125).reshape((5,5,5))
        origidx = np.array([-1, 0, 1])
        idx = np.array(origidx)
        c[idx]
        assert_array_equal(idx, origidx)

    def test_char_dump(self,level=rlevel):
        """Ticket #50"""
        f = StringIO()
        ca = np.char.array(np.arange(1000,1010),itemsize=4)
        ca.dump(f)
        f.seek(0)
        ca = np.load(f)
        f.close()

    def test_noncontiguous_fill(self,level=rlevel):
        """Ticket #58."""
        a = np.zeros((5,3))
        b = a[:,:2,]
        def rs():
            b.shape = (10,)
        self.failUnlessRaises(AttributeError,rs)

    def test_bool(self,level=rlevel):
        """Ticket #60"""
        x = np.bool_(1)

    def test_indexing1(self,level=rlevel):
        """Ticket #64"""
        descr = [('x', [('y', [('z', 'c16', (2,)),]),]),]
        buffer = ((([6j,4j],),),)
        h = np.array(buffer, dtype=descr)
        h['x']['y']['z']

    def test_indexing2(self,level=rlevel):
        """Ticket #65"""
        descr = [('x', 'i4', (2,))]
        buffer = ([3,2],)
        h = np.array(buffer, dtype=descr)
        h['x']

    def test_round(self,level=rlevel):
        """Ticket #67"""
        x = np.array([1+2j])
        assert_almost_equal(x**(-1), [1/(1+2j)])

    def test_scalar_compare(self,level=rlevel):
        """Ticket #72"""
        a = np.array(['test', 'auto'])
        assert_array_equal(a == 'auto', np.array([False,True]))
        self.assert_(a[1] == 'auto')
        self.assert_(a[0] != 'auto')
        b = np.linspace(0, 10, 11)
        self.assert_(b != 'auto')
        self.assert_(b[0] != 'auto')

    def test_unicode_swapping(self,level=rlevel):
        """Ticket #79"""
        ulen = 1
        ucs_value = u'\U0010FFFF'
        ua = np.array([[[ucs_value*ulen]*2]*3]*4, dtype='U%s' % ulen)
        ua2 = ua.newbyteorder()

    def test_object_array_fill(self,level=rlevel):
        """Ticket #86"""
        x = np.zeros(1, 'O')
        x.fill([])

    def test_mem_dtype_align(self,level=rlevel):
        """Ticket #93"""
        self.failUnlessRaises(TypeError,np.dtype,
                              {'names':['a'],'formats':['foo']},align=1)

    def test_intp(self,level=rlevel):
        """Ticket #99"""
        i_width = np.int_(0).nbytes*2 - 1
        long('0x' + 'f'*i_width,16)
        #self.failUnlessRaises(OverflowError,np.intp,'0x' + 'f'*(i_width+1),16)
        #self.failUnlessRaises(ValueError,np.intp,'0x1',32)
        assert_equal(255,np.long('0xFF',16))
        assert_equal(1024,np.long(1024))

    def test_endian_bool_indexing(self,level=rlevel):
        """Ticket #105"""
        a = np.arange(10.,dtype='>f8')
        b = np.arange(10.,dtype='<f8')
        xa = np.where((a>2) & (a<6))
        xb = np.where((b>2) & (b<6))
        ya = ((a>2) & (a<6))
        yb = ((b>2) & (b<6))
        assert_array_almost_equal(xa,ya.nonzero())
        assert_array_almost_equal(xb,yb.nonzero())
        assert(np.all(a[ya] > 0.5))
        assert(np.all(b[yb] > 0.5))

    def test_mem_dot(self,level=rlevel):
        """Ticket #106"""
        x = np.random.randn(0,1)
        y = np.random.randn(10,1)
        z = np.dot(x, np.transpose(y))

    def test_arange_endian(self,level=rlevel):
        """Ticket #111"""
        ref = np.arange(10)
        x = np.arange(10,dtype='<f8')
        assert_array_equal(ref,x)
        x = np.arange(10,dtype='>f8')
        assert_array_equal(ref,x)

#    Longfloat support is not consistent enough across
#     platforms for this test to be meaningful.
#    def test_longfloat_repr(self,level=rlevel):
#        """Ticket #112"""
#        if np.longfloat(0).itemsize > 8:
#            a = np.exp(np.array([1000],dtype=np.longfloat))
#            assert(str(a)[1:9] == str(a[0])[:8])

    def test_argmax(self,level=rlevel):
        """Ticket #119"""
        a = np.random.normal(0,1,(4,5,6,7,8))
        for i in xrange(a.ndim):
            aargmax = a.argmax(i)

    def test_mem_divmod(self,level=rlevel):
        """Ticket #126"""
        for i in range(10):
            divmod(np.array([i])[0],10)


    def test_hstack_invalid_dims(self,level=rlevel):
        """Ticket #128"""
        x = np.arange(9).reshape((3,3))
        y = np.array([0,0,0])
        self.failUnlessRaises(ValueError,np.hstack,(x,y))

    def test_squeeze_type(self,level=rlevel):
        """Ticket #133"""
        a = np.array([3])
        b = np.array(3)
        assert(type(a.squeeze()) is np.ndarray)
        assert(type(b.squeeze()) is np.ndarray)

    def test_add_identity(self,level=rlevel):
        """Ticket #143"""
        assert_equal(0,np.add.identity)

    def test_binary_repr_0(self,level=rlevel):
        """Ticket #151"""
        assert_equal('0',np.binary_repr(0))

    def test_rec_iterate(self,level=rlevel):
        """Ticket #160"""
        descr = np.dtype([('i',int),('f',float),('s','|S3')])
        x = np.rec.array([(1,1.1,'1.0'),
                         (2,2.2,'2.0')],dtype=descr)
        x[0].tolist()
        [i for i in x[0]]

    def test_unicode_string_comparison(self,level=rlevel):
        """Ticket #190"""
        a = np.array('hello',np.unicode_)
        b = np.array('world')
        a == b

    def test_tostring_FORTRANORDER_discontiguous(self,level=rlevel):
        """Fix in r2836"""
        # Create discontiguous Fortran-ordered array
        x = np.array(np.random.rand(3,3),order='F')[:,:2]
        assert_array_almost_equal(x.ravel(),np.fromstring(x.tostring()))

    def test_flat_assignment(self,level=rlevel):
        """Correct behaviour of ticket #194"""
        x = np.empty((3,1))
        x.flat = np.arange(3)
        assert_array_almost_equal(x,[[0],[1],[2]])
        x.flat = np.arange(3,dtype=float)
        assert_array_almost_equal(x,[[0],[1],[2]])

    def test_broadcast_flat_assignment(self,level=rlevel):
        """Ticket #194"""
        x = np.empty((3,1))
        def bfa(): x[:] = np.arange(3)
        def bfb(): x[:] = np.arange(3,dtype=float)
        self.failUnlessRaises(ValueError, bfa)
        self.failUnlessRaises(ValueError, bfb)

    def test_unpickle_dtype_with_object(self,level=rlevel):
        """Implemented in r2840"""
        dt = np.dtype([('x',int),('y',np.object_),('z','O')])
        f = StringIO()
        pickle.dump(dt,f)
        f.seek(0)
        dt_ = pickle.load(f)
        f.close()
        assert_equal(dt,dt_)

    def test_mem_array_creation_invalid_specification(self,level=rlevel):
        """Ticket #196"""
        dt = np.dtype([('x',int),('y',np.object_)])
        # Wrong way
        self.failUnlessRaises(ValueError, np.array, [1,'object'], dt)
        # Correct way
        np.array([(1,'object')],dt)

    def test_recarray_single_element(self,level=rlevel):
        """Ticket #202"""
        a = np.array([1,2,3],dtype=np.int32)
        b = a.copy()
        r = np.rec.array(a,shape=1,formats=['3i4'],names=['d'])
        assert_array_equal(a,b)
        assert_equal(a,r[0][0])

    def test_zero_sized_array_indexing(self,level=rlevel):
        """Ticket #205"""
        tmp = np.array([])
        def index_tmp(): tmp[np.array(10)]
        self.failUnlessRaises(IndexError, index_tmp)

    def test_chararray_rstrip(self,level=rlevel):
        """Ticket #222"""
        x = np.chararray((1,),5)
        x[0] = 'a   '
        x = x.rstrip()
        assert_equal(x[0], 'a')

    def test_object_array_shape(self,level=rlevel):
        """Ticket #239"""
        assert_equal(np.array([[1,2],3,4],dtype=object).shape, (3,))
        assert_equal(np.array([[1,2],[3,4]],dtype=object).shape, (2,2))
        assert_equal(np.array([(1,2),(3,4)],dtype=object).shape, (2,2))
        assert_equal(np.array([],dtype=object).shape, (0,))
        assert_equal(np.array([[],[],[]],dtype=object).shape, (3,0))
        assert_equal(np.array([[3,4],[5,6],None],dtype=object).shape, (3,))

    def test_mem_around(self,level=rlevel):
        """Ticket #243"""
        x = np.zeros((1,))
        y = [0]
        decimal = 6
        np.around(abs(x-y),decimal) <= 10.0**(-decimal)

    def test_character_array_strip(self,level=rlevel):
        """Ticket #246"""
        x = np.char.array(("x","x ","x  "))
        for c in x: assert_equal(c,"x")

    def test_lexsort(self,level=rlevel):
        """Lexsort memory error"""
        v = np.array([1,2,3,4,5,6,7,8,9,10])
        assert_equal(np.lexsort(v),0)

    def test_pickle_dtype(self,level=rlevel):
        """Ticket #251"""
        import pickle
        pickle.dumps(np.float)

    def test_swap_real(self, level=rlevel):
        """Ticket #265"""
        assert_equal(np.arange(4,dtype='>c8').imag.max(),0.0)
        assert_equal(np.arange(4,dtype='<c8').imag.max(),0.0)
        assert_equal(np.arange(4,dtype='>c8').real.max(),3.0)
        assert_equal(np.arange(4,dtype='<c8').real.max(),3.0)

    def test_object_array_from_list(self, level=rlevel):
        """Ticket #270"""
        a = np.array([1,'A',None])

    def test_multiple_assign(self, level=rlevel):
        """Ticket #273"""
        a = np.zeros((3,1),int)
        a[[1,2]] = 1

    def test_empty_array_type(self, level=rlevel):
        assert_equal(np.array([]).dtype, np.zeros(0).dtype)

    def test_void_copyswap(self, level=rlevel):
        dt = np.dtype([('one', '<i4'),('two', '<i4')])
        x = np.array((1,2), dtype=dt)
        x = x.byteswap()
        assert(x['one'] > 1 and x['two'] > 2)

    def test_method_args(self, level=rlevel):
        # Make sure methods and functions have same default axis
        # keyword and arguments
        funcs1= ['argmax', 'argmin', 'sum', ('product', 'prod'),
                 ('sometrue', 'any'),
                 ('alltrue', 'all'), 'cumsum', ('cumproduct', 'cumprod'),
                 'ptp', 'cumprod', 'prod', 'std', 'var', 'mean',
                 'round', 'min', 'max', 'argsort', 'sort']
        funcs2 = ['compress', 'take', 'repeat']

        for func in funcs1:
            arr = np.random.rand(8,7)
            arr2 = arr.copy()
            if isinstance(func, tuple):
                func_meth = func[1]
                func = func[0]
            else:
                func_meth = func
            res1 = getattr(arr, func_meth)()
            res2 = getattr(np, func)(arr2)
            if res1 is None:
                assert abs(arr-res2).max() < 1e-8, func
            else:
                assert abs(res1-res2).max() < 1e-8, func

        for func in funcs2:
            arr1 = np.random.rand(8,7)
            arr2 = np.random.rand(8,7)
            res1 = None
            if func == 'compress':
                arr1 = arr1.ravel()
                res1 = getattr(arr2, func)(arr1)
            else:
                arr2 = (15*arr2).astype(int).ravel()
            if res1 is None:
                res1 = getattr(arr1, func)(arr2)
            res2 = getattr(np, func)(arr1, arr2)
            assert abs(res1-res2).max() < 1e-8, func

    def test_mem_lexsort_strings(self, level=rlevel):
        """Ticket #298"""
        lst = ['abc','cde','fgh']
        np.lexsort((lst,))

    def test_fancy_index(self, level=rlevel):
        """Ticket #302"""
        x = np.array([1,2])[np.array([0])]
        assert_equal(x.shape,(1,))

    def test_recarray_copy(self, level=rlevel):
        """Ticket #312"""
        dt = [('x',np.int16),('y',np.float64)]
        ra = np.array([(1,2.3)], dtype=dt)
        rb = np.rec.array(ra, dtype=dt)
        rb['x'] = 2.
        assert ra['x'] != rb['x']

    def test_rec_fromarray(self, level=rlevel):
        """Ticket #322"""
        x1 = np.array([[1,2],[3,4],[5,6]])
        x2 = np.array(['a','dd','xyz'])
        x3 = np.array([1.1,2,3])
        np.rec.fromarrays([x1,x2,x3], formats="(2,)i4,a3,f8")

    def test_object_array_assign(self, level=rlevel):
        x = np.empty((2,2),object)
        x.flat[2] = (1,2,3)
        assert_equal(x.flat[2],(1,2,3))

    def test_ndmin_float64(self, level=rlevel):
        """Ticket #324"""
        x = np.array([1,2,3],dtype=np.float64)
        assert_equal(np.array(x,dtype=np.float32,ndmin=2).ndim,2)
        assert_equal(np.array(x,dtype=np.float64,ndmin=2).ndim,2)

    def test_mem_axis_minimization(self, level=rlevel):
        """Ticket #327"""
        data = np.arange(5)
        data = np.add.outer(data,data)

    def test_mem_float_imag(self, level=rlevel):
        """Ticket #330"""
        np.float64(1.0).imag

    def test_dtype_tuple(self, level=rlevel):
        """Ticket #334"""
        assert np.dtype('i4') == np.dtype(('i4',()))

    def test_dtype_posttuple(self, level=rlevel):
        """Ticket #335"""
        np.dtype([('col1', '()i4')])

    def test_numeric_carray_compare(self, level=rlevel):
        """Ticket #341"""
        assert_equal(np.array([ 'X' ], 'c'),'X')

    def test_string_array_size(self, level=rlevel):
        """Ticket #342"""
        self.failUnlessRaises(ValueError,
                              np.array,[['X'],['X','X','X']],'|S1')

    def test_dtype_repr(self, level=rlevel):
        """Ticket #344"""
        dt1=np.dtype(('uint32', 2))
        dt2=np.dtype(('uint32', (2,)))
        assert_equal(dt1.__repr__(), dt2.__repr__())

    def test_reshape_order(self, level=rlevel):
        """Make sure reshape order works."""
        a = np.arange(6).reshape(2,3,order='F')
        assert_equal(a,[[0,2,4],[1,3,5]])
        a = np.array([[1,2],[3,4],[5,6],[7,8]])
        b = a[:,1]
        assert_equal(b.reshape(2,2,order='F'), [[2,6],[4,8]])

    def test_repeat_discont(self, level=rlevel):
        """Ticket #352"""
        a = np.arange(12).reshape(4,3)[:,2]
        assert_equal(a.repeat(3), [2,2,2,5,5,5,8,8,8,11,11,11])

    def test_array_index(self, level=rlevel):
        """Make sure optimization is not called in this case."""
        a = np.array([1,2,3])
        a2 = np.array([[1,2,3]])
        assert_equal(a[np.where(a==3)], a2[np.where(a2==3)])

    def test_object_argmax(self, level=rlevel):
        a = np.array([1,2,3],dtype=object)
        assert a.argmax() == 2

    def test_recarray_fields(self, level=rlevel):
        """Ticket #372"""
        dt0 = np.dtype([('f0','i4'),('f1','i4')])
        dt1 = np.dtype([('f0','i8'),('f1','i8')])
        for a in [np.array([(1,2),(3,4)],"i4,i4"),
                  np.rec.array([(1,2),(3,4)],"i4,i4"),
                  np.rec.array([(1,2),(3,4)]),
                  np.rec.fromarrays([(1,2),(3,4)],"i4,i4"),
                  np.rec.fromarrays([(1,2),(3,4)])]:
            assert(a.dtype in [dt0,dt1])

    def test_random_shuffle(self, level=rlevel):
        """Ticket #374"""
        a = np.arange(5).reshape((5,1))
        b = a.copy()
        np.random.shuffle(b)
        assert_equal(np.sort(b, axis=0),a)

    def test_refcount_vdot(self, level=rlevel):
        """Changeset #3443"""
        _assert_valid_refcount(np.vdot)

    def test_startswith(self, level=rlevel):
        ca = np.char.array(['Hi','There'])
        assert_equal(ca.startswith('H'),[True,False])

    def test_noncommutative_reduce_accumulate(self, level=rlevel):
        """Ticket #413"""
        tosubtract = np.arange(5)
        todivide = np.array([2.0, 0.5, 0.25])
        assert_equal(np.subtract.reduce(tosubtract), -10)
        assert_equal(np.divide.reduce(todivide), 16.0)
        assert_array_equal(np.subtract.accumulate(tosubtract),
            np.array([0, -1, -3, -6, -10]))
        assert_array_equal(np.divide.accumulate(todivide),
            np.array([2., 4., 16.]))

    def test_convolve_empty(self, level=rlevel):
        """Convolve should raise an error for empty input array."""
        self.failUnlessRaises(ValueError,np.convolve,[],[1])
        self.failUnlessRaises(ValueError,np.convolve,[1],[])

    def test_multidim_byteswap(self, level=rlevel):
        """Ticket #449"""
        r=np.array([(1,(0,1,2))], dtype="i2,3i2")
        assert_array_equal(r.byteswap(),
                           np.array([(256,(0,256,512))],r.dtype))

    def test_string_NULL(self, level=rlevel):
        """Changeset 3557"""
        assert_equal(np.array("a\x00\x0b\x0c\x00").item(),
                     'a\x00\x0b\x0c')

    def test_junk_in_string_fields_of_recarray(self, level=rlevel):
        """Ticket #483"""
        r = np.array([['abc']], dtype=[('var1', '|S20')])
        assert str(r['var1'][0][0]) == 'abc'

    def test_take_output(self, level=rlevel):
        """Ensure that 'take' honours output parameter."""
        x = np.arange(12).reshape((3,4))
        a = np.take(x,[0,2],axis=1)
        b = np.zeros_like(a)
        np.take(x,[0,2],axis=1,out=b)
        assert_array_equal(a,b)

    def test_array_str_64bit(self, level=rlevel):
        """Ticket #501"""
        s = np.array([1, np.nan],dtype=np.float64)
        errstate = np.seterr(all='raise')
        try:
            sstr = np.array_str(s)
        finally:
            np.seterr(**errstate)

    def test_frompyfunc_endian(self, level=rlevel):
        """Ticket #503"""
        from math import radians
        uradians = np.frompyfunc(radians, 1, 1)
        big_endian = np.array([83.4, 83.5], dtype='>f8')
        little_endian = np.array([83.4, 83.5], dtype='<f8')
        assert_almost_equal(uradians(big_endian).astype(float),
                            uradians(little_endian).astype(float))

    def test_mem_string_arr(self, level=rlevel):
        """Ticket #514"""
        s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        t = []
        np.hstack((t, s ))

    def test_arr_transpose(self, level=rlevel):
        """Ticket #516"""
        x = np.random.rand(*(2,)*16)
        y = x.transpose(range(16))

    def test_string_mergesort(self, level=rlevel):
        """Ticket #540"""
        x = np.array(['a']*32)
        assert_array_equal(x.argsort(kind='m'), np.arange(32))

    def test_argmax_byteorder(self, level=rlevel):
        """Ticket #546"""
        a = np.arange(3, dtype='>f')
        assert a[a.argmax()] == a.max()

    def test_rand_seed(self, level=rlevel):
        """Ticket #555"""
        for l in np.arange(4):
            np.random.seed(l)

    def test_mem_deallocation_leak(self, level=rlevel):
        """Ticket #562"""
        a = np.zeros(5,dtype=float)
        b = np.array(a,dtype=float)
        del a, b

    def test_mem_on_invalid_dtype(self):
        "Ticket #583"
        self.failUnlessRaises(ValueError, np.fromiter, [['12',''],['13','']], str)

    def test_dot_negative_stride(self, level=rlevel):
        """Ticket #588"""
        x = np.array([[1,5,25,125.,625]])
        y = np.array([[20.],[160.],[640.],[1280.],[1024.]])
        z = y[::-1].copy()
        y2 = y[::-1]
        assert_equal(np.dot(x,z),np.dot(x,y2))

    def test_object_casting(self, level=rlevel):
        def rs():
            x = np.ones([484,286])
            y = np.zeros([484,286])
            x |= y
        self.failUnlessRaises(TypeError,rs)

    def test_unicode_scalar(self, level=rlevel):
        """Ticket #600"""
        import cPickle
        x = np.array(["DROND", "DROND1"], dtype="U6")
        el = x[1]
        new = cPickle.loads(cPickle.dumps(el))
        assert_equal(new, el)

    def test_arange_non_native_dtype(self, level=rlevel):
        """Ticket #616"""
        for T in ('>f4','<f4'):
            dt = np.dtype(T)
            assert_equal(np.arange(0,dtype=dt).dtype,dt)
            assert_equal(np.arange(0.5,dtype=dt).dtype,dt)
            assert_equal(np.arange(5,dtype=dt).dtype,dt)

    def test_bool_indexing_invalid_nr_elements(self, level=rlevel):
        s = np.ones(10,dtype=float)
        x = np.array((15,),dtype=float)
        def ia(x,s): x[(s>0)]=1.0
        self.failUnlessRaises(ValueError,ia,x,s)

    def test_mem_scalar_indexing(self, level=rlevel):
        """Ticket #603"""
        x = np.array([0],dtype=float)
        index = np.array(0,dtype=np.int32)
        x[index]

    def test_binary_repr_0_width(self, level=rlevel):
        assert_equal(np.binary_repr(0,width=3),'000')

    def test_fromstring(self, level=rlevel):
        assert_equal(np.fromstring("12:09:09", dtype=int, sep=":"),
                     [12,9,9])

    def test_searchsorted_variable_length(self, level=rlevel):
        x = np.array(['a','aa','b'])
        y = np.array(['d','e'])
        assert_equal(x.searchsorted(y), [3,3])

    def test_string_argsort_with_zeros(self, level=rlevel):
        """Check argsort for strings containing zeros."""
        x = np.fromstring("\x00\x02\x00\x01", dtype="|S2")
        assert_array_equal(x.argsort(kind='m'), np.array([1,0]))
        assert_array_equal(x.argsort(kind='q'), np.array([1,0]))

    def test_string_sort_with_zeros(self, level=rlevel):
        """Check sort for strings containing zeros."""
        x = np.fromstring("\x00\x02\x00\x01", dtype="|S2")
        y = np.fromstring("\x00\x01\x00\x02", dtype="|S2")
        assert_array_equal(np.sort(x, kind="q"), y)

    def test_copy_detection_zero_dim(self, level=rlevel):
        """Ticket #658"""
        np.indices((0,3,4)).T.reshape(-1,3)

    def test_flat_byteorder(self, level=rlevel):
        """Ticket #657"""
        x = np.arange(10)
        assert_array_equal(x.astype('>i4'),x.astype('<i4').flat[:])
        assert_array_equal(x.astype('>i4').flat[:],x.astype('<i4'))

    def test_uint64_from_negative(self, level=rlevel) :
        assert_equal(np.uint64(-2), np.uint64(18446744073709551614))

    def test_sign_bit(self, level=rlevel):
        x = np.array([0,-0.0,0])
        assert_equal(str(np.abs(x)),'[ 0.  0.  0.]')

    def test_flat_index_byteswap(self, level=rlevel):
        for dt in (np.dtype('<i4'),np.dtype('>i4')):
            x = np.array([-1,0,1],dtype=dt)
            assert_equal(x.flat[0].dtype, x[0].dtype)

    def test_copy_detection_corner_case(self, level=rlevel):
        """Ticket #658"""
        np.indices((0,3,4)).T.reshape(-1,3)

    def test_copy_detection_corner_case2(self, level=rlevel):
        """Ticket #771: strides are not set correctly when reshaping 0-sized
        arrays"""
        b = np.indices((0,3,4)).T.reshape(-1,3)
        assert_equal(b.strides, (3 * b.itemsize, b.itemsize))

    def test_object_array_refcounting(self, level=rlevel):
        """Ticket #633"""
        if not hasattr(sys, 'getrefcount'):
            return

        # NB. this is probably CPython-specific

        cnt = sys.getrefcount

        a = object()
        b = object()
        c = object()

        cnt0_a = cnt(a)
        cnt0_b = cnt(b)
        cnt0_c = cnt(c)

        # -- 0d -> 1d broadcasted slice assignment

        arr = np.zeros(5, dtype=np.object_)

        arr[:] = a
        assert cnt(a) == cnt0_a + 5

        arr[:] = b
        assert cnt(a) == cnt0_a
        assert cnt(b) == cnt0_b + 5

        arr[:2] = c
        assert cnt(b) == cnt0_b + 3
        assert cnt(c) == cnt0_c + 2

        del arr

        # -- 1d -> 2d broadcasted slice assignment

        arr  = np.zeros((5, 2), dtype=np.object_)
        arr0 = np.zeros(2, dtype=np.object_)

        arr0[0] = a
        assert cnt(a) == cnt0_a + 1
        arr0[1] = b
        assert cnt(b) == cnt0_b + 1

        arr[:,:] = arr0
        assert cnt(a) == cnt0_a + 6
        assert cnt(b) == cnt0_b + 6

        arr[:,0] = None
        assert cnt(a) == cnt0_a + 1

        del arr, arr0

        # -- 2d copying + flattening

        arr  = np.zeros((5, 2), dtype=np.object_)

        arr[:,0] = a
        arr[:,1] = b
        assert cnt(a) == cnt0_a + 5
        assert cnt(b) == cnt0_b + 5

        arr2 = arr.copy()
        assert cnt(a) == cnt0_a + 10
        assert cnt(b) == cnt0_b + 10

        arr2 = arr[:,0].copy()
        assert cnt(a) == cnt0_a + 10
        assert cnt(b) == cnt0_b + 5

        arr2 = arr.flatten()
        assert cnt(a) == cnt0_a + 10
        assert cnt(b) == cnt0_b + 10

        del arr, arr2

        # -- concatenate, repeat, take, choose

        arr1 = np.zeros((5, 1), dtype=np.object_)
        arr2 = np.zeros((5, 1), dtype=np.object_)

        arr1[...] = a
        arr2[...] = b
        assert cnt(a) == cnt0_a + 5
        assert cnt(b) == cnt0_b + 5

        arr3 = np.concatenate((arr1, arr2))
        assert cnt(a) == cnt0_a + 5 + 5
        assert cnt(b) == cnt0_b + 5 + 5

        arr3 = arr1.repeat(3, axis=0)
        assert cnt(a) == cnt0_a + 5 + 3*5

        arr3 = arr1.take([1,2,3], axis=0)
        assert cnt(a) == cnt0_a + 5 + 3

        x = np.array([[0],[1],[0],[1],[1]], int)
        arr3 = x.choose(arr1, arr2)
        assert cnt(a) == cnt0_a + 5 + 2
        assert cnt(b) == cnt0_b + 5 + 3

    def test_mem_custom_float_to_array(self, level=rlevel):
        """Ticket 702"""
        class MyFloat:
            def __float__(self):
                return 1

        tmp = np.atleast_1d([MyFloat()])
        tmp2 = tmp.astype(float)

    def test_object_array_refcount_self_assign(self, level=rlevel):
        """Ticket #711"""
        class VictimObject(object):
            deleted = False
            def __del__(self):
                self.deleted = True
        d = VictimObject()
        arr = np.zeros(5, dtype=np.object_)
        arr[:] = d
        del d
        arr[:] = arr # refcount of 'd' might hit zero here
        assert not arr[0].deleted
        arr[:] = arr # trying to induce a segfault by doing it again...
        assert not arr[0].deleted

    def test_mem_fromiter_invalid_dtype_string(self, level=rlevel):
        x = [1,2,3]
        self.failUnlessRaises(ValueError,
                              np.fromiter, [xi for xi in x], dtype='S')

    def test_reduce_big_object_array(self, level=rlevel):
        """Ticket #713"""
        oldsize = np.setbufsize(10*16)
        a = np.array([None]*161, object)
        assert not np.any(a)
        np.setbufsize(oldsize)

    def test_mem_0d_array_index(self, level=rlevel):
        """Ticket #714"""
        np.zeros(10)[np.array(0)]

    def test_floats_from_string(self, level=rlevel):
        """Ticket #640, floats from string"""
        fsingle = np.single('1.234')
        fdouble = np.double('1.234')
        flongdouble = np.longdouble('1.234')
        assert_almost_equal(fsingle, 1.234)
        assert_almost_equal(fdouble, 1.234)
        assert_almost_equal(flongdouble, 1.234)

    def test_complex_dtype_printing(self, level=rlevel):
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))], (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])])
        assert_equal(str(dt),
                     "[('top', [('tiles', ('>f4', (64, 64)), (1,)), "
                     "('rtile', '>f4', (64, 36))], (3,)), "
                     "('bottom', [('bleft', ('>f4', (8, 64)), (1,)), "
                     "('bright', '>f4', (8, 36))])]")

    def test_nonnative_endian_fill(self, level=rlevel):
        """ Non-native endian arrays were incorrectly filled with scalars before
        r5034.
        """
        if sys.byteorder == 'little':
            dtype = np.dtype('>i4')
        else:
            dtype = np.dtype('<i4')
        x = np.empty([1], dtype=dtype)
        x.fill(1)
        assert_equal(x, np.array([1], dtype=dtype))

    def test_dot_alignment_sse2(self, level=rlevel):
        """Test for ticket #551, changeset r5140"""
        x = np.zeros((30,40))
        y = pickle.loads(pickle.dumps(x))
        # y is now typically not aligned on a 8-byte boundary
        z = np.ones((1, y.shape[0]))
        # This shouldn't cause a segmentation fault:
        np.dot(z, y)

    def test_astype_copy(self, level=rlevel):
        """Ticket #788, changeset r5155"""
        # The test data file was generated by scipy.io.savemat.
        # The dtype is float64, but the isbuiltin attribute is 0.
        data_dir = path.join(path.dirname(__file__), 'data')
        filename = path.join(data_dir, "astype_copy.pkl")
        xp = pickle.load(open(filename))
        xpd = xp.astype(np.float64)
        assert (xp.__array_interface__['data'][0] !=
                xpd.__array_interface__['data'][0])

    def test_compress_small_type(self, level=rlevel):
        """Ticket #789, changeset 5217.
        """
        # compress with out argument segfaulted if cannot cast safely
        import numpy as np
        a = np.array([[1, 2], [3, 4]])
        b = np.zeros((2, 1), dtype = np.single)
        try:
            a.compress([True, False], axis = 1, out = b)
            raise AssertionError("compress with an out which cannot be " \
                                 "safely casted should not return "\
                                 "successfully")
        except TypeError:
            pass

    def test_attributes(self, level=rlevel):
        """Ticket #791
        """
        import numpy as np
        class TestArray(np.ndarray):
            def __new__(cls, data, info):
                result = np.array(data)
                result = result.view(cls)
                result.info = info
                return result
            def __array_finalize__(self, obj):
                self.info = getattr(obj, 'info', '')
        dat = TestArray([[1,2,3,4],[5,6,7,8]],'jubba')
        assert dat.info == 'jubba'
        dat.resize((4,2))
        assert dat.info == 'jubba'
        dat.sort()
        assert dat.info == 'jubba'
        dat.fill(2)
        assert dat.info == 'jubba'
        dat.put([2,3,4],[6,3,4])
        assert dat.info == 'jubba'
        dat.setfield(4, np.int32,0)
        assert dat.info == 'jubba'
        dat.setflags()
        assert dat.info == 'jubba'
        assert dat.all(1).info == 'jubba'
        assert dat.any(1).info == 'jubba'
        assert dat.argmax(1).info == 'jubba'
        assert dat.argmin(1).info == 'jubba'
        assert dat.argsort(1).info == 'jubba'
        assert dat.astype(TestArray).info == 'jubba'
        assert dat.byteswap().info == 'jubba'
        assert dat.clip(2,7).info == 'jubba'
        assert dat.compress([0,1,1]).info == 'jubba'
        assert dat.conj().info == 'jubba'
        assert dat.conjugate().info == 'jubba'
        assert dat.copy().info == 'jubba'
        dat2 = TestArray([2, 3, 1, 0],'jubba')
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        assert dat2.choose(choices).info == 'jubba'
        assert dat.cumprod(1).info == 'jubba'
        assert dat.cumsum(1).info == 'jubba'
        assert dat.diagonal().info == 'jubba'
        assert dat.flatten().info == 'jubba'
        assert dat.getfield(np.int32,0).info == 'jubba'
        assert dat.imag.info == 'jubba'
        assert dat.max(1).info == 'jubba'
        assert dat.mean(1).info == 'jubba'
        assert dat.min(1).info == 'jubba'
        assert dat.newbyteorder().info == 'jubba'
        assert dat.nonzero()[0].info == 'jubba'
        assert dat.nonzero()[1].info == 'jubba'
        assert dat.prod(1).info == 'jubba'
        assert dat.ptp(1).info == 'jubba'
        assert dat.ravel().info == 'jubba'
        assert dat.real.info == 'jubba'
        assert dat.repeat(2).info == 'jubba'
        assert dat.reshape((2,4)).info == 'jubba'
        assert dat.round().info == 'jubba'
        assert dat.squeeze().info == 'jubba'
        assert dat.std(1).info == 'jubba'
        assert dat.sum(1).info == 'jubba'
        assert dat.swapaxes(0,1).info == 'jubba'
        assert dat.take([2,3,5]).info == 'jubba'
        assert dat.transpose().info == 'jubba'
        assert dat.T.info == 'jubba'
        assert dat.var(1).info == 'jubba'
        assert dat.view(TestArray).info == 'jubba'

    def test_recarray_tolist(self, level=rlevel):
        """Ticket #793, changeset r5215
        """
        # Comparisons fail for NaN, so we can't use random memory
        # for the test.
        buf = np.zeros(40, dtype=np.int8)
        a = np.recarray(2, formats="i4,f8,f8", names="id,x,y", buf=buf)
        b = a.tolist()
        assert( a[0].tolist() == b[0])
        assert( a[1].tolist() == b[1])

    def test_char_array_creation(self, level=rlevel):
        a = np.array('123', dtype='c')
        b = np.array(['1','2','3'])
        assert_equal(a,b)

    def test_unaligned_unicode_access(self, level=rlevel) :
        """Ticket #825"""
        for i in range(1,9) :
            msg = 'unicode offset: %d chars'%i
            t = np.dtype([('a','S%d'%i),('b','U2')])
            x = np.array([('a',u'b')], dtype=t)
            assert_equal(str(x), "[('a', u'b')]", err_msg=msg)

    def test_sign_for_complex_nan(self, level=rlevel):
        """Ticket 794."""
        C = np.array([-np.inf, -2+1j, 0, 2-1j, np.inf, np.nan])
        have = np.sign(C)
        want = np.array([-1+0j, -1+0j, 0+0j, 1+0j, 1+0j, np.nan])
        assert_equal(have, want)

    def test_for_equal_names(self, level=rlevel):
        """Ticket #674"""
        dt = np.dtype([('foo', float), ('bar', float)])
        a = np.zeros(10, dt)
        b = list(a.dtype.names)
        b[0] = "notfoo"
        a.dtype.names = b
        assert a.dtype.names[0] == "notfoo"
        assert a.dtype.names[1] == "bar"

    def test_for_object_scalar_creation(self, level=rlevel):
        """Ticket #816"""
        a = np.object_()
        b = np.object_(3)
        b2 = np.object_(3.0)
        c = np.object_([4,5])
        d = np.object_([None, {}, []])
        assert a is None
        assert type(b) is int
        assert type(b2) is float
        assert type(c) is np.ndarray
        assert c.dtype == object
        assert d.dtype == object

    def test_for_zero_length_in_choose(self, level=rlevel):
        "Ticket #882"
        a = np.array(1)
        self.failUnlessRaises(ValueError, lambda x: x.choose([]), a)

    def test_array_ndmin_overflow(self):
        "Ticket #947."
        self.assertRaises(ValueError, lambda: np.array([1], ndmin=33))

    def test_errobj_reference_leak(self, level=rlevel):
        """Ticket #955"""
        z = int(0)
        p = np.int32(-1)

        gc.collect()
        n_before = len(gc.get_objects())
        z**p  # this shouldn't leak a reference to errobj
        gc.collect()
        n_after = len(gc.get_objects())
        assert n_before >= n_after, (n_before, n_after)

    def test_void_scalar_with_titles(self, level=rlevel):
        """No ticket"""
        data = [('john', 4), ('mary', 5)]
        dtype1 = [(('source:yy', 'name'), 'O'), (('source:xx', 'id'), int)]
        arr = np.array(data, dtype=dtype1)
        assert arr[0][0] == 'john'
        assert arr[0][1] == 4

    def test_blasdot_uninitialized_memory(self):
        """Ticket #950"""
        for m in [0, 1, 2]:
            for n in [0, 1, 2]:
                for k in xrange(3):
                    # Try to ensure that x->data contains non-zero floats
                    x = np.array([123456789e199], dtype=np.float64)
                    x.resize((m, 0))
                    y = np.array([123456789e199], dtype=np.float64)
                    y.resize((0, n))

                    # `dot` should just return zero (m,n) matrix
                    z = np.dot(x, y)
                    assert np.all(z == 0)
                    assert z.shape == (m, n)

    def test_zeros(self):
        """Regression test for #1061."""
        # Set a size which cannot fit into a 64 bits signed integer
        sz = 2 ** 64
        good = 'Maximum allowed dimension exceeded'
        try:
            np.empty(sz)
        except ValueError, e:
            if not str(e) == good:
                self.fail("Got msg '%s', expected '%s'" % (e, good))
        except Exception, e:
            self.fail("Got exception of type %s instead of ValueError" % type(e))

    def test_huge_arange(self):
        """Regression test for #1062."""
        # Set a size which cannot fit into a 64 bits signed integer
        sz = 2 ** 64
        good = 'Maximum allowed size exceeded'
        try:
            a = np.arange(sz)
            self.failUnless(np.size == sz)
        except ValueError, e:
            if not str(e) == good:
                self.fail("Got msg '%s', expected '%s'" % (e, good))
        except Exception, e:
            self.fail("Got exception of type %s instead of ValueError" % type(e))

    def test_fromiter_bytes(self):
        """Ticket #1058"""
        a = np.fromiter(range(10), dtype='b')
        b = np.fromiter(range(10), dtype='B')
        assert np.alltrue(a == np.array([0,1,2,3,4,5,6,7,8,9]))
        assert np.alltrue(b == np.array([0,1,2,3,4,5,6,7,8,9]))

    def test_array_too_big(self):
        """Ticket #1080."""
        assert_raises(ValueError, np.zeros, [2**10]*10)

    def test_dtype_keyerrors_(self):
        """Ticket #1106."""
        dt = np.dtype([('f1', np.uint)])
        assert_raises(KeyError, dt.__getitem__, "f2")
        assert_raises(IndexError, dt.__getitem__, 1)
        assert_raises(ValueError, dt.__getitem__, 0.0)

    def test_lexsort_buffer_length(self):
        """Ticket #1217, don't segfault."""
        a = np.ones(100, dtype=np.int8)
        b = np.ones(100, dtype=np.int32)
        i = np.lexsort((a[::-1], b))
        assert_equal(i, np.arange(100, dtype=np.int))

    def test_object_array_to_fixed_string(self):
        """Ticket #1235."""
        a = np.array(['abcdefgh', 'ijklmnop'], dtype=np.object_)
        b = np.array(a, dtype=(np.string_, 8))
        assert_equal(a, b)
        c = np.array(a, dtype=(np.string_, 5))
        assert_equal(c, np.array(['abcde', 'ijklm']))
        d = np.array(a, dtype=(np.string_, 12))
        assert_equal(a, d)
        e = np.empty((2, ), dtype=(np.string_, 8))
        e[:] = a[:]
        assert_equal(a, e)

    def test_unicode_to_string_cast(self):
        """Ticket #1240."""
        a = np.array([[u'abc', u'\u03a3'], [u'asdf', u'erw']], dtype='U')
        def fail():
            b = np.array(a, 'S4')
        self.failUnlessRaises(UnicodeEncodeError, fail)

    def test_mixed_string_unicode_array_creation(self):
        a = np.array(['1234', u'123'])
        assert a.itemsize == 16
        a = np.array([u'123', '1234'])
        assert a.itemsize == 16
        a = np.array(['1234', u'123', '12345'])
        assert a.itemsize == 20
        a = np.array([u'123', '1234', u'12345'])
        assert a.itemsize == 20
        a = np.array([u'123', '1234', u'1234'])
        assert a.itemsize == 16

    @decorators.skipif(sys.platform.startswith('sunos'))
    def test_misaligned_objects_segfault(self):
        """Ticket #1198"""
        a1 = np.zeros((10,), dtype=[('o', 'O'), ('c', 'c')])
        a2 = np.zeros((10,), 'S10')
        a1['o'] = a2

if __name__ == "__main__":
    run_module_suite()
