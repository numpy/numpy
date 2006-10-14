from numpy.testing import *
from StringIO import StringIO
import pickle

set_local_path()
import numpy as N
restore_path()

rlevel = 1

class test_regression(NumpyTestCase):
    def check_invalid_round(self,level=rlevel):
        """Ticket #3"""
        v = 4.7599999999999998
        assert_array_equal(N.array([v]),N.array(v))

    def check_mem_empty(self,level=rlevel):
        """Ticket #7"""
        N.empty((1,),dtype=[('x',N.int64)])

    def check_pickle_transposed(self,level=rlevel):
        """Ticket #16"""
        a = N.transpose(N.array([[2,9],[7,0],[3,8]]))
        f = StringIO()
        pickle.dump(a,f)
        f.seek(0)
        b = pickle.load(f)
        f.close()
        assert_array_equal(a,b)

    def check_masked_array_create(self,level=rlevel):
        """Ticket #17"""
        x = N.ma.masked_array([0,1,2,3,0,4,5,6],mask=[0,0,0,1,1,1,0,0])
        assert_array_equal(N.ma.nonzero(x),[[1,2,6,7]])

    def check_poly1d(self,level=rlevel):
        """Ticket #28"""
        assert_equal(N.poly1d([1]) - N.poly1d([1,0]),
                     N.poly1d([-1,1]))

    def check_typeNA(self,level=rlevel):
        """Ticket #31"""
        assert_equal(N.typeNA[N.int64],'Int64')
        assert_equal(N.typeNA[N.uint64],'UInt64')

    def check_dtype_names(self,level=rlevel):
        """Ticket #35"""
        dt = N.dtype([(('name','label'),N.int32,3)])

    def check_reduce(self,level=rlevel):
        """Ticket #40"""
        assert_almost_equal(N.add.reduce([1.,.5],dtype=None), 1.5)

    def check_zeros_order(self,level=rlevel):
        """Ticket #43"""
        N.zeros([3], int, 'C')
        N.zeros([3], order='C')
        N.zeros([3], int, order='C')

    def check_sort_bigendian(self,level=rlevel):
        """Ticket #47"""
        a = N.linspace(0, 10, 11)
        c = a.astype(N.dtype('<f8'))
        c.sort()
        assert_array_almost_equal(c, a)

    def check_negative_nd_indexing(self,level=rlevel):
        """Ticket #49"""
        c = N.arange(125).reshape((5,5,5))
        origidx = N.array([-1, 0, 1])
        idx = N.array(origidx)
        c[idx]
        assert_array_equal(idx, origidx)

    def check_char_dump(self,level=rlevel):
        """Ticket #50"""
        import tempfile
        f = StringIO()
        ca = N.char.array(N.arange(1000,1010),itemsize=4)
        ca.dump(f)
        f.seek(0)
        ca = N.load(f)
        f.close()

    def check_noncontiguous_fill(self,level=rlevel):
        """Ticket #58."""
        a = N.zeros((4,2))
        b = a[:,1]
        def rs():
            b.shape = (2,2)
        self.failUnlessRaises(AttributeError,rs)
        
    def check_bool(self,level=rlevel):
        """Ticket #60"""
        x = N.bool_(1)

    def check_masked_array(self,level=rlevel):
        """Ticket #61"""
        x = N.core.ma.array(1,mask=[1])

    def check_mem_masked_where(self,level=rlevel):
        """Ticket #62"""
        from numpy.core.ma import masked_where, MaskType
        a = N.zeros((1,1))
        b = N.zeros(a.shape, MaskType)
        c = masked_where(b,a)
        a-c

    def check_indexing1(self,level=rlevel):
        """Ticket #64"""
        descr = [('x', [('y', [('z', 'c16', (2,)),]),]),]
        buffer = ((([6j,4j],),),)
        h = N.array(buffer, dtype=descr)
        h['x']['y']['z']

    def check_indexing2(self,level=rlevel):
        """Ticket #65"""
        descr = [('x', 'i4', (2,))]
        buffer = ([3,2],)
        h = N.array(buffer, dtype=descr)
        h['x']

    def check_round(self,level=rlevel):
        """Ticket #67"""
        x = N.array([1+2j])
        assert_almost_equal(x**(-1), [1/(1+2j)])

    def check_kron_matrix(self,level=rlevel):
        """Ticket #71"""
        x = N.matrix('[1 0; 1 0]')
        assert_equal(type(N.kron(x,x)),type(x))        

    def check_scalar_compare(self,level=rlevel):
        """Ticket #72"""
        a = N.array(['test', 'auto'])
        assert_array_equal(a == 'auto', N.array([False,True]))
        self.assert_(a[1] == 'auto')
        self.assert_(a[0] != 'auto')
        b = N.linspace(0, 10, 11)
        self.assert_(b != 'auto')
        self.assert_(b[0] != 'auto')

    def check_unicode_swapping(self,level=rlevel):
        """Ticket #79"""
        ulen = 1
        ucs_value = u'\U0010FFFF'
        ua = N.array([[[ucs_value*ulen]*2]*3]*4, dtype='U%s' % ulen)
        ua2 = ua.newbyteorder()

    def check_matrix_std_argmax(self,level=rlevel):
        """Ticket #83"""
        x = N.asmatrix(N.random.uniform(0,1,(3,3)))
        self.assertEqual(x.std().shape, ())
        self.assertEqual(x.argmax().shape, ())

    def check_object_array_fill(self,level=rlevel):
        """Ticket #86"""
        x = N.zeros(1, 'O')
        x.fill([])

    def check_cov_parameters(self,level=rlevel):
        """Ticket #91"""
        x = N.random.random((3,3))
        y = x.copy()
        N.cov(x,rowvar=1)
        N.cov(y,rowvar=0)
        assert_array_equal(x,y)

    def check_mem_dtype_align(self,level=rlevel):
        """Ticket #93"""
        self.failUnlessRaises(TypeError,N.dtype,
                              {'names':['a'],'formats':['foo']},align=1)

    def check_mem_digitize(self,level=rlevel):
        """Ticket #95"""
        for i in range(100):
            N.digitize([1,2,3,4],[1,3])
            N.digitize([0,1,2,3,4],[1,3])

    def check_intp(self,level=rlevel):
        """Ticket #99"""
        i_width = N.int_(0).nbytes*2 - 1
        N.intp('0x' + 'f'*i_width,16)
        self.failUnlessRaises(OverflowError,N.intp,'0x' + 'f'*(i_width+1),16)
        self.failUnlessRaises(ValueError,N.intp,'0x1',32)
        assert_equal(255,N.intp('0xFF',16))
        assert_equal(1024,N.intp(1024))

    def check_endian_bool_indexing(self,level=rlevel):
        """Ticket #105"""
        a = N.arange(10.,dtype='>f8')
        b = N.arange(10.,dtype='<f8')
        xa = N.where((a>2) & (a<6))
        xb = N.where((b>2) & (b<6))
        ya = ((a>2) & (a<6))
        yb = ((b>2) & (b<6))
        assert_array_almost_equal(xa,ya.nonzero())
        assert_array_almost_equal(xb,yb.nonzero())
        assert(N.all(a[ya] > 0.5))
        assert(N.all(b[yb] > 0.5))

    def check_mem_dot(self,level=rlevel):
        """Ticket #106"""
        x = N.random.randn(0,1)
        y = N.random.randn(10,1)
        z = N.dot(x, N.transpose(y))

    def check_arange_endian(self,level=rlevel):
        """Ticket #111"""
        ref = N.arange(10)
        x = N.arange(10,dtype='<f8')
        assert_array_equal(ref,x)
        x = N.arange(10,dtype='>f8')
        assert_array_equal(ref,x)

    def check_longfloat_repr(self,level=rlevel):
        """Ticket #112"""
        a = N.exp(N.array([1000]),dtype=N.longfloat)
        assert(str(a[0]) in str(a))

    def check_argmax(self,level=rlevel):
        """Ticket #119"""
        a = N.random.normal(0,1,(4,5,6,7,8))
        for i in xrange(a.ndim):
            aargmax = a.argmax(i)

    def check_matrix_properties(self,level=rlevel):
        """Ticket #125"""
        a = N.matrix([1.0],dtype=float)
        assert(type(a.real) is N.matrix)
        assert(type(a.imag) is N.matrix)
        c,d = N.matrix([0.0]).nonzero()
        assert(type(c) is N.matrix)
        assert(type(d) is N.matrix)

    def check_mem_divmod(self,level=rlevel):
        """Ticket #126"""
        for i in range(10):
            divmod(N.array([i])[0],10)


    def check_hstack_invalid_dims(self,level=rlevel):
        """Ticket #128"""
        x = N.arange(9).reshape((3,3))
        y = N.array([0,0,0])
        self.failUnlessRaises(ValueError,N.hstack,(x,y))

    def check_squeeze_type(self,level=rlevel):
        """Ticket #133"""
        a = N.array([3])
        b = N.array(3)
        assert(type(a.squeeze()) is N.ndarray)
        assert(type(b.squeeze()) is N.ndarray)

    def check_add_identity(self,level=rlevel):
        """Ticket #143"""
        assert_equal(0,N.add.identity)

    def check_binary_repr_0(self,level=rlevel):
        """Ticket #151"""
        assert_equal('0',N.binary_repr(0))

    def check_rec_iterate(self,level=rlevel):
        """Ticket #160"""
        descr = N.dtype([('i',int),('f',float),('s','|S3')])
        x = N.rec.array([(1,1.1,'1.0'),
                         (2,2.2,'2.0')],dtype=descr)
        x[0].tolist()
        [i for i in x[0]]

    def check_unicode_string_comparison(self,level=rlevel):
        """Ticket #190"""
        a = N.array('hello',N.unicode_)
        b = N.array('world')
        a == b

    def check_tostring_FORTRANORDER_discontiguous(self,level=rlevel):
        """Fix in r2836"""
        # Create discontiguous Fortran-ordered array
        x = N.array(N.random.rand(3,3),order='F')[:,:2]
        assert_array_almost_equal(x.ravel(),N.fromstring(x.tostring()))

    def check_flat_assignment(self,level=rlevel):
        """Correct behaviour of ticket #194"""
        x = N.empty((3,1))
        x.flat = N.arange(3)
        assert_array_almost_equal(x,[[0],[1],[2]])
        x.flat = N.arange(3,dtype=float)
        assert_array_almost_equal(x,[[0],[1],[2]])

    def check_broadcast_flat_assignment(self,level=rlevel):
        """Ticket #194"""
        x = N.empty((3,1))
        def bfa(): x[:] = N.arange(3)
        def bfb(): x[:] = N.arange(3,dtype=float)
        self.failUnlessRaises(ValueError, bfa)
        self.failUnlessRaises(ValueError, bfb)

    def check_unpickle_dtype_with_object(self,level=rlevel):
        """Implemented in r2840"""
        dt = N.dtype([('x',int),('y',N.object_),('z','O')])
        f = StringIO()
        pickle.dump(dt,f)
        f.seek(0)
        dt_ = pickle.load(f)
        f.close()
        assert_equal(dt,dt_)

    def check_mem_array_creation_invalid_specification(self,level=rlevel):
        """Ticket #196"""
        dt = N.dtype([('x',int),('y',N.object_)])
        # Wrong way
        self.failUnlessRaises(ValueError, N.array, [1,'object'], dt)
        # Correct way
        N.array([(1,'object')],dt)

    def check_recarray_single_element(self,level=rlevel):
        """Ticket #202"""
        a = N.array([1,2,3],dtype=N.int32)
        b = a.copy()
        r = N.rec.array(a,shape=1,formats=['3i4'],names=['d'])
        assert_array_equal(a,b)
        assert_equal(a,r[0][0])

    def check_zero_sized_array_indexing(self,level=rlevel):
        """Ticket #205"""
        tmp = N.array([])
        def index_tmp(): tmp[N.array(10)]
        self.failUnlessRaises(IndexError, index_tmp)

    def check_unique_zero_sized(self,level=rlevel):
        """Ticket #205"""
        assert_array_equal([], N.unique(N.array([])))
        
    def check_chararray_rstrip(self,level=rlevel):
        """Ticket #222"""
        x = N.chararray((1,),5)
        x[0] = 'a   '
        x = x.rstrip()
        assert_equal(x[0], 'a')
        
    def check_object_array_shape(self,level=rlevel):
        """Ticket #239"""
        assert_equal(N.array([[1,2],3,4],dtype=object).shape, (3,))
        assert_equal(N.array([[1,2],[3,4]],dtype=object).shape, (2,2))
        assert_equal(N.array([(1,2),(3,4)],dtype=object).shape, (2,2))
        assert_equal(N.array([],dtype=object).shape, (0,))
        assert_equal(N.array([[],[],[]],dtype=object).shape, (3,0))
        assert_equal(N.array([[3,4],[5,6],None],dtype=object).shape, (3,))
        
    def check_mem_around(self,level=rlevel):
        """Ticket #243"""
        x = N.zeros((1,))
        y = [0]
        decimal = 6
        N.around(abs(x-y),decimal) <= 10.0**(-decimal)
        
    def check_character_array_strip(self,level=rlevel):
        """Ticket #246"""
        x = N.char.array(("x","x ","x  "))
        for c in x: assert_equal(c,"x")

    def check_lexsort(self,level=rlevel):
        """Lexsort memory error"""
        v = N.array([1,2,3,4,5,6,7,8,9,10])
        assert_equal(N.lexsort(v),0)
        
    def check_pickle_dtype(self,level=rlevel):
        """Ticket #251"""
        import pickle
        pickle.dumps(N.float)
        
    def check_masked_array_multiply(self,level=rlevel):
        """Ticket #254"""
        a = N.ma.zeros((4,1))
        a[2,0] = N.ma.masked
        b = N.zeros((4,2))
        a*b
        b*a

    def check_swap_real(self, level=rlevel):
        """Ticket #265"""
        assert_equal(N.arange(4,dtype='>c8').imag.max(),0.0)
        assert_equal(N.arange(4,dtype='<c8').imag.max(),0.0)
        assert_equal(N.arange(4,dtype='>c8').real.max(),3.0)
        assert_equal(N.arange(4,dtype='<c8').real.max(),3.0)

    def check_object_array_from_list(self, level=rlevel):
        """Ticket #270"""
        a = N.array([1,'A',None])
        
    def check_masked_array_repeat(self, level=rlevel):
        """Ticket #271"""
        N.ma.array([1],mask=False).repeat(10)
        
    def check_multiple_assign(self, level=rlevel):
        """Ticket #273"""
        a = N.zeros((3,1),int)
        a[[1,2]] = 1
        
    def check_empty_array_type(self, level=rlevel):
        assert_equal(N.array([]).dtype, N.zeros(0).dtype)

    def check_void_coercion(self, level=rlevel):
        dt = N.dtype([('a','f4'),('b','i4')])
        x = N.zeros((1,),dt)
        assert(N.r_[x,x].dtype == dt)

    def check_void_copyswap(self, level=rlevel):
        dt = N.dtype([('one', '<i4'),('two', '<i4')])
        x = N.array((1,2), dtype=dt)
        x = x.byteswap()
        assert(x['one'] > 1 and x['two'] > 2)

    def check_method_args(self, level=rlevel):
        # Make sure methods and functions have same default axis
        # keyword and arguments
        funcs1= ['argmax', 'argmin', 'sum', ('product', 'prod'),
                 ('sometrue', 'any'),
                 ('alltrue', 'all'), 'cumsum', ('cumproduct', 'cumprod'),
                 'ptp', 'cumprod', 'prod', 'std', 'var', 'mean',
                 'round', 'min', 'max', 'argsort', 'sort']
        funcs2 = ['compress', 'take', 'repeat']
        
        for func in funcs1:
            arr = N.random.rand(8,7)
            arr2 = arr.copy()
            if isinstance(func, tuple):
                func_meth = func[1]
                func = func[0]
            else:
                func_meth = func
            res1 = getattr(arr, func_meth)()
            res2 = getattr(N, func)(arr2)
            if res1 is None:
                assert abs(arr-res2).max() < 1e-8, func
            else:
                assert abs(res1-res2).max() < 1e-8, func

        for func in funcs2:
            arr1 = N.random.rand(8,7)
            arr2 = N.random.rand(8,7)
            res1 = None
            if func == 'compress':
                arr1 = arr1.ravel()
                res1 = getattr(arr2, func)(arr1)
            else:
                arr2 = (15*arr2).astype(int).ravel()
            if res1 is None:
                res1 = getattr(arr1, func)(arr2)
            res2 = getattr(N, func)(arr1, arr2)
            assert abs(res1-res2).max() < 1e-8, func
        
    def check_mem_lexsort_strings(self, level=rlevel):
        """Ticket #298"""
        lst = ['abc','cde','fgh']
        N.lexsort((lst,))
        
    def check_fancy_index(self, level=rlevel):
        """Ticket #302"""
        x = N.array([1,2])[N.array([0])]
        assert_equal(x.shape,(1,))
        
    def check_recarray_copy(self, level=rlevel):
        """Ticket #312"""
        dt = [('x',N.int16),('y',N.float64)]
        ra = N.array([(1,2.3)], dtype=dt)
        rb = N.rec.array(ra, dtype=dt)
        rb['x'] = 2.
        assert ra['x'] != rb['x']
        
    def check_rec_fromarray(self, level=rlevel):
        """Ticket #322"""
        x1 = N.array([[1,2],[3,4],[5,6]])
        x2 = N.array(['a','dd','xyz'])
        x3 = N.array([1.1,2,3])
        N.rec.fromarrays([x1,x2,x3], formats="(2,)i4,a3,f8")
        
    def check_object_array_assign(self, level=rlevel):
        x = N.empty((2,2),object)
        x.flat[2] = (1,2,3)
        assert_equal(x.flat[2],(1,2,3))
        
    def check_ndmin_float64(self, level=rlevel):
        """Ticket #324"""
        x = N.array([1,2,3],dtype=N.float64)        
        assert_equal(N.array(x,dtype=N.float32,ndmin=2).ndim,2)        
        assert_equal(N.array(x,dtype=N.float64,ndmin=2).ndim,2)
        
    def check_mem_vectorise(self, level=rlevel):
        """Ticket #325"""
        vt = N.vectorize(lambda *args: args)
        vt(N.zeros((1,2,1)), N.zeros((2,1,1)), N.zeros((1,1,2)))
        vt(N.zeros((1,2,1)), N.zeros((2,1,1)), N.zeros((1,1,2)), N.zeros((2,2)))                
        
    def check_mem_axis_minimization(self, level=rlevel):
        """Ticket #327"""
        data = N.arange(5)
        data = N.add.outer(data,data)
        
    def check_mem_float_imag(self, level=rlevel):
        """Ticket #330"""
        N.float64(1.0).imag
        
    def check_dtype_tuple(self, level=rlevel):
        """Ticket #334"""
        assert N.dtype('i4') == N.dtype(('i4',()))
        
    def check_dtype_posttuple(self, level=rlevel):
        """Ticket #335"""
        N.dtype([('col1', '()i4')])
        
    def check_mgrid_single_element(self, level=rlevel):
        """Ticket #339"""
        assert_array_equal(N.mgrid[0:0:1j],[0])
        assert_array_equal(N.mgrid[0:0],[])
        
    def check_numeric_carray_compare(self, level=rlevel):
        """Ticket #341"""
        assert_equal(N.array([ 'X' ], 'c'),'X')        
        
    def check_string_array_size(self, level=rlevel):
        """Ticket #342"""
        self.failUnlessRaises(ValueError,
                              N.array,[['X'],['X','X','X']],'|S1')
                              
    def check_dtype_repr(self, level=rlevel):
        """Ticket #344"""
        dt1=N.dtype(('uint32', 2))
        dt2=N.dtype(('uint32', (2,)))        
        assert_equal(dt1.__repr__(), dt2.__repr__())
                
if __name__ == "__main__":
    NumpyTest().run()
