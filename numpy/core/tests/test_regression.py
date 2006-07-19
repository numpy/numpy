from numpy.testing import *

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
        import pickle
        import tempfile
        
        a = N.transpose(N.array([[2,9],[7,0],[3,8]]))
        f = tempfile.TemporaryFile()
        pickle.dump(a,f)
        f.seek(0)
        b = pickle.load(f)
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

    def check_bool(self,level=rlevel):
        """Ticket #60"""
        x = N.bool_(1)

    def check_masked_array(self,level=rlevel):
        """Ticket #61"""
        x = N.core.ma.array(1,mask=[1])

    def check_indexing1(self,level=rlevel):
        """Ticket #64"""
        descr = [('x', [('y', [('z', 'c16', (2,)),]),]),]
        buffer = ((([6j,4j],),),)
        h = N.array(buffer, dtype=descr)
        h['x']['y']['z']

    def check_indexing1(self,level=rlevel):
        """Ticket #65"""
        descr = [('x', 'i4', (2,))]
        buffer = ([3,2],)
        h = N.array(buffer, dtype=descr)
        h['x']

    def check_round(self,level=rlevel):
        """Ticket #67"""
        x = N.array([1+2j])
        assert_almost_equal(x**(-1), [1/(1+2j)])

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

if __name__ == "__main__":
    NumpyTest().run()
