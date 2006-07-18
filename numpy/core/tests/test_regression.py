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
        assert_almost_equal(N.add.reduce([1.,.5],dtype=None),
                            1.5)

if __name__ == "__main__":
    NumpyTest().run()
