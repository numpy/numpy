import unittest

import sys
from scipy_test.testing import *
set_package_path()
import scipy_base;reload(scipy_base)
from scipy_base import *
del sys.path[0]


class test_array_split(unittest.TestCase):
    def check_integer_0_split(self):
        a = arange(10)
        try:
            res = array_split(a,0)
            assert(0) # it should have thrown a value error
        except ValueError:
            pass
    def check_integer_split(self):
        a = arange(10)
        res = array_split(a,1)
        desired = [arange(10)]
        compare_results(res,desired)

        res = array_split(a,2)
        desired = [arange(5),arange(5,10)]
        compare_results(res,desired)

        res = array_split(a,3)
        desired = [arange(4),arange(4,7),arange(7,10)]
        compare_results(res,desired)

        res = array_split(a,4)
        desired = [arange(3),arange(3,6),arange(6,8),arange(8,10)]
        compare_results(res,desired)

        res = array_split(a,5)
        desired = [arange(2),arange(2,4),arange(4,6),arange(6,8),arange(8,10)]
        compare_results(res,desired)

        res = array_split(a,6)
        desired = [arange(2),arange(2,4),arange(4,6),arange(6,8),arange(8,9),
                   arange(9,10)]
        compare_results(res,desired)

        res = array_split(a,7)
        desired = [arange(2),arange(2,4),arange(4,6),arange(6,7),arange(7,8),
                   arange(8,9), arange(9,10)]
        compare_results(res,desired)

        res = array_split(a,8)
        desired = [arange(2),arange(2,4),arange(4,5),arange(5,6),arange(6,7),
                   arange(7,8), arange(8,9), arange(9,10)]
        compare_results(res,desired)

        res = array_split(a,9)
        desired = [arange(2),arange(2,3),arange(3,4),arange(4,5),arange(5,6),
                   arange(6,7), arange(7,8), arange(8,9), arange(9,10)]
        compare_results(res,desired)

        res = array_split(a,10)
        desired = [arange(1),arange(1,2),arange(2,3),arange(3,4),
                   arange(4,5),arange(5,6), arange(6,7), arange(7,8),
                   arange(8,9), arange(9,10)]
        compare_results(res,desired)

        res = array_split(a,11)
        desired = [arange(1),arange(1,2),arange(2,3),arange(3,4),
                   arange(4,5),arange(5,6), arange(6,7), arange(7,8),
                   arange(8,9), arange(9,10),array([])]
        compare_results(res,desired)
    def check_integer_split_2D_rows(self):
        a = array([arange(10),arange(10)])
        res = array_split(a,3,axis=0)
        desired = [array([arange(10)]),array([arange(10)]),array([])]
        compare_results(res,desired)
    def check_integer_split_2D_cols(self):
        a = array([arange(10),arange(10)])
        res = array_split(a,3,axis=-1)
        desired = [array([arange(4),arange(4)]),
                   array([arange(4,7),arange(4,7)]),
                   array([arange(7,10),arange(7,10)])]
        compare_results(res,desired)
    def check_integer_split_2D_default(self):
        """ This will fail if we change default axis
        """
        a = array([arange(10),arange(10)])
        res = array_split(a,3)
        desired = [array([arange(10)]),array([arange(10)]),array([])]
        compare_results(res,desired)
    #perhaps should check higher dimensions

    def check_index_split_simple(self):
        a = arange(10)
        indices = [1,5,7]
        res = array_split(a,indices,axis=-1)
        desired = [arange(0,1),arange(1,5),arange(5,7),arange(7,10)]
        compare_results(res,desired)

    def check_index_split_low_bound(self):
        a = arange(10)
        indices = [0,5,7]
        res = array_split(a,indices,axis=-1)
        desired = [array([]),arange(0,5),arange(5,7),arange(7,10)]
        compare_results(res,desired)
    def check_index_split_high_bound(self):
        a = arange(10)
        indices = [0,5,7,10,12]
        res = array_split(a,indices,axis=-1)
        desired = [array([]),arange(0,5),arange(5,7),arange(7,10),
                   array([]),array([])]
        compare_results(res,desired)
        
class test_split(unittest.TestCase):
    """* This function is essentially the same as array_split,
         except that it test if splitting will result in an
         equal split.  Only test for this case.
    *"""
    def check_equal_split(self):
        a = arange(10)
        res = split(a,2)
        desired = [arange(5),arange(5,10)]
        compare_results(res,desired)

    def check_unequal_split(self):
        a = arange(10) 
        try:
            res = split(a,3)
            assert(0) # should raise an error
        except ValueError:
            pass

class test_atleast_1d(unittest.TestCase):
    def check_0D_array(self):
        a = array(1); b = array(2);
        res=map(atleast_1d,[a,b])
        desired = [array([1]),array([2])]
        assert_array_equal(res,desired)
    def check_1D_array(self):
        a = array([1,2]); b = array([2,3]);
        res=map(atleast_1d,[a,b])
        desired = [array([1,2]),array([2,3])]
        assert_array_equal(res,desired)
    def check_2D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        res=map(atleast_1d,[a,b])
        desired = [a,b]
        assert_array_equal(res,desired)
    def check_3D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        a = array([a,a]);b = array([b,b]);
        res=map(atleast_1d,[a,b])
        desired = [a,b]
        assert_array_equal(res,desired)
    def check_r1array(self):
        """ Test to make sure equivalent Travis O's r1array function
        """
        assert(atleast_1d(3).shape == (1,))
        assert(atleast_1d(3j).shape == (1,))
        assert(atleast_1d(3L).shape == (1,))
        assert(atleast_1d(3.0).shape == (1,))
        assert(atleast_1d([[2,3],[4,5]]).shape == (2,2))

class test_atleast_2d(unittest.TestCase):
    def check_0D_array(self):
        a = array(1); b = array(2);
        res=map(atleast_2d,[a,b])
        desired = [array([[1]]),array([[2]])]
        assert_array_equal(res,desired)
    def check_1D_array(self):
        a = array([1,2]); b = array([2,3]);
        res=map(atleast_2d,[a,b])
        desired = [array([[1,2]]),array([[2,3]])]
        assert_array_equal(res,desired)
    def check_2D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        res=map(atleast_2d,[a,b])
        desired = [a,b]
        assert_array_equal(res,desired)
    def check_3D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        a = array([a,a]);b = array([b,b]);
        res=map(atleast_2d,[a,b])
        desired = [a,b]
        assert_array_equal(res,desired)
    def check_r2array(self):
        """ Test to make sure equivalent Travis O's r2array function
        """
        assert(atleast_2d(3).shape == (1,1))
        assert(atleast_2d([3j,1]).shape == (1,2))
        assert(atleast_2d([[[3,1],[4,5]],[[3,5],[1,2]]]).shape == (2,2,2))

class test_atleast_3d(unittest.TestCase):
    def check_0D_array(self):
        a = array(1); b = array(2);
        res=map(atleast_3d,[a,b])
        desired = [array([[[1]]]),array([[[2]]])]
        assert_array_equal(res,desired)
    def check_1D_array(self):
        a = array([1,2]); b = array([2,3]);
        res=map(atleast_3d,[a,b])
        desired = [array([[[1],[2]]]),array([[[2],[3]]])]
        assert_array_equal(res,desired)
    def check_2D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        res=map(atleast_3d,[a,b])
        desired = [a[:,:,NewAxis],b[:,:,NewAxis]]
        assert_array_equal(res,desired)
    def check_3D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        a = array([a,a]);b = array([b,b]);
        res=map(atleast_3d,[a,b])
        desired = [a,b]
        assert_array_equal(res,desired)

class test_hstack(unittest.TestCase):
    def check_0D_array(self):
        a = array(1); b = array(2);
        res=hstack([a,b])
        desired = array([1,2])
        assert_array_equal(res,desired)
    def check_1D_array(self):
        a = array([1]); b = array([2]);
        res=hstack([a,b])
        desired = array([1,2])
        assert_array_equal(res,desired)
    def check_2D_array(self):
        a = array([[1],[2]]); b = array([[1],[2]]);
        res=hstack([a,b])
        desired = array([[1,1],[2,2]])
        assert_array_equal(res,desired)

class test_vstack(unittest.TestCase):
    def check_0D_array(self):
        a = array(1); b = array(2);
        res=vstack([a,b])
        desired = array([[1],[2]])
        assert_array_equal(res,desired)
    def check_1D_array(self):
        a = array([1]); b = array([2]);
        res=vstack([a,b])
        desired = array([[1],[2]])
        assert_array_equal(res,desired)
    def check_2D_array(self):
        a = array([[1],[2]]); b = array([[1],[2]]);
        res=vstack([a,b])
        desired = array([[1],[2],[1],[2]])
        assert_array_equal(res,desired)
    def check_2D_array2(self):
        a = array([1,2]); b = array([1,2]);
        res=vstack([a,b])
        desired = array([[1,2],[1,2]])
        assert_array_equal(res,desired)

class test_dstack(unittest.TestCase):
    def check_0D_array(self):
        a = array(1); b = array(2);
        res=dstack([a,b])
        desired = array([[[1,2]]])
        assert_array_equal(res,desired)
    def check_1D_array(self):
        a = array([1]); b = array([2]);
        res=dstack([a,b])
        desired = array([[[1,2]]])
        assert_array_equal(res,desired)
    def check_2D_array(self):
        a = array([[1],[2]]); b = array([[1],[2]]);
        res=dstack([a,b])
        desired = array([[[1,1]],[[2,2,]]])
        assert_array_equal(res,desired)
    def check_2D_array2(self):
        a = array([1,2]); b = array([1,2]);
        res=dstack([a,b])
        desired = array([[[1,1],[2,2]]])
        assert_array_equal(res,desired)

""" array_split has more comprehensive test of splitting.
    only do simple test on hsplit, vsplit, and dsplit
"""
class test_hsplit(unittest.TestCase):
    """ only testing for integer splits.
    """
    def check_0D_array(self):
        a= array(1)
        try:
            hsplit(a,2)
            assert(0)
        except ValueError:
            pass
    def check_1D_array(self):
        a= array([1,2,3,4])
        res = hsplit(a,2)
        desired = [array([1,2]),array([3,4])]
        compare_results(res,desired)
    def check_2D_array(self):
        a= array([[1,2,3,4],
                  [1,2,3,4]])
        res = hsplit(a,2)
        desired = [array([[1,2],[1,2]]),array([[3,4],[3,4]])]
        compare_results(res,desired)

class test_vsplit(unittest.TestCase):
    """ only testing for integer splits.
    """
    def check_1D_array(self):
        a= array([1,2,3,4])
        try:
            vsplit(a,2)
            assert(0)
        except ValueError:
            pass
    def check_2D_array(self):
        a= array([[1,2,3,4],
                  [1,2,3,4]])
        res = vsplit(a,2)
        desired = [array([[1,2,3,4]]),array([[1,2,3,4]])]
        compare_results(res,desired)

class test_dsplit(unittest.TestCase):
    """ only testing for integer splits.
    """
    def check_2D_array(self):
        a= array([[1,2,3,4],
                  [1,2,3,4]])
        try:
            dsplit(a,2)
            assert(0)
        except ValueError:
            pass
    def check_3D_array(self):
        a= array([[[1,2,3,4],
                   [1,2,3,4]],
                  [[1,2,3,4],
                   [1,2,3,4]]])
        res = dsplit(a,2)
        desired = [array([[[1,2],[1,2]],[[1,2],[1,2]]]),
                   array([[[3,4],[3,4]],[[3,4],[3,4]]])]
        compare_results(res,desired)

class test_squeeze(unittest.TestCase):
    def check_basic(self):
        a = rand(20,10,10,1,1)
        b = rand(20,1,10,1,20)
        c = rand(1,1,20,10)
        assert_array_equal(squeeze(a),reshape(a,(20,10,10)))
        assert_array_equal(squeeze(b),reshape(b,(20,10,20)))
        assert_array_equal(squeeze(c),reshape(c,(20,10)))
                
# Utility

def compare_results(res,desired):
    for i in range(len(desired)):
        assert_array_equal(res[i],desired[i])


if __name__ == "__main__":
    ScipyTest('scipy_base.shape_base').run()
