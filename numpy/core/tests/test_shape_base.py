from numpy.testing import *
from numpy.core import array, atleast_1d, atleast_2d, atleast_3d, vstack, \
        hstack, newaxis

class TestAtleast1d(TestCase):
    def test_0D_array(self):
        a = array(1); b = array(2);
        res=map(atleast_1d,[a,b])
        desired = [array([1]),array([2])]
        assert_array_equal(res,desired)

    def test_1D_array(self):
        a = array([1,2]); b = array([2,3]);
        res=map(atleast_1d,[a,b])
        desired = [array([1,2]),array([2,3])]
        assert_array_equal(res,desired)

    def test_2D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        res=map(atleast_1d,[a,b])
        desired = [a,b]
        assert_array_equal(res,desired)

    def test_3D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        a = array([a,a]);b = array([b,b]);
        res=map(atleast_1d,[a,b])
        desired = [a,b]
        assert_array_equal(res,desired)

    def test_r1array(self):
        """ Test to make sure equivalent Travis O's r1array function
        """
        assert(atleast_1d(3).shape == (1,))
        assert(atleast_1d(3j).shape == (1,))
        assert(atleast_1d(3L).shape == (1,))
        assert(atleast_1d(3.0).shape == (1,))
        assert(atleast_1d([[2,3],[4,5]]).shape == (2,2))

class TestAtleast2d(TestCase):
    def test_0D_array(self):
        a = array(1); b = array(2);
        res=map(atleast_2d,[a,b])
        desired = [array([[1]]),array([[2]])]
        assert_array_equal(res,desired)

    def test_1D_array(self):
        a = array([1,2]); b = array([2,3]);
        res=map(atleast_2d,[a,b])
        desired = [array([[1,2]]),array([[2,3]])]
        assert_array_equal(res,desired)

    def test_2D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        res=map(atleast_2d,[a,b])
        desired = [a,b]
        assert_array_equal(res,desired)

    def test_3D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        a = array([a,a]);b = array([b,b]);
        res=map(atleast_2d,[a,b])
        desired = [a,b]
        assert_array_equal(res,desired)

    def test_r2array(self):
        """ Test to make sure equivalent Travis O's r2array function
        """
        assert(atleast_2d(3).shape == (1,1))
        assert(atleast_2d([3j,1]).shape == (1,2))
        assert(atleast_2d([[[3,1],[4,5]],[[3,5],[1,2]]]).shape == (2,2,2))


class TestAtleast3d(TestCase):
    def test_0D_array(self):
        a = array(1); b = array(2);
        res=map(atleast_3d,[a,b])
        desired = [array([[[1]]]),array([[[2]]])]
        assert_array_equal(res,desired)

    def test_1D_array(self):
        a = array([1,2]); b = array([2,3]);
        res=map(atleast_3d,[a,b])
        desired = [array([[[1],[2]]]),array([[[2],[3]]])]
        assert_array_equal(res,desired)

    def test_2D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        res=map(atleast_3d,[a,b])
        desired = [a[:,:,newaxis],b[:,:,newaxis]]
        assert_array_equal(res,desired)

    def test_3D_array(self):
        a = array([[1,2],[1,2]]); b = array([[2,3],[2,3]]);
        a = array([a,a]);b = array([b,b]);
        res=map(atleast_3d,[a,b])
        desired = [a,b]
        assert_array_equal(res,desired)

class TestHstack(TestCase):
    def test_0D_array(self):
        a = array(1); b = array(2);
        res=hstack([a,b])
        desired = array([1,2])
        assert_array_equal(res,desired)

    def test_1D_array(self):
        a = array([1]); b = array([2]);
        res=hstack([a,b])
        desired = array([1,2])
        assert_array_equal(res,desired)

    def test_2D_array(self):
        a = array([[1],[2]]); b = array([[1],[2]]);
        res=hstack([a,b])
        desired = array([[1,1],[2,2]])
        assert_array_equal(res,desired)

class TestVstack(TestCase):
    def test_0D_array(self):
        a = array(1); b = array(2);
        res=vstack([a,b])
        desired = array([[1],[2]])
        assert_array_equal(res,desired)

    def test_1D_array(self):
        a = array([1]); b = array([2]);
        res=vstack([a,b])
        desired = array([[1],[2]])
        assert_array_equal(res,desired)

    def test_2D_array(self):
        a = array([[1],[2]]); b = array([[1],[2]]);
        res=vstack([a,b])
        desired = array([[1],[2],[1],[2]])
        assert_array_equal(res,desired)

    def test_2D_array2(self):
        a = array([1,2]); b = array([1,2]);
        res=vstack([a,b])
        desired = array([[1,2],[1,2]])
        assert_array_equal(res,desired)

