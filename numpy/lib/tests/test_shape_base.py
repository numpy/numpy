from __future__ import division, absolute_import, print_function

from numpy.testing import *
from numpy.lib import *
from numpy.core import *
from numpy import matrix, asmatrix

class TestApplyAlongAxis(TestCase):
    def test_simple(self):
        a = ones((20,10),'d')
        assert_array_equal(apply_along_axis(len,0,a),len(a)*ones(shape(a)[1]))

    def test_simple101(self,level=11):
        a = ones((10,101),'d')
        assert_array_equal(apply_along_axis(len,0,a),len(a)*ones(shape(a)[1]))

    def test_3d(self):
        a = arange(27).reshape((3,3,3))
        assert_array_equal(apply_along_axis(sum,0,a),
                           [[27,30,33],[36,39,42],[45,48,51]])


class TestApplyOverAxes(TestCase):
    def test_simple(self):
        a = arange(24).reshape(2,3,4)
        aoa_a = apply_over_axes(sum, a, [0,2])
        assert_array_equal(aoa_a, array([[[60],[92],[124]]]))


class TestArraySplit(TestCase):
    def test_integer_0_split(self):
        a = arange(10)
        try:
            res = array_split(a,0)
            assert_(0) # it should have thrown a value error
        except ValueError:
            pass

    def test_integer_split(self):
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

    def test_integer_split_2D_rows(self):
        a = array([arange(10),arange(10)])
        res = array_split(a,3,axis=0)
        desired = [array([arange(10)]),array([arange(10)]),array([])]
        compare_results(res,desired)

    def test_integer_split_2D_cols(self):
        a = array([arange(10),arange(10)])
        res = array_split(a,3,axis=-1)
        desired = [array([arange(4),arange(4)]),
                   array([arange(4,7),arange(4,7)]),
                   array([arange(7,10),arange(7,10)])]
        compare_results(res,desired)

    def test_integer_split_2D_default(self):
        """ This will fail if we change default axis
        """
        a = array([arange(10),arange(10)])
        res = array_split(a,3)
        desired = [array([arange(10)]),array([arange(10)]),array([])]
        compare_results(res,desired)
    #perhaps should check higher dimensions

    def test_index_split_simple(self):
        a = arange(10)
        indices = [1,5,7]
        res = array_split(a,indices,axis=-1)
        desired = [arange(0,1),arange(1,5),arange(5,7),arange(7,10)]
        compare_results(res,desired)

    def test_index_split_low_bound(self):
        a = arange(10)
        indices = [0,5,7]
        res = array_split(a,indices,axis=-1)
        desired = [array([]),arange(0,5),arange(5,7),arange(7,10)]
        compare_results(res,desired)

    def test_index_split_high_bound(self):
        a = arange(10)
        indices = [0,5,7,10,12]
        res = array_split(a,indices,axis=-1)
        desired = [array([]),arange(0,5),arange(5,7),arange(7,10),
                   array([]),array([])]
        compare_results(res,desired)


class TestSplit(TestCase):
    """* This function is essentially the same as array_split,
         except that it test if splitting will result in an
         equal split.  Only test for this case.
    *"""
    def test_equal_split(self):
        a = arange(10)
        res = split(a,2)
        desired = [arange(5),arange(5,10)]
        compare_results(res,desired)

    def test_unequal_split(self):
        a = arange(10)
        try:
            res = split(a,3)
            assert_(0) # should raise an error
        except ValueError:
            pass


class TestDstack(TestCase):
    def test_0D_array(self):
        a = array(1); b = array(2);
        res=dstack([a,b])
        desired = array([[[1,2]]])
        assert_array_equal(res,desired)

    def test_1D_array(self):
        a = array([1]); b = array([2]);
        res=dstack([a,b])
        desired = array([[[1,2]]])
        assert_array_equal(res,desired)

    def test_2D_array(self):
        a = array([[1],[2]]); b = array([[1],[2]]);
        res=dstack([a,b])
        desired = array([[[1,1]],[[2,2,]]])
        assert_array_equal(res,desired)

    def test_2D_array2(self):
        a = array([1,2]); b = array([1,2]);
        res=dstack([a,b])
        desired = array([[[1,1],[2,2]]])
        assert_array_equal(res,desired)

""" array_split has more comprehensive test of splitting.
    only do simple test on hsplit, vsplit, and dsplit
"""
class TestHsplit(TestCase):
    """ only testing for integer splits.
    """
    def test_0D_array(self):
        a= array(1)
        try:
            hsplit(a,2)
            assert_(0)
        except ValueError:
            pass

    def test_1D_array(self):
        a= array([1,2,3,4])
        res = hsplit(a,2)
        desired = [array([1,2]),array([3,4])]
        compare_results(res,desired)

    def test_2D_array(self):
        a= array([[1,2,3,4],
                  [1,2,3,4]])
        res = hsplit(a,2)
        desired = [array([[1,2],[1,2]]),array([[3,4],[3,4]])]
        compare_results(res,desired)


class TestVsplit(TestCase):
    """ only testing for integer splits.
    """
    def test_1D_array(self):
        a= array([1,2,3,4])
        try:
            vsplit(a,2)
            assert_(0)
        except ValueError:
            pass

    def test_2D_array(self):
        a= array([[1,2,3,4],
                  [1,2,3,4]])
        res = vsplit(a,2)
        desired = [array([[1,2,3,4]]),array([[1,2,3,4]])]
        compare_results(res,desired)


class TestDsplit(TestCase):
    """ only testing for integer splits.
    """
    def test_2D_array(self):
        a= array([[1,2,3,4],
                  [1,2,3,4]])
        try:
            dsplit(a,2)
            assert_(0)
        except ValueError:
            pass

    def test_3D_array(self):
        a= array([[[1,2,3,4],
                   [1,2,3,4]],
                  [[1,2,3,4],
                   [1,2,3,4]]])
        res = dsplit(a,2)
        desired = [array([[[1,2],[1,2]],[[1,2],[1,2]]]),
                   array([[[3,4],[3,4]],[[3,4],[3,4]]])]
        compare_results(res,desired)


class TestSqueeze(TestCase):
    def test_basic(self):
        a = rand(20,10,10,1,1)
        b = rand(20,1,10,1,20)
        c = rand(1,1,20,10)
        assert_array_equal(squeeze(a),reshape(a,(20,10,10)))
        assert_array_equal(squeeze(b),reshape(b,(20,10,20)))
        assert_array_equal(squeeze(c),reshape(c,(20,10)))

        # Squeezing to 0-dim should still give an ndarray
        a = [[[1.5]]]
        res = squeeze(a)
        assert_equal(res, 1.5)
        assert_equal(res.ndim, 0)
        assert_equal(type(res), ndarray)


class TestKron(TestCase):
    def test_return_type(self):
        a = ones([2,2])
        m = asmatrix(a)
        assert_equal(type(kron(a,a)), ndarray)
        assert_equal(type(kron(m,m)), matrix)
        assert_equal(type(kron(a,m)), matrix)
        assert_equal(type(kron(m,a)), matrix)
        class myarray(ndarray):
            __array_priority__ = 0.0
        ma = myarray(a.shape, a.dtype, a.data)
        assert_equal(type(kron(a,a)), ndarray)
        assert_equal(type(kron(ma,ma)), myarray)
        assert_equal(type(kron(a,ma)), ndarray)
        assert_equal(type(kron(ma,a)), myarray)


class TestTile(TestCase):
    def test_basic(self):
        a = array([0,1,2])
        b = [[1,2],[3,4]]
        assert_equal(tile(a,2), [0,1,2,0,1,2])
        assert_equal(tile(a,(2,2)), [[0,1,2,0,1,2],[0,1,2,0,1,2]])
        assert_equal(tile(a,(1,2)), [[0,1,2,0,1,2]])
        assert_equal(tile(b, 2), [[1,2,1,2],[3,4,3,4]])
        assert_equal(tile(b,(2,1)),[[1,2],[3,4],[1,2],[3,4]])
        assert_equal(tile(b,(2,2)),[[1,2,1,2],[3,4,3,4],
                                    [1,2,1,2],[3,4,3,4]])

    def test_empty(self):
        a = array([[[]]])
        d = tile(a,(3,2,5)).shape
        assert_equal(d,(3,2,0))

    def test_kroncompare(self):
        import numpy.random as nr
        reps=[(2,),(1,2),(2,1),(2,2),(2,3,2),(3,2)]
        shape=[(3,),(2,3),(3,4,3),(3,2,3),(4,3,2,4),(2,2)]
        for s in shape:
            b = nr.randint(0,10,size=s)
            for r in reps:
                a = ones(r, b.dtype)
                large = tile(b, r)
                klarge = kron(a, b)
                assert_equal(large, klarge)


class TestMayShareMemory(TestCase):
    def test_basic(self):
        d = ones((50, 60))
        d2 = ones((30, 60, 6))
        self.assertTrue(may_share_memory(d, d))
        self.assertTrue(may_share_memory(d, d[::-1]))
        self.assertTrue(may_share_memory(d, d[::2]))
        self.assertTrue(may_share_memory(d, d[1:, ::-1]))

        self.assertFalse(may_share_memory(d[::-1], d2))
        self.assertFalse(may_share_memory(d[::2], d2))
        self.assertFalse(may_share_memory(d[1:, ::-1], d2))
        self.assertTrue(may_share_memory(d2[1:, ::-1], d2))


# Utility
def compare_results(res,desired):
    for i in range(len(desired)):
        assert_array_equal(res[i],desired[i])


if __name__ == "__main__":
    run_module_suite()
