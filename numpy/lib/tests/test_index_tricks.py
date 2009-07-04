from numpy.testing import *
from numpy import ( array, ones, r_, mgrid, unravel_index, zeros, where,
                    ndenumerate, fill_diagonal, diag_indices,
                    diag_indices_from )

class TestUnravelIndex(TestCase):
    def test_basic(self):
        assert unravel_index(2,(2,2)) == (1,0)
        assert unravel_index(254,(17,94)) == (2, 66)
        assert_raises(ValueError, unravel_index, 4,(2,2))


class TestGrid(TestCase):
    def test_basic(self):
        a = mgrid[-1:1:10j]
        b = mgrid[-1:1:0.1]
        assert(a.shape == (10,))
        assert(b.shape == (20,))
        assert(a[0] == -1)
        assert_almost_equal(a[-1],1)
        assert(b[0] == -1)
        assert_almost_equal(b[1]-b[0],0.1,11)
        assert_almost_equal(b[-1],b[0]+19*0.1,11)
        assert_almost_equal(a[1]-a[0],2.0/9.0,11)

    def test_nd(self):
        c = mgrid[-1:1:10j,-2:2:10j]
        d = mgrid[-1:1:0.1,-2:2:0.2]
        assert(c.shape == (2,10,10))
        assert(d.shape == (2,20,20))
        assert_array_equal(c[0][0,:],-ones(10,'d'))
        assert_array_equal(c[1][:,0],-2*ones(10,'d'))
        assert_array_almost_equal(c[0][-1,:],ones(10,'d'),11)
        assert_array_almost_equal(c[1][:,-1],2*ones(10,'d'),11)
        assert_array_almost_equal(d[0,1,:]-d[0,0,:], 0.1*ones(20,'d'),11)
        assert_array_almost_equal(d[1,:,1]-d[1,:,0], 0.2*ones(20,'d'),11)


class TestConcatenator(TestCase):
    def test_1d(self):
        assert_array_equal(r_[1,2,3,4,5,6],array([1,2,3,4,5,6]))
        b = ones(5)
        c = r_[b,0,0,b]
        assert_array_equal(c,[1,1,1,1,1,0,0,1,1,1,1,1])

    def test_mixed_type(self):
        g = r_[10.1, 1:10]
        assert(g.dtype == 'f8')

    def test_more_mixed_type(self):
        g = r_[-10.1, array([1]), array([2,3,4]), 10.0]
        assert(g.dtype == 'f8')

    def test_2d(self):
        b = rand(5,5)
        c = rand(5,5)
        d = r_['1',b,c]  # append columns
        assert(d.shape == (5,10))
        assert_array_equal(d[:,:5],b)
        assert_array_equal(d[:,5:],c)
        d = r_[b,c]
        assert(d.shape == (10,5))
        assert_array_equal(d[:5,:],b)
        assert_array_equal(d[5:,:],c)


class TestNdenumerate(TestCase):
    def test_basic(self):
        a = array([[1,2], [3,4]])
        assert_equal(list(ndenumerate(a)),
                     [((0,0), 1), ((0,1), 2), ((1,0), 3), ((1,1), 4)])


def test_fill_diagonal():
    a = zeros((3, 3),int)
    fill_diagonal(a, 5)
    yield (assert_array_equal, a,
           array([[5, 0, 0],
                  [0, 5, 0],
                  [0, 0, 5]]))

    # The same function can operate on a 4-d array:
    a = zeros((3, 3, 3, 3), int)
    fill_diagonal(a, 4)
    i = array([0, 1, 2])
    yield (assert_equal, where(a != 0), (i, i, i, i))


def test_diag_indices():
    di = diag_indices(4)
    a = array([[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
               [13, 14, 15, 16]])
    a[di] = 100
    yield (assert_array_equal, a,
           array([[100,   2,   3,   4],
                  [  5, 100,   7,   8],
                  [  9,  10, 100,  12],
                  [ 13,  14,  15, 100]]))

    # Now, we create indices to manipulate a 3-d array:
    d3 = diag_indices(2, 3)

    # And use it to set the diagonal of a zeros array to 1:
    a = zeros((2, 2, 2),int)
    a[d3] = 1
    yield (assert_array_equal, a,
           array([[[1, 0],
                   [0, 0]],

                  [[0, 0],
                   [0, 1]]]) )


if __name__ == "__main__":
    run_module_suite()
