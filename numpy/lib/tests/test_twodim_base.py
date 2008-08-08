""" Test functions for matrix module

"""

from numpy.testing import *
from numpy import arange, rot90, add, fliplr, flipud, zeros, ones, eye, \
     array, diag, histogram2d, tri
import numpy as np

def get_mat(n):
    data = arange(n)
    data = add.outer(data,data)
    return data

class TestEye(TestCase):
    def test_basic(self):
        assert_equal(eye(4),array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]]))
        assert_equal(eye(4,dtype='f'),array([[1,0,0,0],
                                                [0,1,0,0],
                                                [0,0,1,0],
                                                [0,0,0,1]],'f'))
        assert_equal(eye(3) == 1, eye(3,dtype=bool))

    def test_diag(self):
        assert_equal(eye(4,k=1),array([[0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1],
                                       [0,0,0,0]]))
        assert_equal(eye(4,k=-1),array([[0,0,0,0],
                                        [1,0,0,0],
                                        [0,1,0,0],
                                        [0,0,1,0]]))
    def test_2d(self):
        assert_equal(eye(4,3),array([[1,0,0],
                                     [0,1,0],
                                     [0,0,1],
                                     [0,0,0]]))
        assert_equal(eye(3,4),array([[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,1,0]]))
    def test_diag2d(self):
        assert_equal(eye(3,4,k=2),array([[0,0,1,0],
                                         [0,0,0,1],
                                         [0,0,0,0]]))
        assert_equal(eye(4,3,k=-2),array([[0,0,0],
                                          [0,0,0],
                                          [1,0,0],
                                          [0,1,0]]))

class TestDiag(TestCase):
    def test_vector(self):
        vals = (100*arange(5)).astype('l')
        b = zeros((5,5))
        for k in range(5):
            b[k,k] = vals[k]
        assert_equal(diag(vals),b)
        b = zeros((7,7))
        c = b.copy()
        for k in range(5):
            b[k,k+2] = vals[k]
            c[k+2,k] = vals[k]
        assert_equal(diag(vals,k=2), b)
        assert_equal(diag(vals,k=-2), c)

    def test_matrix(self):
        vals = (100*get_mat(5)+1).astype('l')
        b = zeros((5,))
        for k in range(5):
            b[k] = vals[k,k]
        assert_equal(diag(vals),b)
        b = b*0
        for k in range(3):
            b[k] = vals[k,k+2]
        assert_equal(diag(vals,2),b[:3])
        for k in range(3):
            b[k] = vals[k+2,k]
        assert_equal(diag(vals,-2),b[:3])

class TestFliplr(TestCase):
    def test_basic(self):
        self.failUnlessRaises(ValueError, fliplr, ones(4))
        a = get_mat(4)
        b = a[:,::-1]
        assert_equal(fliplr(a),b)
        a = [[0,1,2],
             [3,4,5]]
        b = [[2,1,0],
             [5,4,3]]
        assert_equal(fliplr(a),b)

class TestFlipud(TestCase):
    def test_basic(self):
        a = get_mat(4)
        b = a[::-1,:]
        assert_equal(flipud(a),b)
        a = [[0,1,2],
             [3,4,5]]
        b = [[3,4,5],
             [0,1,2]]
        assert_equal(flipud(a),b)

class TestRot90(TestCase):
    def test_basic(self):
        self.failUnlessRaises(ValueError, rot90, ones(4))

        a = [[0,1,2],
             [3,4,5]]
        b1 = [[2,5],
              [1,4],
              [0,3]]
        b2 = [[5,4,3],
              [2,1,0]]
        b3 = [[3,0],
              [4,1],
              [5,2]]
        b4 = [[0,1,2],
              [3,4,5]]

        for k in range(-3,13,4):
            assert_equal(rot90(a,k=k),b1)
        for k in range(-2,13,4):
            assert_equal(rot90(a,k=k),b2)
        for k in range(-1,13,4):
            assert_equal(rot90(a,k=k),b3)
        for k in range(0,13,4):
            assert_equal(rot90(a,k=k),b4)

    def test_axes(self):
        a = ones((50,40,3))
        assert_equal(rot90(a).shape,(40,50,3))

class TestHistogram2d(TestCase):
    def test_simple(self):
        x = array([ 0.41702200,  0.72032449,  0.00011437481, 0.302332573,  0.146755891])
        y = array([ 0.09233859,  0.18626021,  0.34556073,  0.39676747,  0.53881673])
        xedges = np.linspace(0,1,10)
        yedges = np.linspace(0,1,10)
        H = histogram2d(x, y, (xedges, yedges))[0]
        answer = array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        assert_array_equal(H.T, answer)
        H = histogram2d(x, y, xedges)[0]
        assert_array_equal(H.T, answer)
        H,xedges,yedges = histogram2d(range(10),range(10))
        assert_array_equal(H, eye(10,10))
        assert_array_equal(xedges, np.linspace(0,9,11))
        assert_array_equal(yedges, np.linspace(0,9,11))

    def test_asym(self):
        x = array([1, 1, 2, 3, 4, 4, 4, 5])
        y = array([1, 3, 2, 0, 1, 2, 3, 4])
        H, xed, yed = histogram2d(x,y, (6, 5), range = [[0,6],[0,5]], normed=True)
        answer = array([[0.,0,0,0,0],
                        [0,1,0,1,0],
                        [0,0,1,0,0],
                        [1,0,0,0,0],
                        [0,1,1,1,0],
                        [0,0,0,0,1]])
        assert_array_almost_equal(H, answer/8., 3)
        assert_array_equal(xed, np.linspace(0,6,7))
        assert_array_equal(yed, np.linspace(0,5,6))
    def test_norm(self):
        x = array([1,2,3,1,2,3,1,2,3])
        y = array([1,1,1,2,2,2,3,3,3])
        H, xed, yed = histogram2d(x,y,[[1,2,3,5], [1,2,3,5]], normed=True)
        answer=array([[1,1,.5],
                     [1,1,.5],
                     [.5,.5,.25]])/9.
        assert_array_almost_equal(H, answer, 3)

    def test_all_outliers(self):
        r = rand(100)+1.
        H, xed, yed = histogram2d(r, r, (4, 5), range=([0,1], [0,1]))
        assert_array_equal(H, 0)


class TestTri(TestCase):
    def test_dtype(self):
        out = array([[1,0,0],
                     [1,1,0],
                     [1,1,1]])
        assert_array_equal(tri(3),out)
        assert_array_equal(tri(3,dtype=bool),out.astype(bool))


if __name__ == "__main__":
    run_module_suite()
