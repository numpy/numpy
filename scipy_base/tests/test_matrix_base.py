""" Test functions for basic module

"""

import unittest

import sys
from scipy_test.testing import *
set_package_path()
import scipy_base;reload(scipy_base)
from scipy_base import *
del sys.path[0]

##################################################

val = limits.double_resolution


def get_mat(n):
    data = arange(n)
    data = add.outer(data,data)
    return data

class test_eye(unittest.TestCase):
    def check_basic(self):
        assert_equal(eye(4),array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0],
                                   [0,0,0,1]]))
        assert_equal(eye(4,typecode='f'),array([[1,0,0,0],
                                                [0,1,0,0],
                                                [0,0,1,0],
                                                [0,0,0,1]],'f'))
    def check_diag(self):
        assert_equal(eye(4,k=1),array([[0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1],
                                       [0,0,0,0]]))
        assert_equal(eye(4,k=-1),array([[0,0,0,0],
                                        [1,0,0,0],
                                        [0,1,0,0],
                                        [0,0,1,0]]))
    def check_2d(self):
        assert_equal(eye(4,3),array([[1,0,0],
                                     [0,1,0],
                                     [0,0,1],
                                     [0,0,0]]))
        assert_equal(eye(3,4),array([[1,0,0,0],
                                     [0,1,0,0],
                                     [0,0,1,0]]))        
    def check_diag2d(self):
        assert_equal(eye(3,4,k=2),array([[0,0,1,0],
                                         [0,0,0,1],
                                         [0,0,0,0]]))
        assert_equal(eye(4,3,k=-2),array([[0,0,0],
                                          [0,0,0],
                                          [1,0,0],
                                          [0,1,0]]))

class test_diag(unittest.TestCase):
    def check_vector(self):
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

    def check_matrix(self):
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

class test_fliplr(unittest.TestCase):
    def check_basic(self):
        self.failUnlessRaises(ValueError, fliplr, ones(4))        
        self.failUnlessRaises(ValueError, fliplr, ones((4,3,2)))
        a = get_mat(4)
        b = a[:,::-1]
        assert_equal(fliplr(a),b)
        a = [[0,1,2],
             [3,4,5]]
        b = [[2,1,0],
             [5,4,3]]
        assert_equal(fliplr(a),b)

class test_flipud(unittest.TestCase):
    def check_basic(self):
        self.failUnlessRaises(ValueError, flipud, ones(4))
        self.failUnlessRaises(ValueError, flipud, ones((4,3,2)))
        a = get_mat(4)
        b = a[::-1,:]
        assert_equal(flipud(a),b)
        a = [[0,1,2],
             [3,4,5]]
        b = [[3,4,5],
             [0,1,2]]
        assert_equal(flipud(a),b)

class test_rot90(unittest.TestCase):
    def check_basic(self):
        self.failUnlessRaises(ValueError, rot90, ones(4))
        self.failUnlessRaises(ValueError, rot90, ones((4,3,2)))

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

if __name__ == "__main__":
    ScipyTest('scipy_base.matrix_base').run()
