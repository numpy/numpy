import unittest
import sys

from scipy.test.testing import *
set_package_path()
import scipy.base;reload(scipy.base)
from scipy.base import *
del sys.path[0]

class test_ctor(ScipyTestCase):
    def test_basic(self):
        A = array([[1,2],[3,4]])
        mA = matrix(A)
        assert all(mA.A == A)

        B = bmat("A,A;A,A")
        C = bmat([[A,A], [A,A]])
        D = array([[1,2,1,2],
                   [3,4,3,4],
                   [1,2,1,2],
                   [3,4,3,4]])
        assert all(B.A == D)
        assert all(C.A == D)
        
        vec = arange(5)
        mvec = matrix(vec)
        assert mvec.shape == (1,5)
        
class test_properties(ScipyTestCase):
    def test_basic(self):
        from scipy import linalg
        
        A = array([[1., 2.], 
                   [3., 4.]])
        mA = matrix(A)
        assert allclose(linalg.inv(A), mA.I)
        assert all(array(transpose(A) == mA.T))
        assert all(array(transpose(A) == mA.H))
        assert all(A == mA.A)
        
        B = A + 2j*A
        mB = matrix(B)
        assert allclose(linalg.inv(B), mB.I)
        assert all(array(transpose(B) == mB.T))
        assert all(array(conjugate(transpose(B)), mB.H))

class test_algebra(ScipyTestCase):
    def test_basic(self):
        from scipy import linalg
        
        A = array([[1., 2.],
                   [3., 4.]])
        mA = matrix(A)

        B = identity(2)
        for i in xrange(6):
            assert allclose((mA ** i).A, B)
            B = dot(B, A)
        
        Ainv = linalg.inv(A)
        B = identity(2)
        for i in xrange(6):
            assert allclose((mA ** -i).A, B)
            B = dot(B, Ainv)

        assert allclose((mA * mA).A, dot(A, A))
        assert allclose((mA + mA).A, (A + A))        
        assert allclose((3*mA).A, (3*A))

