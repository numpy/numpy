
from numpy.testing import *
set_package_path()
import numpy.core;reload(numpy.core)
from numpy.core import *
restore_path()

class test_ctor(ScipyTestCase):
    def check_basic(self):
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
    def check_basic(self):
        import numpy.linalg as linalg
        
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
        assert all(array(conjugate(transpose(B)) == mB.H))

    def check_comparisons(self):
        A = arange(100).reshape(10,10)
        mA = matrix(A)
        mB = matrix(A) + 0.1
        assert all(mB == A+0.1)
        assert all(mB == matrix(A+0.1))
        assert not any(mB == matrix(A-0.1))
        assert all(mA < mB)
        assert all(mA <= mB)
        assert all(mA <= mA)
        assert not any(mA < mA)
        
        assert not any(mB < mA)
        assert all(mB >= mA)
        assert all(mB >= mB)
        assert not any(mB > mB)
        
        assert all(mA == mA)
        assert not any(mA == mB)
        assert all(mB != mA)
        
        assert not all(abs(mA) > 0)
        assert all(abs(mB > 0))
    
    def check_asmatrix(self):
        A = arange(100).reshape(10,10)
        mA = asmatrix(A)
        mB = matrix(A)
        A[0,0] = -10
        assert A[0,0] == mA[0,0]
        assert A[0,0] != mB[0,0]

    def check_noaxis(self):
        A = matrix([[1,0],[0,1]])
	assert A.sum() == matrix(2)
	assert A.mean() == matrix(0.5)

class test_autocasting(ScipyTestCase):
    def check_basic(self):
        A = arange(100).reshape(10,10)
        mA = matrix(A)
        
        mB = mA.copy()
        O = ones((10,10), float64) * 0.1
        mB = mB + O
        assert mB.dtype.type == float64
        assert all(mA != mB)
        assert all(mB == mA+0.1)
        
        mC = mA.copy()
        O = ones((10,10), complex128)
        mC = mC * O
        assert mC.dtype.type == complex128
        assert all(mA != mB)

class test_algebra(ScipyTestCase):
    def check_basic(self):
        import numpy.linalg as linalg
        
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

if __name__ == "__main__":
    ScipyTest().run()
