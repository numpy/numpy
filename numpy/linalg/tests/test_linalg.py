""" Test functions for linalg module
"""

from numpy.testing import *
set_package_path()
from numpy import array, single, double, csingle, cdouble, dot, identity, \
        multiply, atleast_2d
from numpy import linalg
from linalg import matrix_power
restore_path()

old_assert_almost_equal = assert_almost_equal
def assert_almost_equal(a, b, **kw):
    if a.dtype.type in (single, csingle):
        decimal = 6
    else:
        decimal = 12
    old_assert_almost_equal(a, b, decimal=decimal, **kw)

class LinalgTestCase(NumpyTestCase):
    def check_single(self):
        a = array([[1.,2.], [3.,4.]], dtype=single)
        b = array([2., 1.], dtype=single)
        self.do(a, b)

    def check_double(self):
        a = array([[1.,2.], [3.,4.]], dtype=double)
        b = array([2., 1.], dtype=double)
        self.do(a, b)

    def check_csingle(self):
        a = array([[1.+2j,2+3j], [3+4j,4+5j]], dtype=csingle)
        b = array([2.+1j, 1.+2j], dtype=csingle)
        self.do(a, b)

    def check_cdouble(self):
        a = array([[1.+2j,2+3j], [3+4j,4+5j]], dtype=cdouble)
        b = array([2.+1j, 1.+2j], dtype=cdouble)
        self.do(a, b)

    def check_empty(self):
        a = atleast_2d(array([], dtype = double))
        b = atleast_2d(array([], dtype = double))
        try:
            self.do(a, b)
            raise AssertionError("%s should fail with empty matrices", self.__name__[5:])
        except linalg.LinAlgError, e:
            pass

class TestSolve(LinalgTestCase):
    def do(self, a, b):
        x = linalg.solve(a, b)
        assert_almost_equal(b, dot(a, x))

class TestInv(LinalgTestCase):
    def do(self, a, b):
        a_inv = linalg.inv(a)
        assert_almost_equal(dot(a, a_inv), identity(a.shape[0]))

class TestEigvals(LinalgTestCase):
    def do(self, a, b):
        ev = linalg.eigvals(a)
        evalues, evectors = linalg.eig(a)
        assert_almost_equal(ev, evalues)

class TestEig(LinalgTestCase):
    def do(self, a, b):
        evalues, evectors = linalg.eig(a)
        assert_almost_equal(dot(a, evectors), evectors*evalues)

class TestSVD(LinalgTestCase):
    def do(self, a, b):
        u, s, vt = linalg.svd(a, 0)
        assert_almost_equal(a, dot(u*s, vt))

class TestPinv(LinalgTestCase):
    def do(self, a, b):
        a_ginv = linalg.pinv(a)
        assert_almost_equal(dot(a, a_ginv), identity(a.shape[0]))

class TestDet(LinalgTestCase):
    def do(self, a, b):
        d = linalg.det(a)
        if a.dtype.type in (single, double):
            ad = a.astype(double)
        else:
            ad = a.astype(cdouble)
        ev = linalg.eigvals(ad)
        assert_almost_equal(d, multiply.reduce(ev))

class TestLstsq(LinalgTestCase):
    def do(self, a, b):
        u, s, vt = linalg.svd(a, 0)
        x, residuals, rank, sv = linalg.lstsq(a, b)
        assert_almost_equal(b, dot(a, x))
        assert_equal(rank, a.shape[0])
        assert_almost_equal(sv, s)

class TestMatrixPower(ParametricTestCase):
    R90 = array([[0,1],[-1,0]])
    Arb22 = array([[4,-7],[-2,10]])
    noninv = array([[1,0],[0,0]])
    arbfloat = array([[0.1,3.2],[1.2,0.7]])

    large = identity(10)
    t = large[1,:].copy()
    large[1,:] = large[0,:]
    large[0,:] = t

    def test_large_power(self):
        assert_equal(matrix_power(self.R90,2L**100+2**10+2**5+1),self.R90)
    def test_large_power_trailing_zero(self):
        assert_equal(matrix_power(self.R90,2L**100+2**10+2**5),identity(2))

    def testip_zero(self):
        def tz(M):
            mz = matrix_power(M,0)
            assert_equal(mz, identity(M.shape[0]))
            assert_equal(mz.dtype, M.dtype)
        for M in [self.Arb22, self.arbfloat, self.large]:
            yield tz, M

    def testip_one(self):
        def tz(M):
            mz = matrix_power(M,1)
            assert_equal(mz, M)
            assert_equal(mz.dtype, M.dtype)
        for M in [self.Arb22, self.arbfloat, self.large]:
            yield tz, M

    def testip_two(self):
        def tz(M):
            mz = matrix_power(M,2)
            assert_equal(mz, dot(M,M))
            assert_equal(mz.dtype, M.dtype)
        for M in [self.Arb22, self.arbfloat, self.large]:
            yield tz, M

    def testip_invert(self):
        def tz(M):
            mz = matrix_power(M,-1)
            assert_almost_equal(identity(M.shape[0]), dot(mz,M))
        for M in [self.R90, self.Arb22, self.arbfloat, self.large]:
            yield tz, M

    def test_invert_noninvertible(self):
        import numpy.linalg
        self.assertRaises(numpy.linalg.linalg.LinAlgError,
                lambda: matrix_power(self.noninv,-1))


if __name__ == '__main__':
    NumpyTest().run()
