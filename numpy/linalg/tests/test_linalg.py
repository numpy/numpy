""" Test functions for linalg module
"""
import sys

import numpy as np
from numpy.testing import (TestCase, assert_, assert_equal, assert_raises,
                           assert_array_equal, assert_almost_equal,
                           run_module_suite)
from numpy import array, single, double, csingle, cdouble, dot, identity
from numpy import multiply, atleast_2d, inf, asarray, matrix
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank

def ifthen(a, b):
    return not a or b

old_assert_almost_equal = assert_almost_equal
def imply(a, b):
    return not a or b

def assert_almost_equal(a, b, **kw):
    if asarray(a).dtype.type in (single, csingle):
        decimal = 6
    else:
        decimal = 12
    old_assert_almost_equal(a, b, decimal=decimal, **kw)


class LinalgTestCase(object):
    def test_single(self):
        a = array([[1.,2.], [3.,4.]], dtype=single)
        b = array([2., 1.], dtype=single)
        self.do(a, b)

    def test_double(self):
        a = array([[1.,2.], [3.,4.]], dtype=double)
        b = array([2., 1.], dtype=double)
        self.do(a, b)

    def test_double_2(self):
        a = array([[1.,2.], [3.,4.]], dtype=double)
        b = array([[2., 1., 4.], [3., 4., 6.]], dtype=double)
        self.do(a, b)

    def test_csingle(self):
        a = array([[1.+2j,2+3j], [3+4j,4+5j]], dtype=csingle)
        b = array([2.+1j, 1.+2j], dtype=csingle)
        self.do(a, b)

    def test_cdouble(self):
        a = array([[1.+2j,2+3j], [3+4j,4+5j]], dtype=cdouble)
        b = array([2.+1j, 1.+2j], dtype=cdouble)
        self.do(a, b)

    def test_cdouble_2(self):
        a = array([[1.+2j,2+3j], [3+4j,4+5j]], dtype=cdouble)
        b = array([[2.+1j, 1.+2j, 1+3j], [1-2j, 1-3j, 1-6j]], dtype=cdouble)
        self.do(a, b)

    def test_empty(self):
        a = atleast_2d(array([], dtype = double))
        b = atleast_2d(array([], dtype = double))
        try:
            self.do(a, b)
            raise AssertionError("%s should fail with empty matrices", self.__name__[5:])
        except linalg.LinAlgError, e:
            pass

    def test_nonarray(self):
        a = [[1,2], [3,4]]
        b = [2, 1]
        self.do(a,b)

    def test_matrix_b_only(self):
        """Check that matrix type is preserved."""
        a = array([[1.,2.], [3.,4.]])
        b = matrix([2., 1.]).T
        self.do(a, b)

    def test_matrix_a_and_b(self):
        """Check that matrix type is preserved."""
        a = matrix([[1.,2.], [3.,4.]])
        b = matrix([2., 1.]).T
        self.do(a, b)


class LinalgNonsquareTestCase(object):
    def test_single_nsq_1(self):
        a = array([[1.,2.,3.], [3.,4.,6.]], dtype=single)
        b = array([2., 1.], dtype=single)
        self.do(a, b)

    def test_single_nsq_2(self):
        a = array([[1.,2.], [3.,4.], [5.,6.]], dtype=single)
        b = array([2., 1., 3.], dtype=single)
        self.do(a, b)

    def test_double_nsq_1(self):
        a = array([[1.,2.,3.], [3.,4.,6.]], dtype=double)
        b = array([2., 1.], dtype=double)
        self.do(a, b)

    def test_double_nsq_2(self):
        a = array([[1.,2.], [3.,4.], [5.,6.]], dtype=double)
        b = array([2., 1., 3.], dtype=double)
        self.do(a, b)

    def test_csingle_nsq_1(self):
        a = array([[1.+1j,2.+2j,3.-3j], [3.-5j,4.+9j,6.+2j]], dtype=csingle)
        b = array([2.+1j, 1.+2j], dtype=csingle)
        self.do(a, b)

    def test_csingle_nsq_2(self):
        a = array([[1.+1j,2.+2j], [3.-3j,4.-9j], [5.-4j,6.+8j]], dtype=csingle)
        b = array([2.+1j, 1.+2j, 3.-3j], dtype=csingle)
        self.do(a, b)

    def test_cdouble_nsq_1(self):
        a = array([[1.+1j,2.+2j,3.-3j], [3.-5j,4.+9j,6.+2j]], dtype=cdouble)
        b = array([2.+1j, 1.+2j], dtype=cdouble)
        self.do(a, b)

    def test_cdouble_nsq_2(self):
        a = array([[1.+1j,2.+2j], [3.-3j,4.-9j], [5.-4j,6.+8j]], dtype=cdouble)
        b = array([2.+1j, 1.+2j, 3.-3j], dtype=cdouble)
        self.do(a, b)

    def test_cdouble_nsq_1_2(self):
        a = array([[1.+1j,2.+2j,3.-3j], [3.-5j,4.+9j,6.+2j]], dtype=cdouble)
        b = array([[2.+1j, 1.+2j], [1-1j, 2-2j]], dtype=cdouble)
        self.do(a, b)

    def test_cdouble_nsq_2_2(self):
        a = array([[1.+1j,2.+2j], [3.-3j,4.-9j], [5.-4j,6.+8j]], dtype=cdouble)
        b = array([[2.+1j, 1.+2j], [1-1j, 2-2j], [1-1j, 2-2j]], dtype=cdouble)
        self.do(a, b)


class TestSolve(LinalgTestCase, TestCase):
    def do(self, a, b):
        x = linalg.solve(a, b)
        assert_almost_equal(b, dot(a, x))
        assert_(imply(isinstance(b, matrix), isinstance(x, matrix)))


class TestInv(LinalgTestCase, TestCase):
    def do(self, a, b):
        a_inv = linalg.inv(a)
        assert_almost_equal(dot(a, a_inv), identity(asarray(a).shape[0]))
        assert_(imply(isinstance(a, matrix), isinstance(a_inv, matrix)))


class TestEigvals(LinalgTestCase, TestCase):
    def do(self, a, b):
        ev = linalg.eigvals(a)
        evalues, evectors = linalg.eig(a)
        assert_almost_equal(ev, evalues)


class TestEig(LinalgTestCase, TestCase):
    def do(self, a, b):
        evalues, evectors = linalg.eig(a)
        assert_almost_equal(dot(a, evectors), multiply(evectors, evalues))
        assert_(imply(isinstance(a, matrix), isinstance(evectors, matrix)))


class TestSVD(LinalgTestCase, TestCase):
    def do(self, a, b):
        u, s, vt = linalg.svd(a, 0)
        assert_almost_equal(a, dot(multiply(u, s), vt))
        assert_(imply(isinstance(a, matrix), isinstance(u, matrix)))
        assert_(imply(isinstance(a, matrix), isinstance(vt, matrix)))


class TestCondSVD(LinalgTestCase, TestCase):
    def do(self, a, b):
        c = asarray(a) # a might be a matrix
        s = linalg.svd(c, compute_uv=False)
        old_assert_almost_equal(s[0]/s[-1], linalg.cond(a), decimal=5)


class TestCond2(LinalgTestCase, TestCase):
    def do(self, a, b):
        c = asarray(a) # a might be a matrix
        s = linalg.svd(c, compute_uv=False)
        old_assert_almost_equal(s[0]/s[-1], linalg.cond(a,2), decimal=5)


class TestCondInf(TestCase):
    def test(self):
        A = array([[1.,0,0],[0,-2.,0],[0,0,3.]])
        assert_almost_equal(linalg.cond(A,inf),3.)


class TestPinv(LinalgTestCase, TestCase):
    def do(self, a, b):
        a_ginv = linalg.pinv(a)
        assert_almost_equal(dot(a, a_ginv), identity(asarray(a).shape[0]))
        assert_(imply(isinstance(a, matrix), isinstance(a_ginv, matrix)))


class TestDet(LinalgTestCase, TestCase):
    def do(self, a, b):
        d = linalg.det(a)
        (s, ld) = linalg.slogdet(a)
        if asarray(a).dtype.type in (single, double):
            ad = asarray(a).astype(double)
        else:
            ad = asarray(a).astype(cdouble)
        ev = linalg.eigvals(ad)
        assert_almost_equal(d, multiply.reduce(ev))
        assert_almost_equal(s * np.exp(ld), multiply.reduce(ev))
        if s != 0:
            assert_almost_equal(np.abs(s), 1)
        else:
            assert_equal(ld, -inf)

    def test_zero(self):
        assert_equal(linalg.det([[0.0]]), 0.0)
        assert_equal(type(linalg.det([[0.0]])), double)
        assert_equal(linalg.det([[0.0j]]), 0.0)
        assert_equal(type(linalg.det([[0.0j]])), cdouble)

        assert_equal(linalg.slogdet([[0.0]]), (0.0, -inf))
        assert_equal(type(linalg.slogdet([[0.0]])[0]), double)
        assert_equal(type(linalg.slogdet([[0.0]])[1]), double)
        assert_equal(linalg.slogdet([[0.0j]]), (0.0j, -inf))
        assert_equal(type(linalg.slogdet([[0.0j]])[0]), cdouble)
        assert_equal(type(linalg.slogdet([[0.0j]])[1]), double)


class TestLstsq(LinalgTestCase, LinalgNonsquareTestCase, TestCase):
    def do(self, a, b):
        arr = np.asarray(a)
        m, n = arr.shape
        u, s, vt = linalg.svd(a, 0)
        x, residuals, rank, sv = linalg.lstsq(a, b)
        if m <= n:
            assert_almost_equal(b, dot(a, x))
            assert_equal(rank, m)
        else:
            assert_equal(rank, n)
        assert_almost_equal(sv, sv.__array_wrap__(s))
        if rank == n and m > n:
            expect_resids = (np.asarray(abs(np.dot(a, x) - b))**2).sum(axis=0)
            expect_resids = np.asarray(expect_resids)
            if len(np.asarray(b).shape) == 1:
                expect_resids.shape = (1,)
                assert_equal(residuals.shape, expect_resids.shape)
        else:
            expect_resids = type(x)([])
        assert_almost_equal(residuals, expect_resids)
        assert_(np.issubdtype(residuals.dtype, np.floating))
        assert_(imply(isinstance(b, matrix), isinstance(x, matrix)))
        assert_(imply(isinstance(b, matrix), isinstance(residuals, matrix)))


class TestMatrixPower(object):
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
        assert_raises(numpy.linalg.linalg.LinAlgError,
                      lambda: matrix_power(self.noninv,-1))


class TestBoolPower(TestCase):
    def test_square(self):
        A = array([[True,False],[True,True]])
        assert_equal(matrix_power(A,2),A)


class HermitianTestCase(object):
    def test_single(self):
        a = array([[1.,2.], [2.,1.]], dtype=single)
        self.do(a)

    def test_double(self):
        a = array([[1.,2.], [2.,1.]], dtype=double)
        self.do(a)

    def test_csingle(self):
        a = array([[1.,2+3j], [2-3j,1]], dtype=csingle)
        self.do(a)

    def test_cdouble(self):
        a = array([[1.,2+3j], [2-3j,1]], dtype=cdouble)
        self.do(a)

    def test_empty(self):
        a = atleast_2d(array([], dtype = double))
        assert_raises(linalg.LinAlgError, self.do, a)

    def test_nonarray(self):
        a = [[1,2], [2,1]]
        self.do(a)

    def test_matrix_b_only(self):
        """Check that matrix type is preserved."""
        a = array([[1.,2.], [2.,1.]])
        self.do(a)

    def test_matrix_a_and_b(self):
        """Check that matrix type is preserved."""
        a = matrix([[1.,2.], [2.,1.]])
        self.do(a)


class TestEigvalsh(HermitianTestCase, TestCase):
    def do(self, a):
        # note that eigenvalue arrays must be sorted since
        # their order isn't guaranteed.
        ev = linalg.eigvalsh(a)
        evalues, evectors = linalg.eig(a)
        ev.sort()
        evalues.sort()
        assert_almost_equal(ev, evalues)


class TestEigh(HermitianTestCase, TestCase):
    def do(self, a):
        # note that eigenvalue arrays must be sorted since
        # their order isn't guaranteed.
        ev, evc = linalg.eigh(a)
        evalues, evectors = linalg.eig(a)
        ev.sort()
        evalues.sort()
        assert_almost_equal(ev, evalues)


class _TestNorm(TestCase):
    dt = None
    dec = None
    def test_empty(self):
        assert_equal(norm([]), 0.0)
        assert_equal(norm(array([], dtype=self.dt)), 0.0)
        assert_equal(norm(atleast_2d(array([], dtype=self.dt))), 0.0)

    def test_vector(self):
        a = [1.0,2.0,3.0,4.0]
        b = [-1.0,-2.0,-3.0,-4.0]
        c = [-1.0, 2.0,-3.0, 4.0]

        def _test(v):
            np.testing.assert_almost_equal(norm(v), 30**0.5, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v,inf), 4.0, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v,-inf), 1.0, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v,1), 10.0, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v,-1), 12.0/25,
                    decimal=self.dec)
            np.testing.assert_almost_equal(norm(v,2), 30**0.5,
                    decimal=self.dec)
            np.testing.assert_almost_equal(norm(v,-2), ((205./144)**-0.5),
                    decimal=self.dec)
            np.testing.assert_almost_equal(norm(v,0), 4, decimal=self.dec)

        for v in (a, b, c,):
            _test(v)

        for v in (array(a, dtype=self.dt), array(b, dtype=self.dt),
                  array(c, dtype=self.dt)):
            _test(v)

    def test_matrix(self):
        A = matrix([[1.,3.],[5.,7.]], dtype=self.dt)
        A = matrix([[1.,3.],[5.,7.]], dtype=self.dt)
        assert_almost_equal(norm(A), 84**0.5)
        assert_almost_equal(norm(A,'fro'), 84**0.5)
        assert_almost_equal(norm(A,inf), 12.0)
        assert_almost_equal(norm(A,-inf), 4.0)
        assert_almost_equal(norm(A,1), 10.0)
        assert_almost_equal(norm(A,-1), 6.0)
        assert_almost_equal(norm(A,2), 9.1231056256176615)
        assert_almost_equal(norm(A,-2), 0.87689437438234041)

        self.assertRaises(ValueError, norm, A, 'nofro')
        self.assertRaises(ValueError, norm, A, -3)
        self.assertRaises(ValueError, norm, A, 0)


class TestNormDouble(_TestNorm):
    dt = np.double
    dec= 12


class TestNormSingle(_TestNorm):
    dt = np.float32
    dec = 6


class TestMatrixRank(object):
    def test_matrix_rank(self):
        # Full rank matrix
        yield assert_equal, 4, matrix_rank(np.eye(4))
        # rank deficient matrix
        I=np.eye(4); I[-1,-1] = 0.
        yield assert_equal, matrix_rank(I), 3
        # All zeros - zero rank
        yield assert_equal, matrix_rank(np.zeros((4,4))), 0
        # 1 dimension - rank 1 unless all 0
        yield assert_equal, matrix_rank([1, 0, 0, 0]), 1
        yield assert_equal, matrix_rank(np.zeros((4,))), 0
        # accepts array-like
        yield assert_equal, matrix_rank([1]), 1
        # greater than 2 dimensions raises error
        yield assert_raises, TypeError, matrix_rank, np.zeros((2,2,2))
        # works on scalar
        yield assert_equal, matrix_rank(1), 1


def test_reduced_rank():
    # Test matrices with reduced rank
    rng = np.random.RandomState(20120714)
    for i in range(100):
        # Make a rank deficient matrix
        X = rng.normal(size=(40, 10))
        X[:, 0] = X[:, 1] + X[:, 2]
        # Assert that matrix_rank detected deficiency
        assert_equal(matrix_rank(X), 9)
        X[:, 3] = X[:, 4] + X[:, 5]
        assert_equal(matrix_rank(X), 8)


class TestQR(TestCase):
    def test_qr_empty(self):
        a = np.zeros((0,2))
        self.assertRaises(linalg.LinAlgError, linalg.qr, a)


def test_byteorder_check():
    # Byte order check should pass for native order
    if sys.byteorder == 'little':
        native = '<'
    else:
        native = '>'

    for dtt in (np.float32, np.float64):
        arr = np.eye(4, dtype=dtt)
        n_arr = arr.newbyteorder(native)
        sw_arr = arr.newbyteorder('S').byteswap()
        assert_equal(arr.dtype.byteorder, '=')
        for routine in (linalg.inv, linalg.det, linalg.pinv):
            # Normal call
            res = routine(arr)
            # Native but not '='
            assert_array_equal(res, routine(n_arr))
            # Swapped
            assert_array_equal(res, routine(sw_arr))


if __name__ == "__main__":
    run_module_suite()
