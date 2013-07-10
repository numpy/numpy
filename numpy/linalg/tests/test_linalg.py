""" Test functions for linalg module
"""
from __future__ import division, absolute_import, print_function

import os
import sys

import numpy as np
from numpy.testing import (TestCase, assert_, assert_equal, assert_raises,
                           assert_array_equal, assert_almost_equal,
                           run_module_suite, dec)
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

def get_real_dtype(dtype):
    return {single: single, double: double,
            csingle: single, cdouble: double}[dtype]

def get_complex_dtype(dtype):
    return {single: csingle, double: cdouble,
            csingle: csingle, cdouble: cdouble}[dtype]


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
        except linalg.LinAlgError as e:
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


def _generalized_testcase(new_cls_name, old_cls):
    def get_method(old_name, new_name):
        def method(self):
            base = old_cls()
            def do(a, b):
                a = np.array([a, a, a])
                b = np.array([b, b, b])
                self.do(a, b)
            base.do = do
            getattr(base, old_name)()
        method.__name__ = new_name
        return method

    dct = dict()
    for old_name in dir(old_cls):
        if old_name.startswith('test_'):
            new_name = old_name + '_generalized'
            dct[new_name] = get_method(old_name, new_name)

    return type(new_cls_name, (object,), dct)

LinalgGeneralizedTestCase = _generalized_testcase(
    'LinalgGeneralizedTestCase', LinalgTestCase)
LinalgGeneralizedNonsquareTestCase = _generalized_testcase(
    'LinalgGeneralizedNonsquareTestCase', LinalgNonsquareTestCase)


def dot_generalized(a, b):
    a = asarray(a)
    if a.ndim == 3:
        return np.array([dot(ax, bx) for ax, bx in zip(a, b)])
    elif a.ndim > 3:
        raise ValueError("Not implemented...")
    return dot(a, b)

def identity_like_generalized(a):
    a = asarray(a)
    if a.ndim == 3:
        return np.array([identity(a.shape[-2]) for ax in a])
    elif a.ndim > 3:
        raise ValueError("Not implemented...")
    return identity(a.shape[0])


class TestSolve(LinalgTestCase, LinalgGeneralizedTestCase, TestCase):
    def do(self, a, b):
        x = linalg.solve(a, b)
        assert_almost_equal(b, dot_generalized(a, x))
        assert_(imply(isinstance(b, matrix), isinstance(x, matrix)))

    def test_types(self):
        def check(dtype):
            x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
            assert_equal(linalg.solve(x, x).dtype, dtype)
        for dtype in [single, double, csingle, cdouble]:
            yield check, dtype

    def test_0_size(self):
        class ArraySubclass(np.ndarray):
            pass
        # Test system of 0x0 matrices
        a = np.arange(8).reshape(2, 2, 2)
        b = np.arange(6).reshape(1, 2, 3).view(ArraySubclass)

        expected = linalg.solve(a, b)[:,0:0,:]
        result = linalg.solve(a[:,0:0,0:0], b[:,0:0,:])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        # Test errors for non-square and only b's dimension being 0
        assert_raises(linalg.LinAlgError, linalg.solve, a[:,0:0,0:1], b)
        assert_raises(ValueError, linalg.solve, a, b[:,0:0,:])

        # Test broadcasting error
        b = np.arange(6).reshape(1, 3, 2) # broadcasting error
        assert_raises(ValueError, linalg.solve, a, b)
        assert_raises(ValueError, linalg.solve, a[0:0], b[0:0])

        # Test zero "single equations" with 0x0 matrices.
        b = np.arange(2).reshape(1, 2).view(ArraySubclass)
        expected = linalg.solve(a, b)[:,0:0]
        result = linalg.solve(a[:,0:0,0:0], b[:,0:0])
        assert_array_equal(result, expected)
        assert_(isinstance(result, ArraySubclass))

        b = np.arange(3).reshape(1, 3)
        assert_raises(ValueError, linalg.solve, a, b)
        assert_raises(ValueError, linalg.solve, a[0:0], b[0:0])
        assert_raises(ValueError, linalg.solve, a[:,0:0,0:0], b)


class TestInv(LinalgTestCase, LinalgGeneralizedTestCase, TestCase):
    def do(self, a, b):
        a_inv = linalg.inv(a)
        assert_almost_equal(dot_generalized(a, a_inv),
                            identity_like_generalized(a))
        assert_(imply(isinstance(a, matrix), isinstance(a_inv, matrix)))

    def test_types(self):
        def check(dtype):
            x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
            assert_equal(linalg.inv(x).dtype, dtype)
        for dtype in [single, double, csingle, cdouble]:
            yield check, dtype

    def test_0_size(self):
        # Check that all kinds of 0-sized arrays work
        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0,1,1), dtype=np.int_).view(ArraySubclass)
        res = linalg.inv(a)
        assert_(res.dtype.type is np.float64)
        assert_equal(a.shape, res.shape)
        assert_(isinstance(a, ArraySubclass))

        a = np.zeros((0,0), dtype=np.complex64).view(ArraySubclass)
        res = linalg.inv(a)
        assert_(res.dtype.type is np.complex64)
        assert_equal(a.shape, res.shape)


class TestEigvals(LinalgTestCase, LinalgGeneralizedTestCase, TestCase):
    def do(self, a, b):
        ev = linalg.eigvals(a)
        evalues, evectors = linalg.eig(a)
        assert_almost_equal(ev, evalues)

    def test_types(self):
        def check(dtype):
            x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
            assert_equal(linalg.eigvals(x).dtype, dtype)
            x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
            assert_equal(linalg.eigvals(x).dtype, get_complex_dtype(dtype))
        for dtype in [single, double, csingle, cdouble]:
            yield check, dtype


class TestEig(LinalgTestCase, LinalgGeneralizedTestCase, TestCase):
    def do(self, a, b):
        evalues, evectors = linalg.eig(a)
        if evectors.ndim == 3:
            assert_almost_equal(dot_generalized(a, evectors), evectors * evalues[:,None,:])
        else:
            assert_almost_equal(dot(a, evectors), multiply(evectors, evalues))
        assert_(imply(isinstance(a, matrix), isinstance(evectors, matrix)))

    def test_types(self):
        def check(dtype):
            x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
            w, v = np.linalg.eig(x)
            assert_equal(w.dtype, dtype)
            assert_equal(v.dtype, dtype)

            x = np.array([[1, 0.5], [-1, 1]], dtype=dtype)
            w, v = np.linalg.eig(x)
            assert_equal(w.dtype, get_complex_dtype(dtype))
            assert_equal(v.dtype, get_complex_dtype(dtype))

        for dtype in [single, double, csingle, cdouble]:
            yield dtype


class TestSVD(LinalgTestCase, LinalgGeneralizedTestCase, TestCase):
    def do(self, a, b):
        u, s, vt = linalg.svd(a, 0)
        if u.ndim == 3:
            assert_almost_equal(a, dot_generalized(u * s[:,None,:], vt))
        else:
            assert_almost_equal(a, dot(multiply(u, s), vt))
        assert_(imply(isinstance(a, matrix), isinstance(u, matrix)))
        assert_(imply(isinstance(a, matrix), isinstance(vt, matrix)))

    def test_types(self):
        def check(dtype):
            x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
            u, s, vh = linalg.svd(x)
            assert_equal(u.dtype, dtype)
            assert_equal(s.dtype, get_real_dtype(dtype))
            assert_equal(vh.dtype, dtype)
            s = linalg.svd(x, compute_uv=False)
            assert_equal(s.dtype, get_real_dtype(dtype))

        for dtype in [single, double, csingle, cdouble]:
            yield check, dtype


class TestCondSVD(LinalgTestCase, LinalgGeneralizedTestCase, TestCase):
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


class TestDet(LinalgTestCase, LinalgGeneralizedTestCase, TestCase):
    def do(self, a, b):
        d = linalg.det(a)
        (s, ld) = linalg.slogdet(a)
        if asarray(a).dtype.type in (single, double):
            ad = asarray(a).astype(double)
        else:
            ad = asarray(a).astype(cdouble)
        ev = linalg.eigvals(ad)
        assert_almost_equal(d, multiply.reduce(ev, axis=-1))
        assert_almost_equal(s * np.exp(ld), multiply.reduce(ev, axis=-1))

        s = np.atleast_1d(s)
        ld = np.atleast_1d(ld)
        m = (s != 0)
        assert_almost_equal(np.abs(s[m]), 1)
        assert_equal(ld[~m], -inf)

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

    def test_types(self):
        def check(dtype):
            x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
            assert_equal(np.linalg.det(x), get_real_dtype(dtype))
        for dtype in [single, double, csingle, cdouble]:
            yield check, dtype


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
        assert_equal(matrix_power(self.R90,2**100+2**10+2**5+1),self.R90)

    def test_large_power_trailing_zero(self):
        assert_equal(matrix_power(self.R90,2**100+2**10+2**5),identity(2))

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
        self.do(a, None)

    def test_double(self):
        a = array([[1.,2.], [2.,1.]], dtype=double)
        self.do(a, None)

    def test_csingle(self):
        a = array([[1.,2+3j], [2-3j,1]], dtype=csingle)
        self.do(a, None)

    def test_cdouble(self):
        a = array([[1.,2+3j], [2-3j,1]], dtype=cdouble)
        self.do(a, None)

    def test_empty(self):
        a = atleast_2d(array([], dtype = double))
        assert_raises(linalg.LinAlgError, self.do, a, None)

    def test_nonarray(self):
        a = [[1,2], [2,1]]
        self.do(a, None)

    def test_matrix_b_only(self):
        """Check that matrix type is preserved."""
        a = array([[1.,2.], [2.,1.]])
        self.do(a, None)

    def test_matrix_a_and_b(self):
        """Check that matrix type is preserved."""
        a = matrix([[1.,2.], [2.,1.]])
        self.do(a, None)


HermitianGeneralizedTestCase = _generalized_testcase(
    'HermitianGeneralizedTestCase', HermitianTestCase)

class TestEigvalsh(HermitianTestCase, HermitianGeneralizedTestCase, TestCase):
    def do(self, a, b):
        # note that eigenvalue arrays must be sorted since
        # their order isn't guaranteed.
        ev = linalg.eigvalsh(a)
        evalues, evectors = linalg.eig(a)
        ev.sort(axis=-1)
        evalues.sort(axis=-1)
        assert_almost_equal(ev, evalues)

    def test_types(self):
        def check(dtype):
            x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
            assert_equal(np.linalg.eigvalsh(x), get_real_dtype(dtype))
        for dtype in [single, double, csingle, cdouble]:
            yield check, dtype


class TestEigh(HermitianTestCase, HermitianGeneralizedTestCase, TestCase):
    def do(self, a, b):
        # note that eigenvalue arrays must be sorted since
        # their order isn't guaranteed.
        ev, evc = linalg.eigh(a)
        evalues, evectors = linalg.eig(a)
        ev.sort(axis=-1)
        evalues.sort(axis=-1)
        assert_almost_equal(ev, evalues)

    def test_types(self):
        def check(dtype):
            x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
            w, v = np.linalg.eig(x)
            assert_equal(w, get_real_dtype(dtype))
            assert_equal(v, dtype)
        for dtype in [single, double, csingle, cdouble]:
            yield check, dtype


class _TestNorm(TestCase):

    dt = None
    dec = None

    def test_empty(self):
        assert_equal(norm([]), 0.0)
        assert_equal(norm(array([], dtype=self.dt)), 0.0)
        assert_equal(norm(atleast_2d(array([], dtype=self.dt))), 0.0)

    def test_vector(self):
        a = [1, 2, 3, 4]
        b = [-1, -2, -3, -4]
        c = [-1, 2, -3, 4]

        def _test(v):
            np.testing.assert_almost_equal(norm(v), 30**0.5,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, inf), 4.0,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -inf), 1.0,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 1), 10.0,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -1), 12.0/25,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 2), 30**0.5,
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -2), ((205./144)**-0.5),
                                           decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 0), 4,
                                           decimal=self.dec)

        for v in (a, b, c,):
            _test(v)

        for v in (array(a, dtype=self.dt), array(b, dtype=self.dt),
                  array(c, dtype=self.dt)):
            _test(v)

    def test_matrix(self):
        A = matrix([[1, 3], [5, 7]], dtype=self.dt)
        assert_almost_equal(norm(A), 84**0.5)
        assert_almost_equal(norm(A, 'fro'), 84**0.5)
        assert_almost_equal(norm(A, inf), 12.0)
        assert_almost_equal(norm(A, -inf), 4.0)
        assert_almost_equal(norm(A, 1), 10.0)
        assert_almost_equal(norm(A, -1), 6.0)
        assert_almost_equal(norm(A, 2), 9.1231056256176615)
        assert_almost_equal(norm(A, -2), 0.87689437438234041)

        self.assertRaises(ValueError, norm, A, 'nofro')
        self.assertRaises(ValueError, norm, A, -3)
        self.assertRaises(ValueError, norm, A, 0)

    def test_axis(self):
        # Vector norms.
        # Compare the use of `axis` with computing the norm of each row
        # or column separately.
        A = array([[1, 2, 3], [4, 5, 6]], dtype=self.dt)
        for order in [None, -1, 0, 1, 2, 3, np.Inf, -np.Inf]:
            expected0 = [norm(A[:,k], ord=order) for k in range(A.shape[1])]
            assert_almost_equal(norm(A, ord=order, axis=0), expected0)
            expected1 = [norm(A[k,:], ord=order) for k in range(A.shape[0])]
            assert_almost_equal(norm(A, ord=order, axis=1), expected1)

        # Matrix norms.
        B = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)

        for order in [None, -2, 2, -1, 1, np.Inf, -np.Inf, 'fro']:
            assert_almost_equal(norm(A, ord=order), norm(A, ord=order,
                                                         axis=(0, 1)))

            n = norm(B, ord=order, axis=(1, 2))
            expected = [norm(B[k], ord=order) for k in range(B.shape[0])]
            assert_almost_equal(n, expected)

            n = norm(B, ord=order, axis=(2, 1))
            expected = [norm(B[k].T, ord=order) for k in range(B.shape[0])]
            assert_almost_equal(n, expected)

            n = norm(B, ord=order, axis=(0, 2))
            expected = [norm(B[:,k,:], ord=order) for k in range(B.shape[1])]
            assert_almost_equal(n, expected)

            n = norm(B, ord=order, axis=(0, 1))
            expected = [norm(B[:,:,k], ord=order) for k in range(B.shape[2])]
            assert_almost_equal(n, expected)

    def test_bad_args(self):
        # Check that bad arguments raise the appropriate exceptions.

        A = array([[1, 2, 3], [4, 5, 6]], dtype=self.dt)
        B = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)

        # Using `axis=<integer>` or passing in a 1-D array implies vector
        # norms are being computed, so also using `ord='fro'` raises a
        # ValueError.
        self.assertRaises(ValueError, norm, A, 'fro', 0)
        self.assertRaises(ValueError, norm, [3, 4], 'fro', None)

        # Similarly, norm should raise an exception when ord is any finite
        # number other than 1, 2, -1 or -2 when computing matrix norms.
        for order in [0, 3]:
            self.assertRaises(ValueError, norm, A, order, None)
            self.assertRaises(ValueError, norm, A, order, (0, 1))
            self.assertRaises(ValueError, norm, B, order, (1, 2))

        # Invalid axis
        self.assertRaises(ValueError, norm, B, None, 3)
        self.assertRaises(ValueError, norm, B, None, (2, 3))
        self.assertRaises(ValueError, norm, B, None, (0, 1, 2))


class TestNormDouble(_TestNorm):
    dt = np.double
    dec = 12


class TestNormSingle(_TestNorm):
    dt = np.float32
    dec = 6


class TestNormInt64(_TestNorm):
    dt = np.int64
    dec = 12


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


    def check_qr(self, a):
        # This test expects the argument `a` to be an ndarray or
        # a subclass of an ndarray of inexact type.
        a_type = type(a)
        a_dtype = a.dtype
        m, n = a.shape
        k = min(m, n)

        # mode == 'complete'
        q, r = linalg.qr(a, mode='complete')
        assert_(q.dtype == a_dtype)
        assert_(r.dtype == a_dtype)
        assert_(isinstance(q, a_type))
        assert_(isinstance(r, a_type))
        assert_(q.shape == (m, m))
        assert_(r.shape == (m, n))
        assert_almost_equal(dot(q, r), a)
        assert_almost_equal(dot(q.T.conj(), q), np.eye(m))
        assert_almost_equal(np.triu(r), r)


        # mode == 'reduced'
        q1, r1 = linalg.qr(a, mode='reduced')
        assert_(q1.dtype == a_dtype)
        assert_(r1.dtype == a_dtype)
        assert_(isinstance(q1, a_type))
        assert_(isinstance(r1, a_type))
        assert_(q1.shape == (m, k))
        assert_(r1.shape == (k, n))
        assert_almost_equal(dot(q1, r1), a)
        assert_almost_equal(dot(q1.T.conj(), q1), np.eye(k))
        assert_almost_equal(np.triu(r1), r1)

        # mode == 'r'
        r2 = linalg.qr(a, mode='r')
        assert_(r2.dtype == a_dtype)
        assert_(isinstance(r2, a_type))
        assert_almost_equal(r2, r1)



    def test_qr_empty(self):
        a = np.zeros((0,2))
        self.assertRaises(linalg.LinAlgError, linalg.qr, a)


    def test_mode_raw(self):
        a = array([[1, 2], [3, 4], [5, 6]], dtype=np.double)
        b = a.astype(np.single)

        # m > n
        h1, tau1 = (
                array([[-5.91607978,  0.43377175,  0.72295291],
                      [-7.43735744,  0.82807867,  0.89262383]]),
                array([ 1.16903085,  1.113104  ])
                )
        # m > n
        h2, tau2 = (
                array([[-2.23606798,  0.61803399],
                       [-4.91934955, -0.89442719],
                       [-7.60263112, -1.78885438]]),
                array([ 1.4472136,  0.       ])
                )

        # Test double
        h, tau = linalg.qr(a, mode='raw')
        assert_(h.dtype == np.double)
        assert_(tau.dtype == np.double)
        old_assert_almost_equal(h, h1, decimal=8)
        old_assert_almost_equal(tau, tau1, decimal=8)

        h, tau = linalg.qr(a.T, mode='raw')
        assert_(h.dtype == np.double)
        assert_(tau.dtype == np.double)
        old_assert_almost_equal(h, h2, decimal=8)
        old_assert_almost_equal(tau, tau2, decimal=8)

        # Test single
        h, tau = linalg.qr(b, mode='raw')
        assert_(h.dtype == np.double)
        assert_(tau.dtype == np.double)
        old_assert_almost_equal(h, h1, decimal=8)
        old_assert_almost_equal(tau, tau1, decimal=8)


    def test_mode_all_but_economic(self):
        a = array([[1, 2], [3, 4]])
        b = array([[1, 2], [3, 4], [5, 6]])
        for dt in "fd":
            m1 = a.astype(dt)
            m2 = b.astype(dt)
            self.check_qr(m1)
            self.check_qr(m2)
            self.check_qr(m2.T)
            self.check_qr(matrix(m1))
        for dt in "fd":
            m1 = 1 + 1j * a.astype(dt)
            m2 = 1 + 1j * b.astype(dt)
            self.check_qr(m1)
            self.check_qr(m2)
            self.check_qr(m2.T)
            self.check_qr(matrix(m1))





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


def test_generalized_raise_multiloop():
    # It should raise an error even if the error doesn't occur in the
    # last iteration of the ufunc inner loop

    invertible = np.array([[1, 2], [3, 4]])
    non_invertible = np.array([[1, 1], [1, 1]])

    x = np.zeros([4, 4, 2, 2])[1::2]
    x[...] = invertible
    x[0,0] = non_invertible

    assert_raises(np.linalg.LinAlgError, np.linalg.inv, x)

def _is_xerbla_safe():
    """
    Check that running the xerbla test is safe --- if python_xerbla
    is not successfully linked in, the standard xerbla routine is called,
    which aborts the process.

    """

    try:
        pid = os.fork()
    except (OSError, AttributeError):
        # fork failed, or not running on POSIX
        return False

    if pid == 0:
        # child; close i/o file handles
        os.close(1)
        os.close(0)
        # avoid producing core files
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        # these calls may abort
        try:
            a = np.array([[1]])
            np.linalg.lapack_lite.dgetrf(
                1, 1, a.astype(np.double),
                0, # <- invalid value
                a.astype(np.intc), 0)
        except:
            pass
        try:
            np.linalg.lapack_lite.xerbla()
        except:
            pass
        os._exit(111)
    else:
        # parent
        pid, status = os.wait()
        if os.WEXITSTATUS(status) == 111 and not os.WIFSIGNALED(status):
            return True
    return False

@dec.skipif(not _is_xerbla_safe(), "python_xerbla not found")
def test_xerbla():
    # Test that xerbla works (with GIL)
    a = np.array([[1]])
    try:
        np.linalg.lapack_lite.dgetrf(
            1, 1, a.astype(np.double),
            0, # <- invalid value
            a.astype(np.intc), 0)
    except ValueError as e:
        assert_("DGETRF parameter number 4" in str(e))
    else:
        assert_(False)

    # Test that xerbla works (without GIL)
    assert_raises(ValueError, np.linalg.lapack_lite.xerbla)

if __name__ == "__main__":
    run_module_suite()
