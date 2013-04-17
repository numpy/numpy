"""
Test functions for gufuncs_linalg module
Heavily inspired (ripped in part) test_linalg
"""
from __future__ import division, print_function

################################################################################
# The following functions are implemented in the module "gufuncs_linalg"
#
# category "linalg"
# - inv (TestInv)
# - poinv (TestPoinv)
# - det (TestDet)
# - slogdet (TestDet)
# - eig (TestEig)
# - eigh (TestEigh)
# - eigvals (TestEigvals)
# - eigvalsh (TestEigvalsh)
# - cholesky
# - solve (TestSolve)
# - chosolve (TestChosolve)
# - svd (TestSVD)

# ** unimplemented **
# - qr
# - matrix_power
# - matrix_rank
# - pinv
# - lstsq
# - tensorinv
# - tensorsolve
# - norm
# - cond
#
# category "inspired by pdl"
# - quadratic_form
# - matrix_multiply3
# - add3 (TestAdd3)
# - multiply3 (TestMultiply3)
# - multiply3_add (TestMultiply3Add)
# - multiply_add (TestMultiplyAdd)
# - multiply_add2 (TestMultiplyAdd2)
# - multiply4 (TestMultiply4)
# - multiply4_add (TestMultiply4Add)
#
# category "others"
# - convolve
# - inner1d
# - innerwt
# - matrix_multiply

from nose.plugins.skip import Skip, SkipTest
import numpy as np

from numpy.testing import (TestCase, assert_, assert_equal, assert_raises,
                           assert_array_equal, assert_almost_equal,
                           run_module_suite)

from numpy import array, single, double, csingle, cdouble, dot, identity
from numpy import multiply, inf
import numpy.linalg._gufuncs_linalg as gula

old_assert_almost_equal = assert_almost_equal

def assert_almost_equal(a, b, **kw):
    if a.dtype.type in (single, csingle):
        decimal = 5
    else:
        decimal = 10
    old_assert_almost_equal(a, b, decimal = decimal, **kw)


def assert_valid_eigen_no_broadcast(M, w, v, **kw):
    lhs = gula.matrix_multiply(M,v)
    rhs = w*v
    assert_almost_equal(lhs, rhs, **kw)


def assert_valid_eigen_recurse(M, w, v, **kw):
    """check that w and v are valid eigenvalues/eigenvectors for matrix M
    broadcast"""
    if len(M.shape) > 2:
        for i in range(M.shape[0]):
            assert_valid_eigen_recurse(M[i], w[i], v[i], **kw)
    else:
        if len(M.shape) == 2:
            assert_valid_eigen_no_broadcast(M, w, v, **kw)
        else:
            raise AssertionError('Not enough dimensions')


def assert_valid_eigen(M, w, v, **kw):
    if np.any(np.isnan(M)):
        raise AssertionError('nan found in matrix')
    if np.any(np.isnan(w)):
        raise AssertionError('nan found in eigenvalues')
    if np.any(np.isnan(v)):
        raise AssertionError('nan found in eigenvectors')

    assert_valid_eigen_recurse(M, w, v, **kw)


def assert_valid_eigenvals_no_broadcast(M, w, **kw):
    ident = np.eye(M.shape[0], dtype=M.dtype)
    for i in range(w.shape[0]):
        assert_almost_equal(gula.det(M - w[i]*ident), 0.0, **kw)


def assert_valid_eigenvals_recurse(M, w, **kw):
    if len(M.shape) > 2:
        for i in range(M.shape[0]):
            assert_valid_eigenvals_recurse(M[i], w[i], **kw)
    else:
        if len(M.shape) == 2:
            assert_valid_eigenvals_no_broadcast(M, w, **kw)
        else:
            raise AssertionError('Not enough dimensions')


def assert_valid_eigenvals(M, w, **kw):
    if np.any(np.isnan(M)):
        raise AssertionError('nan found in matrix')
    if np.any(np.isnan(w)):
        raise AssertionError('nan found in eigenvalues')
    assert_valid_eigenvals_recurse(M, w, **kw)


class MatrixGenerator(object):
    def real_matrices(self):
        a = [[1,2],
             [3,4]]

        b = [[4,3],
             [2,1]]

        return a, b
    
    def real_symmetric_matrices(self):
        a = [[ 2 ,-1],
             [-1 , 2]]

        b = [[4,3],
             [2,1]]

        return a, b

    def complex_matrices(self):
        a = [[1+2j,2+3j],
             [3+4j,4+5j]]

        b = [[4+3j,3+2j],
             [2+1j,1+0j]]

        return a, b

    def complex_hermitian_matrices(self):
        a = [[2,-1],
             [-1, 2]]

        b = [[4+3j,3+2j],
             [2-1j,1+0j]]
        
        return a, b

    def real_matrices_vector(self):
        a, b = self.real_matrices()
        return [a], [b]

    def real_symmetric_matrices_vector(self):
        a, b = self.real_symmetric_matrices()
        return [a], [b]

    def complex_matrices_vector(self):
        a, b = self.complex_matrices()
        return [a], [b]

    def complex_hermitian_matrices_vector(self):
        a, b = self.complex_hermitian_matrices()
        return [a], [b]


class GeneralTestCase(MatrixGenerator):
    def test_single(self):
        a,b = self.real_matrices()
        self.do(array(a, dtype=single),
                array(b, dtype=single))

    def test_double(self):
        a, b = self.real_matrices()
        self.do(array(a, dtype=double),
                array(b, dtype=double))

    def test_csingle(self):
        a, b = self.complex_matrices()
        self.do(array(a, dtype=csingle),
                array(b, dtype=csingle))

    def test_cdouble(self):
        a, b = self.complex_matrices()
        self.do(array(a, dtype=cdouble),
                array(b, dtype=cdouble))

    def test_vector_single(self):
        a,b = self.real_matrices_vector()
        self.do(array(a, dtype=single),
                array(b, dtype=single))

    def test_vector_double(self):
        a, b = self.real_matrices_vector()
        self.do(array(a, dtype=double),
                array(b, dtype=double))

    def test_vector_csingle(self):
        a, b = self.complex_matrices_vector()
        self.do(array(a, dtype=csingle),
                array(b, dtype=csingle))

    def test_vector_cdouble(self):
        a, b = self.complex_matrices_vector()
        self.do(array(a, dtype=cdouble),
                array(b, dtype=cdouble))


class HermitianTestCase(MatrixGenerator):
    def test_single(self):
        a,b = self.real_symmetric_matrices()
        self.do(array(a, dtype=single),
                array(b, dtype=single))

    def test_double(self):
        a, b = self.real_symmetric_matrices()
        self.do(array(a, dtype=double),
                array(b, dtype=double))

    def test_csingle(self):
        a, b = self.complex_hermitian_matrices()
        self.do(array(a, dtype=csingle),
                array(b, dtype=csingle))

    def test_cdouble(self):
        a, b = self.complex_hermitian_matrices()
        self.do(array(a, dtype=cdouble),
                array(b, dtype=cdouble))

    def test_vector_single(self):
        a,b = self.real_symmetric_matrices_vector()
        self.do(array(a, dtype=single),
                array(b, dtype=single))

    def test_vector_double(self):
        a, b = self.real_symmetric_matrices_vector()
        self.do(array(a, dtype=double),
                array(b, dtype=double))

    def test_vector_csingle(self):
        a, b = self.complex_hermitian_matrices_vector()
        self.do(array(a, dtype=csingle),
                array(b, dtype=csingle))

    def test_vector_cdouble(self):
        a, b = self.complex_hermitian_matrices_vector()
        self.do(array(a, dtype=cdouble),
                array(b, dtype=cdouble))


class TestMatrixMultiply(GeneralTestCase):
    def do(self, a, b):
        res = gula.matrix_multiply(a,b)
        if a.ndim == 2:
            assert_almost_equal(res, np.dot(a,b))
        else:
            assert_almost_equal(res[0], np.dot(a[0],b[0]))

    def test_column_matrix(self):
        A = np.arange(2*2).reshape((2,2))
        B = np.arange(2*1).reshape((2,1))
        res = gula.matrix_multiply(A,B)
        assert_almost_equal(res, np.dot(A,B))

class TestInv(GeneralTestCase, TestCase):
    def do(self, a, b):
        a_inv = gula.inv(a)
        ident = identity(a.shape[-1])
        if 3 == len(a.shape):
            ident = ident.reshape((1, ident.shape[0], ident.shape[1]))
        assert_almost_equal(gula.matrix_multiply(a, a_inv), ident)


class TestPoinv(HermitianTestCase, TestCase):
    def do(self, a, b):
        a_inv = gula.poinv(a)
        ident = identity(a.shape[-1])
        if 3 == len(a.shape):
            ident = ident.reshape((1,ident.shape[0], ident.shape[1]))

        assert_almost_equal(a_inv, gula.inv(a))
        assert_almost_equal(gula.matrix_multiply(a, a_inv), ident)


class TestDet(GeneralTestCase, TestCase):
    def do(self, a, b):
        d = gula.det(a)
        s, ld = gula.slogdet(a)
        assert_almost_equal(s * np.exp(ld), d)

        if np.csingle == a.dtype.type or np.single == a.dtype.type:
            cmp_type=np.csingle
        else:
            cmp_type=np.cdouble

        ev = gula.eigvals(a.astype(cmp_type))
        assert_almost_equal(d.astype(cmp_type),
                            multiply.reduce(ev.astype(cmp_type),
                                            axis=(ev.ndim-1)))
        if s != 0:
            assert_almost_equal(np.abs(s), 1)
        else:
            assert_equal(ld, -inf)

    def test_zero(self):
        assert_equal(gula.det(array([[0.0]], dtype=single)), 0.0)
        assert_equal(gula.det(array([[0.0]], dtype=double)), 0.0)
        assert_equal(gula.det(array([[0.0]], dtype=csingle)), 0.0)
        assert_equal(gula.det(array([[0.0]], dtype=cdouble)), 0.0)

        assert_equal(gula.slogdet(array([[0.0]], dtype=single)), (0.0, -inf))
        assert_equal(gula.slogdet(array([[0.0]], dtype=double)), (0.0, -inf))
        assert_equal(gula.slogdet(array([[0.0]], dtype=csingle)), (0.0, -inf))
        assert_equal(gula.slogdet(array([[0.0]], dtype=cdouble)), (0.0, -inf))

    def test_types(self):
        for typ in [(single, single), 
                    (double, double), 
                    (csingle, single),
                    (cdouble, double)]:
            for x in [ [0], [[0]], [[[0]]] ]:
                assert_equal(gula.det(array(x, dtype=typ[0])).dtype, typ[0])
                assert_equal(gula.slogdet(array(x, dtype=typ[0]))[0].dtype, typ[0])
                assert_equal(gula.slogdet(array(x, dtype=typ[0]))[1].dtype, typ[1])
        

class TestEig(GeneralTestCase, TestCase):
    def do(self, a, b):
        evalues, evectors = gula.eig(a)
        assert_valid_eigenvals(a, evalues)
        assert_valid_eigen(a, evalues, evectors)
        ev = gula.eigvals(a)
        assert_valid_eigenvals(a, evalues)
        assert_almost_equal(ev, evalues)


class TestEigh(HermitianTestCase, TestCase):
    def do(self, a, b):
        evalues_lo, evectors_lo = gula.eigh(a, UPLO='L')
        evalues_up, evectors_up = gula.eigh(a, UPLO='U')

        assert_valid_eigenvals(a, evalues_lo)
        assert_valid_eigenvals(a, evalues_up)
        assert_valid_eigen(a, evalues_lo, evectors_lo)
        assert_valid_eigen(a, evalues_up, evectors_up)
        assert_almost_equal(evalues_lo, evalues_up)
        assert_almost_equal(evectors_lo, evectors_up)

        ev_lo = gula.eigvalsh(a, UPLO='L')
        ev_up = gula.eigvalsh(a, UPLO='U')
        assert_valid_eigenvals(a, ev_lo)
        assert_valid_eigenvals(a, ev_up)
        assert_almost_equal(ev_lo, evalues_lo)
        assert_almost_equal(ev_up, evalues_up)


class TestSolve(GeneralTestCase,TestCase):
    def do(self, a, b):
        x = gula.solve(a,b)
        assert_almost_equal(b, gula.matrix_multiply(a,x))


class TestChosolve(HermitianTestCase, TestCase):
    def do(self, a, b):
        """ 
        inner1d not defined for complex types.
        todo: implement alternative test
        """
        if csingle == a.dtype or cdouble == a.dtype:
            raise SkipTest

        x_lo = gula.chosolve(a, b, UPLO='L')
        x_up = gula.chosolve(a, b, UPLO='U')
        assert_almost_equal(x_lo, x_up)
        # inner1d not defined for complex types
        # todo: implement alternative test
        assert_almost_equal(b, gula.matrix_multiply(a,x_lo))
        assert_almost_equal(b, gula.matrix_multiply(a,x_up))


class TestSVD(GeneralTestCase, TestCase):
    def do(self, a, b):
        """ still work in progress """
        raise SkipTest
        u, s, vt = gula.svd(a, 0)
        assert_almost_equal(a, dot(multiply(u, s), vt))

"""
class TestCholesky(HermitianTestCase, TestCase):
    def do(self, a, b):
        pass
"""

################################################################################
# ufuncs inspired by pdl
# - add3
# - multiply3
# - multiply3_add
# - multiply_add
# - multiply_add2
# - multiply4
# - multiply4_add

class UfuncTestCase(object):
    parameter = range(0,10)

    def _check_for_type(self, typ):
        a = np.array(self.__class__.parameter, dtype=typ)
        self.do(a)

    def _check_for_type_vector(self, typ):
        parameter = self.__class__.parameter
        a = np.array([parameter, parameter], dtype=typ)
        self.do(a)
 
    def test_single(self):
        self._check_for_type(single)

    def test_double(self):
        self._check_for_type(double)

    def test_csingle(self):
        self._check_for_type(csingle)

    def test_cdouble(self):
        self._check_for_type(cdouble)

    def test_single_vector(self):
        self._check_for_type_vector(single)

    def test_double_vector(self):
        self._check_for_type_vector(double)

    def test_csingle_vector(self):
        self._check_for_type_vector(csingle)

    def test_cdouble_vector(self):
        self._check_for_type_vector(cdouble)


class TestAdd3(UfuncTestCase, TestCase):
    def do(self, a):
        r = gula.add3(a,a,a)
        assert_almost_equal(r, a+a+a)


class TestMultiply3(UfuncTestCase, TestCase):
    def do(self, a):
        r = gula.multiply3(a,a,a)
        assert_almost_equal(r, a*a*a)


class TestMultiply3Add(UfuncTestCase, TestCase):
    def do(self, a):
        r = gula.multiply3_add(a,a,a,a)
        assert_almost_equal(r, a*a*a+a)


class TestMultiplyAdd(UfuncTestCase, TestCase):
    def do(self, a):
        r = gula.multiply_add(a,a,a)
        assert_almost_equal(r, a*a+a)


class TestMultiplyAdd2(UfuncTestCase, TestCase):
    def do(self, a):
        r = gula.multiply_add2(a,a,a,a)
        assert_almost_equal(r, a*a+a+a)


class TestMultiply4(UfuncTestCase, TestCase):
    def do(self, a):
        r = gula.multiply4(a,a,a,a)
        assert_almost_equal(r, a*a*a*a)


class TestMultiply4_add(UfuncTestCase, TestCase):
    def do(self, a):
        r = gula.multiply4_add(a,a,a,a,a)
        assert_almost_equal(r, a*a*a*a+a)


if __name__ == "__main__":
    print('testing gufuncs_linalg; gufuncs version: %s' % gula._impl.__version__)
    run_module_suite()
