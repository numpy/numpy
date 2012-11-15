"""
Test functions for gufuncs_linalg module
Heavily inspired (ripped in part) test_linalg
"""


"""
TODO:
    Implement proper tests for Eig
"""

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
# - add3
# - multiply3
# - multiply3_add
# - multiply_add
# - multiply_add2
# - multiply4
# - multiply4_add
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
import numpy.core.gufuncs_linalg as gula

old_assert_almost_equal = assert_almost_equal


def assert_almost_equal(a, b, **kw):
    if a.dtype.type in (single, csingle):
        decimal = 6
    else:
        decimal = 12
    old_assert_almost_equal(a, b, decimal = decimal, **kw)


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
#        ev = gula.eigvals(a)
#        assert_almost_equal(d, multiply.reduce(ev, axis=(ev.ndim-1)))
#        assert_almost_equal(s * np.exp(ld), multiply.reduce(ev, axis=(ev.ndim-1)))
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


class TestEig(GeneralTestCase, TestCase):
    def do(self, a, b):
        evalues, evectors = gula.eig(a)
        ev = gula.eigvals(a)

class TestEigh(HermitianTestCase, TestCase):
    def do(self, a, b):
        """ still work in progress """
        raise SkipTest
        evalues_lo, evectors_lo = gula.eigh(a, UPLO='L')
        evalues_up, evectors_up = gula.eigh(a, UPLO='U')

        assert_almost_equal(dot(a, evectors_lo), multiply(evectors_lo, evalues_lo))
        assert_almost_equal(dot(a, evectors_up), multiply(evectors_up, evalues_up))
        assert_almost_equal(evalues_lo, evalues_up)
        assert_almost_equal(evectors_lo, evectors_up)


class TestEigVals(GeneralTestCase, TestCase):
    def do(self, a, b):
        ev = gula.eigvals(a)
        evalues, evectors = gula.eig(a)
        assert_almost_equal(ev, evalues)


class TestEigvalsh(HermitianTestCase, TestCase):
    def do(self, a, b):
        ev_lo = gula.eigvalsh(a, UPLO='L')
        ev_up = gula.eigvalsh(a, UPLO='U')
        evalues_lo, evectors_lo = gula.eigh(a, UPLO='L')
        evalues_up, evectors_up = gula.eigh(a, UPLO='U')
        assert_equal(ev_lo, evalues_lo)
        assert_equal(ev_up, evalues_up)


class TestSolve(GeneralTestCase,TestCase):
    def do(self, a, b):
        x = gula.solve(a,b)
        assert_almost_equal(b,gula.matrix_multiply(a,x))

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

if __name__ == "__main__":
    run_module_suite()
