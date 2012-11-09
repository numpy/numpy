"""
Test functions for gufuncs_linalg module
Heavily inspired (ripped in part) test_linalg
"""

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


class GeneralTestCase(object):
    def test_single(self):
        a = array([[[1.,2.], [3.,4.]]], dtype=single)
        b = array([[2.,1.]], dtype=single)
        self.do(a, b)

    def test_double(self):
        a = array([[[1.,2.], [3.,4.]]], dtype=double)
        b = array([[2.,1.]], dtype=double)
        self.do(a, b)

    def test_double_2(self):
        a = array([[[1.,2.], [3.,4.]]], dtype=double)
        b = array([[[2.,1., 4.], [3.,4.,6.]]], dtype=double)
        self.do(a, b)

    def test_csingle(self):
        a = array([[[1+2j,2+3j],[3+4j,4+5j]]], dtype=csingle)
        b = array([[2+1j,1+2j]], dtype=csingle)
        self.do(a, b)

    def test_double(self):
        a = array([[[1+2j, 2+3j], [3+4j, 4+5j]]], dtype=cdouble)
        b = array([[2+1j,1+2j]], dtype=cdouble)
        self.do(a,b)

    def test_cdouble_2(self):
        a = array([[[1+2j, 2+3j], [3+4j, 4+5j]]], dtype=cdouble)
        b = array([[[2+1j, 1+2j, 1+3j], [1-2j, 1-3j, 1-6j]]], dtype=cdouble)
        self.do(a,b)


class HermitianTestCase(object):
    def test_single(self):
        a = array([[[1,2], [2,1]]], dtype=single)
        b = array([[2.,1.]], dtype=single)
        self.do(a,b)

    def test_double(self):
        a = array([[[1,2], [2,1]]], dtype=double)
        b = array([[2.,1.]], dtype=double)
        self.do(a,b)

    def test_double_2(self):
        a = array([[[1.,2.], [3.,4.]]], dtype=double)
        b = array([[[2.,1., 4.], [3.,4.,6.]]], dtype=double)
        self.do(a,b)

    def test_csingle(self):
        a = array([[[1,2+3j], [2-3j,1]]], dtype=csingle)
        b = array([[2+1j,1+2j]], dtype=csingle)
        self.do(a,b)

    def test_cdouble(self):
        a = array([[[1,2+3j], [2-3j,1]]], dtype=cdouble)
        b = array([[2+1j,1+2j]], dtype=cdouble)
        self.do(a,b)

    def test_cdouble_2(self):
        a = array([[[1,2+3j], [2-3j,1]]], dtype=cdouble)
        b = array([[[2+1j, 1+2j, 1+3j], [1-2j, 1-3j, 1-6j]]], dtype=cdouble)
        self.do(a,b)

class TestInv(GeneralTestCase, TestCase):
    def do(self, a, b):
        a_inv = gula.inv(a)
        assert_almost_equal(dot(a, a_inv), identity(a.shape[0]))


class TestPoinv(HermitianTestCase, TestCase):
    def do(self, a, b):
        a_inv = gula.poinv(a)
        assert_almost_equal(dot(a, a_inv), identity(a.shape[0]))


class TestDet(GeneralTestCase, TestCase):
    def do(self, a, b):
        d = gula.det(a)
        s, ld = gula.slogdet(a)
        ev = gula.eigvals(a)
        assert_almost_equal(d, multiply.reduce(ev, axis=(ev.ndim-1)))
        assert_almost_equal(s * np.exp(ld), multiply.reduce(ev, axis=(ev.ndim-1)))
        if s != 0:
            assert_almost_equal(np.abs(s), 1)
        else:
            assert_equal(ld, -inf)

    def test_zero(self):
        assert_equal(gula.det([[0.0]]), 0.0)
        assert_equal(gula.det([[0.0j]]), 0.0)
        assert_equal(gula.slogdet([[0.0]]), (0.0, -inf))
        assert_equal(gula.slogdet([[0-0j]]), (0.0j, -inf))


class TestEig(GeneralTestCase, TestCase):
    def do(self, a, b):
        evalues, evectors = gula.eig(a)
        assert_almost_equal(dot(a, evectors), multiply(evectors, evalues))


class TestEigh(HermitianTestCase, TestCase):
    def do(self, a, b):
        evalues_lo, evectors_lo = gula.eigh(a,'L')
        evalues_up, evectors_up = gula.eigh(a,'U')

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
        ev_lo = gula.eigvalsh(a, 'L')
        ev_up = gula.eigvalsh(a, 'U')
        evalues_lo, evectors_lo = gula.eigh(a, 'L')
        evalues_up, evectors_up = gula.eigh(a, 'U')
        assert_equal(ev_lo, evalues_lo)
        assert_equal(ev_up, evalues_up)

"""
class TestSolve(GeneralTestCase,TestCase):
    def do(self, a, b):
        print a
        print b
        x = gula.solve(a,b)
        assert_almost_equal(b,gula.matrix_multiply(a,x.T))
"""

class TestChosolve(HermitianTestCase, TestCase):
    def do(self, a, b):
        x_lo = gula.chosolve(a,b,'L')
        x_up = gula.chosolve(a,b,'U')
        assert_almost_equal(x_lo, x_up)
        if a.dtype == single or a.dtype == double:
            assert_almost_equal(b, gula.inner1d(a,x_lo))
            assert_almost_equal(a, gula.inner1d(a,x_up))



class TestSVD(GeneralTestCase, TestCase):
    def do(self, a, b):
        u, s, vt = gula.svd(a, 0)
        assert_almost_equal(a, dot(multiply(u, s), vt))

"""
class TestCholesky(HermitianTestCase, TestCase):
    def do(self, a, b):
        pass
"""

if __name__ == "__main__":
    run_module_suite()
