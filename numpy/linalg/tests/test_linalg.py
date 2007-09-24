""" Test functions for linalg module
"""

from numpy.testing import *
set_package_path()
from numpy import array, single, double, csingle, cdouble, dot, identity, \
        multiply, atleast_2d
from numpy import linalg
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

class test_solve(LinalgTestCase):
    def do(self, a, b):
        x = linalg.solve(a, b)
        assert_almost_equal(b, dot(a, x))

class test_inv(LinalgTestCase):
    def do(self, a, b):
        a_inv = linalg.inv(a)
        assert_almost_equal(dot(a, a_inv), identity(a.shape[0]))

class test_eigvals(LinalgTestCase):
    def do(self, a, b):
        ev = linalg.eigvals(a)
        evalues, evectors = linalg.eig(a)
        assert_almost_equal(ev, evalues)

class test_eig(LinalgTestCase):
    def do(self, a, b):
        evalues, evectors = linalg.eig(a)
        assert_almost_equal(dot(a, evectors), evectors*evalues)

class test_svd(LinalgTestCase):
    def do(self, a, b):
        u, s, vt = linalg.svd(a, 0)
        assert_almost_equal(a, dot(u*s, vt))

class test_pinv(LinalgTestCase):
    def do(self, a, b):
        a_ginv = linalg.pinv(a)
        assert_almost_equal(dot(a, a_ginv), identity(a.shape[0]))

class test_det(LinalgTestCase):
    def do(self, a, b):
        d = linalg.det(a)
        if a.dtype.type in (single, double):
            ad = a.astype(double)
        else:
            ad = a.astype(cdouble)
        ev = linalg.eigvals(ad)
        assert_almost_equal(d, multiply.reduce(ev))

class test_lstsq(LinalgTestCase):
    def do(self, a, b):
        u, s, vt = linalg.svd(a, 0)
        x, residuals, rank, sv = linalg.lstsq(a, b)
        assert_almost_equal(b, dot(a, x))
        assert_equal(rank, a.shape[0])
        assert_almost_equal(sv, s)

if __name__ == '__main__':
    NumpyTest().run()
