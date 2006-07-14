""" Test functions for linalg module
"""

from numpy.testing import *
set_package_path()
from numpy import array, single, double, csingle, cdouble, dot, identity, \
        multiply
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
    def _check(self, dtype):
        a = array([[1.,2.], [3.,4.]], dtype=dtype)
        b = array([2., 1.], dtype=dtype)
        self.do(a, b)

    def check_single(self):
        self._check(single)
    def check_double(self):
        self._check(double)
    def check_csingle(self):
        self._check(csingle)
    def check_cdouble(self):
        self._check(cdouble)

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
        ev = linalg.eigvals(a)
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
