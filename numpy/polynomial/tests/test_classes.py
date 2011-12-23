"""Test inter-conversion of different polynomial classes.

This tests the convert and cast methods of all the polynomial classes.

"""
from __future__ import division

import numpy as np
from numpy.polynomial import (
        Polynomial, Legendre, Chebyshev, Laguerre,
        Hermite, HermiteE)
from numpy.testing import (
        TestCase, assert_almost_equal, assert_raises,
        assert_equal, assert_, run_module_suite)

classes = (
        Polynomial, Legendre, Chebyshev, Laguerre,
        Hermite, HermiteE)

class TestClassConversions(TestCase):

    def test_conversion(self):
        x = np.linspace(0, 1, 10)
        window = np.array([0, 1])
        domain = np.array([0, 1])
        for Poly1 in classes:
            d1 = domain + np.random.random((2,))*.25
            w1 = window + np.random.random((2,))*.25
            c1 = np.random.random((3,))
            p1 = Poly1(c1, domain=d1, window=w1)
            for Poly2 in classes:
                msg = "-- %s -> %s" % (Poly1.__name__, Poly2.__name__)
                d2 = domain + np.random.random((2,))*.25
                w2 = window + np.random.random((2,))*.25
                p2 = p1.convert(kind=Poly2, domain=d2, window=w2)
                assert_almost_equal(p2.domain, d2, err_msg=msg)
                assert_almost_equal(p2.window, w2, err_msg=msg)
                assert_almost_equal(p2(x), p1(x), err_msg=msg)

    def test_cast(self):
        x = np.linspace(0, 1, 10)
        window = np.array([0, 1])
        domain = np.array([0, 1])
        for Poly1 in classes:
            d1 = domain + np.random.random((2,))*.25
            w1 = window + np.random.random((2,))*.25
            c1 = np.random.random((3,))
            p1 = Poly1(c1, domain=d1, window=w1)
            for Poly2 in classes:
                msg = "-- %s -> %s" % (Poly1.__name__, Poly2.__name__)
                d2 = domain + np.random.random((2,))*.25
                w2 = window + np.random.random((2,))*.25
                p2 = Poly2.cast(p1, domain=d2, window=w2)
                assert_almost_equal(p2.domain, d2, err_msg=msg)
                assert_almost_equal(p2.window, w2, err_msg=msg)
                assert_almost_equal(p2(x), p1(x), err_msg=msg)


class TestClasses(TestCase):

    def test_basis(self):
        window = np.array([0, 1])
        domain = np.array([0, 1])
        for Poly in classes:
            msg = "-- %s" % (Poly.__name__,)
            d = domain + np.random.random((2,))*.25
            w = window + np.random.random((2,))*.25
            p = Poly.basis(5, domain=d, window=w)
            assert_equal(p.domain, d, err_msg=msg)
            assert_equal(p.window, w, err_msg=msg)
            assert_equal(p.coef, [0]*5 + [1])

    def test_fromroots(self):
        window = np.array([0, 1])
        domain = np.array([0, 1])
        for Poly in classes:
            msg = "-- %s" % (Poly.__name__,)
            d = domain + np.random.random((2,))*.25
            w = window + np.random.random((2,))*.25
            r = np.random.random((5,))
            p1 = Poly.fromroots(r, domain=d, window=w)
            assert_equal(p1.domain, d, err_msg=msg)
            assert_equal(p1.window, w, err_msg=msg)
            assert_almost_equal(p1(r), 0, err_msg=msg)
            # test monic with same domain and window
            p2 = Polynomial.cast(p1, domain=d, window=w)
            assert_almost_equal(p2.coef[-1], 1, err_msg=msg)

    def test_roots(self):
        window = np.array([0, 1])
        domain = np.array([0, 1])
        for Poly in classes:
            msg = "-- %s" % (Poly.__name__,)
            d = domain + np.random.random((2,))*.25
            w = window + np.random.random((2,))*.25
            tgt = np.sort(np.random.random((5,)))
            res = np.sort(Poly.fromroots(tgt).roots())
            assert_almost_equal(res, tgt, err_msg=msg)

    def test_degree(self):
        for Poly in classes:
            msg = "-- %s" % (Poly.__name__,)
            p = Poly.basis(5)
            assert_equal(p.degree(), 5, err_msg=msg)

    def test_copy(self):
        for Poly in classes:
            msg = "-- %s" % (Poly.__name__,)
            p1 = Poly.basis(5)
            p2 = p1.copy()
            assert_(p1 == p2, msg)
            assert_(p1 is not p2, msg)
            assert_(p1.coef is not p2.coef, msg)
            assert_(p1.domain is not p2.domain, msg)
            assert_(p1.window is not p2.window, msg)

    def test_deriv(self):
        # Check that the derivative is the inverse of integration. It is
        # assumes that the integration has been tested elsewhere.
        window = np.array([0, 1])
        domain = np.array([0, 1])
        for Poly in classes:
            msg = "-- %s" % (Poly.__name__,)
            d = domain + np.random.random((2,))*.25
            w = window + np.random.random((2,))*.25
            p1 = Poly([1, 2, 3], domain=d, window=w)
            p2 = p1.integ(2, k=[1, 2])
            p3 = p1.integ(1, k=[1])
            assert_almost_equal(p2.deriv(1).coef, p3.coef, err_msg=msg)
            assert_almost_equal(p2.deriv(2).coef, p1.coef, err_msg=msg)

    def test_linspace(self):
        window = np.array([0, 1])
        domain = np.array([0, 1])
        for Poly in classes:
            msg = "-- %s" % (Poly.__name__,)
            d = domain + np.random.random((2,))*.25
            w = window + np.random.random((2,))*.25
            p = Poly([1,2,3], domain=d, window=w)
            # test default domain
            xtgt = np.linspace(d[0], d[1], 20)
            ytgt = p(xtgt)
            xres, yres = p.linspace(20)
            assert_almost_equal(xres, xtgt, err_msg=msg)
            assert_almost_equal(yres, ytgt, err_msg=msg)
            # test specified domain
            xtgt = np.linspace(0, 2, 20)
            ytgt = p(xtgt)
            xres, yres = p.linspace(20, domain=[0, 2])
            assert_almost_equal(xres, xtgt, err_msg=msg)
            assert_almost_equal(yres, ytgt, err_msg=msg)

    def test_fit(self) :

        def f(x) :
            return x*(x - 1)*(x - 2)
        x = np.linspace(0,3)
        y = f(x)

        window = np.array([0, 1])
        domain = np.array([0, 1])
        for Poly in classes:
            msg = "-- %s" % (Poly.__name__,)
            # test default value of domain
            p = Poly.fit(x, y, 3)
            assert_almost_equal(p.domain, [0,3], err_msg=msg)
            assert_almost_equal(p(x), y, err_msg=msg)
            assert_equal(p.degree(), 3, err_msg=msg)

            # test with given windows and domains
            d = domain + np.random.random((2,))*.25
            w = window + np.random.random((2,))*.25
            p = Poly.fit(x, y, 3, domain=d, window=w)
            assert_almost_equal(p(x), y, err_msg=msg)
            assert_almost_equal(p.domain, d, err_msg=msg)
            assert_almost_equal(p.window, w, err_msg=msg)

            # test with class domain default
            p = Poly.fit(x, y, 3, [])
            assert_equal(p.domain, Poly.domain, err_msg=msg)
            assert_equal(p.window, Poly.window, err_msg=msg)

            # test that fit accepts weights.
            w = np.zeros_like(x)
            z = y + np.random.random(y.shape)*.25
            w[::2] = 1
            p1 = Poly.fit(x[::2], z[::2], 3)
            p2 = Poly.fit(x, z, 3, w=w)
            assert_almost_equal(p1(x), p2(x), err_msg=msg)


if __name__ == "__main__":
    run_module_suite()
