"""Tests for hermendre module.

"""
from __future__ import division

import numpy as np
import numpy.polynomial.hermite as herm
import numpy.polynomial.polynomial as poly
from numpy.testing import *

H0 = np.array([   1])
H1 = np.array([0,     2])
H2 = np.array([  -2, 0,      4])
H3 = np.array([0,   -12, 0,      8])
H4 = np.array([  12, 0,    -48, 0,    16])
H5 = np.array([0,   120, 0,   -160, 0,    32])
H6 = np.array([-120, 0,    720, 0,  -480, 0,    64])
H7 = np.array([0, -1680, 0,   3360, 0, -1344, 0,   128])
H8 = np.array([1680, 0, -13440, 0, 13440, 0, -3584, 0, 256])
H9 = np.array([0, 30240, 0, -80640, 0, 48384, 0, -9216, 0, 512])

Hlist = [H0, H1, H2, H3, H4, H5, H6, H7, H8, H9]


def trim(x) :
    return herm.hermtrim(x, tol=1e-6)


class TestConstants(TestCase) :

    def test_hermdomain(self) :
        assert_equal(herm.hermdomain, [-1, 1])

    def test_hermzero(self) :
        assert_equal(herm.hermzero, [0])

    def test_hermone(self) :
        assert_equal(herm.hermone, [1])

    def test_hermx(self) :
        assert_equal(herm.hermx, [0, .5])


class TestArithmetic(TestCase) :
    x = np.linspace(-3, 3, 100)
    y0 = poly.polyval(x, H0)
    y1 = poly.polyval(x, H1)
    y2 = poly.polyval(x, H2)
    y3 = poly.polyval(x, H3)
    y4 = poly.polyval(x, H4)
    y5 = poly.polyval(x, H5)
    y6 = poly.polyval(x, H6)
    y7 = poly.polyval(x, H7)
    y8 = poly.polyval(x, H8)
    y9 = poly.polyval(x, H9)
    y = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9]

    def test_hermval(self) :
        def f(x) :
            return x*(x**2 - 1)

        #check empty input
        assert_equal(herm.hermval([], [1]).size, 0)

        #check normal input)
        for i in range(10) :
            msg = "At i=%d" % i
            ser = np.zeros
            tgt = self.y[i]
            res = herm.hermval(self.x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        #check that shape is preserved
        for i in range(3) :
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(herm.hermval(x, [1]).shape, dims)
            assert_equal(herm.hermval(x, [1,0]).shape, dims)
            assert_equal(herm.hermval(x, [1,0,0]).shape, dims)

    def test_hermadd(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = herm.hermadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermsub(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = herm.hermsub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermmulx(self):
        assert_equal(herm.hermmulx([0]), [0])
        assert_equal(herm.hermmulx([1]), [0,.5])
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [i, 0, .5]
            assert_equal(herm.hermmulx(ser), tgt)

    def test_hermmul(self) :
        # check values of result
        for i in range(5) :
            pol1 = [0]*i + [1]
            val1 = herm.hermval(self.x, pol1)
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                pol2 = [0]*j + [1]
                val2 = herm.hermval(self.x, pol2)
                pol3 = herm.hermmul(pol1, pol2)
                val3 = herm.hermval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1*val2, err_msg=msg)

    def test_hermdiv(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                ci = [0]*i + [1]
                cj = [0]*j + [1]
                tgt = herm.hermadd(ci, cj)
                quo, rem = herm.hermdiv(tgt, ci)
                res = herm.hermadd(herm.hermmul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestCalculus(TestCase) :

    def test_hermint(self) :
        # check exceptions
        assert_raises(ValueError, herm.hermint, [0], .5)
        assert_raises(ValueError, herm.hermint, [0], -1)
        assert_raises(ValueError, herm.hermint, [0], 1, [0,0])

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = herm.hermint([0], m=i, k=k)
            assert_almost_equal(res, [0, .5])

        # check single integration with integration constant
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            hermpol = herm.poly2herm(pol)
            hermint = herm.hermint(hermpol, m=1, k=[i])
            res = herm.herm2poly(hermint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            hermpol = herm.poly2herm(pol)
            hermint = herm.hermint(hermpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(herm.hermval(-1, hermint), i)

        # check single integration with integration constant and scaling
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            hermpol = herm.poly2herm(pol)
            hermint = herm.hermint(hermpol, m=1, k=[i], scl=2)
            res = herm.herm2poly(hermint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = herm.hermint(tgt, m=1)
                res = herm.hermint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = herm.hermint(tgt, m=1, k=[k])
                res = herm.hermint(pol, m=j, k=range(j))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = herm.hermint(tgt, m=1, k=[k], lbnd=-1)
                res = herm.hermint(pol, m=j, k=range(j), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = herm.hermint(tgt, m=1, k=[k], scl=2)
                res = herm.hermint(pol, m=j, k=range(j), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermder(self) :
        # check exceptions
        assert_raises(ValueError, herm.hermder, [0], .5)
        assert_raises(ValueError, herm.hermder, [0], -1)

        # check that zeroth deriviative does nothing
        for i in range(5) :
            tgt = [1] + [0]*i
            res = herm.hermder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = herm.hermder(herm.hermint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = herm.hermder(herm.hermint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))


class TestMisc(TestCase) :

    def test_hermfromroots(self) :
        res = herm.hermfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1,5) :
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            pol = herm.hermfromroots(roots)
            res = herm.hermval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(herm.herm2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_hermroots(self) :
        assert_almost_equal(herm.hermroots([1]), [])
        assert_almost_equal(herm.hermroots([1, 1]), [-.5])
        for i in range(2,5) :
            tgt = np.linspace(-1, 1, i)
            res = herm.hermroots(herm.hermfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_hermvander(self) :
        # check for 1d x
        x = np.arange(3)
        v = herm.hermvander(x, 3)
        assert_(v.shape == (3,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], herm.hermval(x, coef))

        # check for 2d x
        x = np.array([[1,2],[3,4],[5,6]])
        v = herm.hermvander(x, 3)
        assert_(v.shape == (3,2,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], herm.hermval(x, coef))

    def test_hermfit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)

        # Test exceptions
        assert_raises(ValueError, herm.hermfit, [1],    [1],     -1)
        assert_raises(TypeError,  herm.hermfit, [[1]],  [1],      0)
        assert_raises(TypeError,  herm.hermfit, [],     [1],      0)
        assert_raises(TypeError,  herm.hermfit, [1],    [[[1]]],  0)
        assert_raises(TypeError,  herm.hermfit, [1, 2], [1],      0)
        assert_raises(TypeError,  herm.hermfit, [1],    [1, 2],   0)
        assert_raises(TypeError,  herm.hermfit, [1],    [1],   0, w=[[1]])
        assert_raises(TypeError,  herm.hermfit, [1],    [1],   0, w=[1,1])

        # Test fit
        x = np.linspace(0,2)
        y = f(x)
        #
        coef3 = herm.hermfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(herm.hermval(x, coef3), y)
        #
        coef4 = herm.hermfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(herm.hermval(x, coef4), y)
        #
        coef2d = herm.hermfit(x, np.array([y,y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3,coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = herm.hermfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = herm.hermfit(x, np.array([yw,yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3,coef3]).T)

    def test_hermtrim(self) :
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, herm.hermtrim, coef, -1)

        # Test results
        assert_equal(herm.hermtrim(coef), coef[:-1])
        assert_equal(herm.hermtrim(coef, 1), coef[:-3])
        assert_equal(herm.hermtrim(coef, 2), [0])

    def test_hermline(self) :
        assert_equal(herm.hermline(3,4), [3, 2])

    def test_herm2poly(self) :
        for i in range(10) :
            assert_almost_equal(herm.herm2poly([0]*i + [1]), Hlist[i])

    def test_poly2herm(self) :
        for i in range(10) :
            assert_almost_equal(herm.poly2herm(Hlist[i]), [0]*i + [1])


def assert_poly_almost_equal(p1, p2):
    assert_almost_equal(p1.coef, p2.coef)
    assert_equal(p1.domain, p2.domain)


class TestHermiteClass(TestCase) :

    p1 = herm.Hermite([1,2,3])
    p2 = herm.Hermite([1,2,3], [0,1])
    p3 = herm.Hermite([1,2])
    p4 = herm.Hermite([2,2,3])
    p5 = herm.Hermite([3,2,3])

    def test_equal(self) :
        assert_(self.p1 == self.p1)
        assert_(self.p2 == self.p2)
        assert_(not self.p1 == self.p2)
        assert_(not self.p1 == self.p3)
        assert_(not self.p1 == [1,2,3])

    def test_not_equal(self) :
        assert_(not self.p1 != self.p1)
        assert_(not self.p2 != self.p2)
        assert_(self.p1 != self.p2)
        assert_(self.p1 != self.p3)
        assert_(self.p1 != [1,2,3])

    def test_add(self) :
        tgt = herm.Hermite([2,4,6])
        assert_(self.p1 + self.p1 == tgt)
        assert_(self.p1 + [1,2,3] == tgt)
        assert_([1,2,3] + self.p1 == tgt)

    def test_sub(self) :
        tgt = herm.Hermite([1])
        assert_(self.p4 - self.p1 == tgt)
        assert_(self.p4 - [1,2,3] == tgt)
        assert_([2,2,3] - self.p1 == tgt)

    def test_mul(self) :
        tgt = herm.Hermite([ 81.,  52.,  82.,  12.,   9.])
        assert_poly_almost_equal(self.p1 * self.p1, tgt)
        assert_poly_almost_equal(self.p1 * [1,2,3], tgt)
        assert_poly_almost_equal([1,2,3] * self.p1, tgt)

    def test_floordiv(self) :
        tgt = herm.Hermite([1])
        assert_(self.p4 // self.p1 == tgt)
        assert_(self.p4 // [1,2,3] == tgt)
        assert_([2,2,3] // self.p1 == tgt)

    def test_mod(self) :
        tgt = herm.Hermite([1])
        assert_((self.p4 % self.p1) == tgt)
        assert_((self.p4 % [1,2,3]) == tgt)
        assert_(([2,2,3] % self.p1) == tgt)

    def test_divmod(self) :
        tquo = herm.Hermite([1])
        trem = herm.Hermite([2])
        quo, rem = divmod(self.p5, self.p1)
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod(self.p5, [1,2,3])
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod([3,2,3], self.p1)
        assert_(quo == tquo and rem == trem)

    def test_pow(self) :
        tgt = herm.Hermite([1])
        for i in range(5) :
            res = self.p1**i
            assert_(res == tgt)
            tgt = tgt*self.p1

    def test_call(self) :
        # domain = [-1, 1]
        x = np.linspace(-1, 1)
        tgt = 3*(4*x**2 - 2) + 2*(2*x) + 1
        assert_almost_equal(self.p1(x), tgt)

        # domain = [0, 1]
        x = np.linspace(0, 1)
        xx = 2*x - 1
        assert_almost_equal(self.p2(x), self.p1(xx))

    def test_degree(self) :
        assert_equal(self.p1.degree(), 2)

    def test_cutdeg(self) :
        assert_raises(ValueError, self.p1.cutdeg, .5)
        assert_raises(ValueError, self.p1.cutdeg, -1)
        assert_equal(len(self.p1.cutdeg(3)), 3)
        assert_equal(len(self.p1.cutdeg(2)), 3)
        assert_equal(len(self.p1.cutdeg(1)), 2)
        assert_equal(len(self.p1.cutdeg(0)), 1)

    def test_convert(self) :
        x = np.linspace(-1,1)
        p = self.p1.convert(domain=[0,1])
        assert_almost_equal(p(x), self.p1(x))

    def test_mapparms(self) :
        parms = self.p2.mapparms()
        assert_almost_equal(parms, [-1, 2])

    def test_trim(self) :
        coef = [1, 1e-6, 1e-12, 0]
        p = herm.Hermite(coef)
        assert_equal(p.trim().coef, coef[:3])
        assert_equal(p.trim(1e-10).coef, coef[:2])
        assert_equal(p.trim(1e-5).coef, coef[:1])

    def test_truncate(self) :
        assert_raises(ValueError, self.p1.truncate, .5)
        assert_raises(ValueError, self.p1.truncate, 0)
        assert_equal(len(self.p1.truncate(4)), 3)
        assert_equal(len(self.p1.truncate(3)), 3)
        assert_equal(len(self.p1.truncate(2)), 2)
        assert_equal(len(self.p1.truncate(1)), 1)

    def test_copy(self) :
        p = self.p1.copy()
        assert_(self.p1 == p)

    def test_integ(self) :
        p = self.p2.integ()
        assert_almost_equal(p.coef, herm.hermint([1,2,3], 1, 0, scl=.5))
        p = self.p2.integ(lbnd=0)
        assert_almost_equal(p(0), 0)
        p = self.p2.integ(1, 1)
        assert_almost_equal(p.coef, herm.hermint([1,2,3], 1, 1, scl=.5))
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.coef, herm.hermint([1,2,3], 2, [1,2], scl=.5))

    def test_deriv(self) :
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.deriv(1).coef, self.p2.integ(1, [1]).coef)
        assert_almost_equal(p.deriv(2).coef, self.p2.coef)

    def test_roots(self) :
        p = herm.Hermite(herm.poly2herm([0, -1, 0, 1]), [0, 1])
        res = p.roots()
        tgt = [0, .5, 1]
        assert_almost_equal(res, tgt)

    def test_linspace(self):
        xdes = np.linspace(0, 1, 20)
        ydes = self.p2(xdes)
        xres, yres = self.p2.linspace(20)
        assert_almost_equal(xres, xdes)
        assert_almost_equal(yres, ydes)

    def test_fromroots(self) :
        roots = [0, .5, 1]
        p = herm.Hermite.fromroots(roots, domain=[0, 1])
        res = p.coef
        tgt = herm.poly2herm([0, -1, 0, 1])
        assert_almost_equal(res, tgt)

    def test_fit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)
        x = np.linspace(0,3)
        y = f(x)

        # test default value of domain
        p = herm.Hermite.fit(x, y, 3)
        assert_almost_equal(p.domain, [0,3])

        # test that fit works in given domains
        p = herm.Hermite.fit(x, y, 3, None)
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [0,3])
        p = herm.Hermite.fit(x, y, 3, [])
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [-1, 1])
        # test that fit accepts weights.
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        p = herm.Hermite.fit(x, yw, 3, w=w)
        assert_almost_equal(p(x), y)

    def test_identity(self) :
        x = np.linspace(0,3)
        p = herm.Hermite.identity()
        assert_almost_equal(p(x), x)
        p = herm.Hermite.identity([1,3])
        assert_almost_equal(p(x), x)
#

if __name__ == "__main__":
    run_module_suite()
