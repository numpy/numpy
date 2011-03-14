"""Tests for hermendre module.

"""
from __future__ import division

import numpy as np
import numpy.polynomial.laguerre as lag
import numpy.polynomial.polynomial as poly
from numpy.testing import *

L0 = np.array([1 ])/1
L1 = np.array([1 , -1 ])/1
L2 = np.array([2 , -4 , 1 ])/2
L3 = np.array([6 , -18 , 9 , -1 ])/6
L4 = np.array([24 , -96 , 72 , -16 , 1 ])/24
L5 = np.array([120 , -600 , 600 , -200 , 25 , -1 ])/120
L6 = np.array([720 , -4320 , 5400 , -2400 , 450 , -36 , 1 ])/720

Llist = [L0, L1, L2, L3, L4, L5, L6]

def trim(x) :
    return lag.lagtrim(x, tol=1e-6)


class TestConstants(TestCase) :

    def test_lagdomain(self) :
        assert_equal(lag.lagdomain, [0, 1])

    def test_lagzero(self) :
        assert_equal(lag.lagzero, [0])

    def test_lagone(self) :
        assert_equal(lag.lagone, [1])

    def test_lagx(self) :
        assert_equal(lag.lagx, [1, -1])


class TestArithmetic(TestCase) :
    x = np.linspace(-3, 3, 100)
    y0 = poly.polyval(x, L0)
    y1 = poly.polyval(x, L1)
    y2 = poly.polyval(x, L2)
    y3 = poly.polyval(x, L3)
    y4 = poly.polyval(x, L4)
    y5 = poly.polyval(x, L5)
    y6 = poly.polyval(x, L6)
    y = [y0, y1, y2, y3, y4, y5, y6]

    def test_lagval(self) :
        def f(x) :
            return x*(x**2 - 1)

        #check empty input
        assert_equal(lag.lagval([], [1]).size, 0)

        #check normal input)
        for i in range(7) :
            msg = "At i=%d" % i
            ser = np.zeros
            tgt = self.y[i]
            res = lag.lagval(self.x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        #check that shape is preserved
        for i in range(3) :
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(lag.lagval(x, [1]).shape, dims)
            assert_equal(lag.lagval(x, [1,0]).shape, dims)
            assert_equal(lag.lagval(x, [1,0,0]).shape, dims)

    def test_lagadd(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = lag.lagadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagsub(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = lag.lagsub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagmulx(self):
        assert_equal(lag.lagmulx([0]), [0])
        assert_equal(lag.lagmulx([1]), [1,-1])
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [-i, 2*i + 1, -(i + 1)]
            assert_almost_equal(lag.lagmulx(ser), tgt)

    def test_lagmul(self) :
        # check values of result
        for i in range(5) :
            pol1 = [0]*i + [1]
            val1 = lag.lagval(self.x, pol1)
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                pol2 = [0]*j + [1]
                val2 = lag.lagval(self.x, pol2)
                pol3 = lag.lagmul(pol1, pol2)
                val3 = lag.lagval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1*val2, err_msg=msg)

    def test_lagdiv(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                ci = [0]*i + [1]
                cj = [0]*j + [1]
                tgt = lag.lagadd(ci, cj)
                quo, rem = lag.lagdiv(tgt, ci)
                res = lag.lagadd(lag.lagmul(quo, ci), rem)
                assert_almost_equal(trim(res), trim(tgt), err_msg=msg)


class TestCalculus(TestCase) :

    def test_lagint(self) :
        # check exceptions
        assert_raises(ValueError, lag.lagint, [0], .5)
        assert_raises(ValueError, lag.lagint, [0], -1)
        assert_raises(ValueError, lag.lagint, [0], 1, [0,0])

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = lag.lagint([0], m=i, k=k)
            assert_almost_equal(res, [1, -1])

        # check single integration with integration constant
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            lagpol = lag.poly2lag(pol)
            lagint = lag.lagint(lagpol, m=1, k=[i])
            res = lag.lag2poly(lagint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            lagpol = lag.poly2lag(pol)
            lagint = lag.lagint(lagpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(lag.lagval(-1, lagint), i)

        # check single integration with integration constant and scaling
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            lagpol = lag.poly2lag(pol)
            lagint = lag.lagint(lagpol, m=1, k=[i], scl=2)
            res = lag.lag2poly(lagint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = lag.lagint(tgt, m=1)
                res = lag.lagint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = lag.lagint(tgt, m=1, k=[k])
                res = lag.lagint(pol, m=j, k=range(j))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = lag.lagint(tgt, m=1, k=[k], lbnd=-1)
                res = lag.lagint(pol, m=j, k=range(j), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = lag.lagint(tgt, m=1, k=[k], scl=2)
                res = lag.lagint(pol, m=j, k=range(j), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_lagder(self) :
        # check exceptions
        assert_raises(ValueError, lag.lagder, [0], .5)
        assert_raises(ValueError, lag.lagder, [0], -1)

        # check that zeroth deriviative does nothing
        for i in range(5) :
            tgt = [1] + [0]*i
            res = lag.lagder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = lag.lagder(lag.lagint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = lag.lagder(lag.lagint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))


class TestMisc(TestCase) :

    def test_lagfromroots(self) :
        res = lag.lagfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1,5) :
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            pol = lag.lagfromroots(roots)
            res = lag.lagval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(lag.lag2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_lagroots(self) :
        assert_almost_equal(lag.lagroots([1]), [])
        assert_almost_equal(lag.lagroots([0, 1]), [1])
        for i in range(2,5) :
            tgt = np.linspace(0, 3, i)
            res = lag.lagroots(lag.lagfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_lagvander(self) :
        # check for 1d x
        x = np.arange(3)
        v = lag.lagvander(x, 3)
        assert_(v.shape == (3,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], lag.lagval(x, coef))

        # check for 2d x
        x = np.array([[1,2],[3,4],[5,6]])
        v = lag.lagvander(x, 3)
        assert_(v.shape == (3,2,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], lag.lagval(x, coef))

    def test_lagfit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)

        # Test exceptions
        assert_raises(ValueError, lag.lagfit, [1],    [1],     -1)
        assert_raises(TypeError,  lag.lagfit, [[1]],  [1],      0)
        assert_raises(TypeError,  lag.lagfit, [],     [1],      0)
        assert_raises(TypeError,  lag.lagfit, [1],    [[[1]]],  0)
        assert_raises(TypeError,  lag.lagfit, [1, 2], [1],      0)
        assert_raises(TypeError,  lag.lagfit, [1],    [1, 2],   0)
        assert_raises(TypeError,  lag.lagfit, [1],    [1],   0, w=[[1]])
        assert_raises(TypeError,  lag.lagfit, [1],    [1],   0, w=[1,1])

        # Test fit
        x = np.linspace(0,2)
        y = f(x)
        #
        coef3 = lag.lagfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(lag.lagval(x, coef3), y)
        #
        coef4 = lag.lagfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(lag.lagval(x, coef4), y)
        #
        coef2d = lag.lagfit(x, np.array([y,y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3,coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = lag.lagfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = lag.lagfit(x, np.array([yw,yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3,coef3]).T)

    def test_lagtrim(self) :
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, lag.lagtrim, coef, -1)

        # Test results
        assert_equal(lag.lagtrim(coef), coef[:-1])
        assert_equal(lag.lagtrim(coef, 1), coef[:-3])
        assert_equal(lag.lagtrim(coef, 2), [0])

    def test_lagline(self) :
        assert_equal(lag.lagline(3,4), [7, -4])

    def test_lag2poly(self) :
        for i in range(7) :
            assert_almost_equal(lag.lag2poly([0]*i + [1]), Llist[i])

    def test_poly2lag(self) :
        for i in range(7) :
            assert_almost_equal(lag.poly2lag(Llist[i]), [0]*i + [1])


def assert_poly_almost_equal(p1, p2):
    assert_almost_equal(p1.coef, p2.coef)
    assert_equal(p1.domain, p2.domain)


class TestLaguerreClass(TestCase) :

    p1 = lag.Laguerre([1,2,3])
    p2 = lag.Laguerre([1,2,3], [0,1])
    p3 = lag.Laguerre([1,2])
    p4 = lag.Laguerre([2,2,3])
    p5 = lag.Laguerre([3,2,3])

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
        tgt = lag.Laguerre([2,4,6])
        assert_(self.p1 + self.p1 == tgt)
        assert_(self.p1 + [1,2,3] == tgt)
        assert_([1,2,3] + self.p1 == tgt)

    def test_sub(self) :
        tgt = lag.Laguerre([1])
        assert_(self.p4 - self.p1 == tgt)
        assert_(self.p4 - [1,2,3] == tgt)
        assert_([2,2,3] - self.p1 == tgt)

    def test_mul(self) :
        tgt = lag.Laguerre([ 14., -16.,  56., -72.,  54.])
        assert_poly_almost_equal(self.p1 * self.p1, tgt)
        assert_poly_almost_equal(self.p1 * [1,2,3], tgt)
        assert_poly_almost_equal([1,2,3] * self.p1, tgt)

    def test_floordiv(self) :
        tgt = lag.Laguerre([1])
        assert_(self.p4 // self.p1 == tgt)
        assert_(self.p4 // [1,2,3] == tgt)
        assert_([2,2,3] // self.p1 == tgt)

    def test_mod(self) :
        tgt = lag.Laguerre([1])
        assert_((self.p4 % self.p1) == tgt)
        assert_((self.p4 % [1,2,3]) == tgt)
        assert_(([2,2,3] % self.p1) == tgt)

    def test_divmod(self) :
        tquo = lag.Laguerre([1])
        trem = lag.Laguerre([2])
        quo, rem = divmod(self.p5, self.p1)
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod(self.p5, [1,2,3])
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod([3,2,3], self.p1)
        assert_(quo == tquo and rem == trem)

    def test_pow(self) :
        tgt = lag.Laguerre([1])
        for i in range(5) :
            res = self.p1**i
            assert_(res == tgt)
            tgt = tgt*self.p1

    def test_call(self) :
        # domain = [0, 1]
        x = np.linspace(0, 1)
        tgt = 3*(.5*x**2 - 2*x + 1) + 2*(-x + 1) + 1
        assert_almost_equal(self.p1(x), tgt)

        # domain = [0, 1]
        x = np.linspace(.5, 1)
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
        p = lag.Laguerre(coef)
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
        assert_almost_equal(p.coef, lag.lagint([1,2,3], 1, 0, scl=.5))
        p = self.p2.integ(lbnd=0)
        assert_almost_equal(p(0), 0)
        p = self.p2.integ(1, 1)
        assert_almost_equal(p.coef, lag.lagint([1,2,3], 1, 1, scl=.5))
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.coef, lag.lagint([1,2,3], 2, [1,2], scl=.5))

    def test_deriv(self) :
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.deriv(1).coef, self.p2.integ(1, [1]).coef)
        assert_almost_equal(p.deriv(2).coef, self.p2.coef)

    def test_roots(self) :
        p = lag.Laguerre(lag.poly2lag([0, -1, 0, 1]), [0, 1])
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
        p = lag.Laguerre.fromroots(roots, domain=[0, 1])
        res = p.coef
        tgt = lag.poly2lag([0, -1, 0, 1])
        assert_almost_equal(res, tgt)

    def test_fit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)
        x = np.linspace(0,3)
        y = f(x)

        # test default value of domain
        p = lag.Laguerre.fit(x, y, 3)
        assert_almost_equal(p.domain, [0,3])

        # test that fit works in given domains
        p = lag.Laguerre.fit(x, y, 3, None)
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [0,3])
        p = lag.Laguerre.fit(x, y, 3, [])
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [-1, 1])
        # test that fit accepts weights.
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        p = lag.Laguerre.fit(x, yw, 3, w=w)
        assert_almost_equal(p(x), y)

    def test_identity(self) :
        x = np.linspace(0,3)
        p = lag.Laguerre.identity()
        assert_almost_equal(p(x), x)
        p = lag.Laguerre.identity([1,3])
        assert_almost_equal(p(x), x)
#

if __name__ == "__main__":
    run_module_suite()
