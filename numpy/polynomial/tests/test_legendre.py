"""Tests for legendre module.

"""
from __future__ import division

import numpy as np
import numpy.polynomial.legendre as leg
import numpy.polynomial.polynomial as poly
from numpy.testing import *

P0 = np.array([ 1])
P1 = np.array([ 0,  1])
P2 = np.array([-1,  0,    3])/2
P3 = np.array([ 0, -3,    0,    5])/2
P4 = np.array([ 3,  0,  -30,    0,  35])/8
P5 = np.array([ 0, 15,    0,  -70,   0,   63])/8
P6 = np.array([-5,  0,  105,    0,-315,    0,   231])/16
P7 = np.array([ 0,-35,    0,  315,   0, -693,     0,   429])/16
P8 = np.array([35,  0,-1260,    0,6930,    0,-12012,     0,6435])/128
P9 = np.array([ 0,315,    0,-4620,   0,18018,     0,-25740,   0,12155])/128

Plist = [P0, P1, P2, P3, P4, P5, P6, P7, P8, P9]

def trim(x) :
    return leg.legtrim(x, tol=1e-6)


class TestConstants(TestCase) :

    def test_legdomain(self) :
        assert_equal(leg.legdomain, [-1, 1])

    def test_legzero(self) :
        assert_equal(leg.legzero, [0])

    def test_legone(self) :
        assert_equal(leg.legone, [1])

    def test_legx(self) :
        assert_equal(leg.legx, [0, 1])


class TestArithmetic(TestCase) :
    x = np.linspace(-1, 1, 100)
    y0 = poly.polyval(x, P0)
    y1 = poly.polyval(x, P1)
    y2 = poly.polyval(x, P2)
    y3 = poly.polyval(x, P3)
    y4 = poly.polyval(x, P4)
    y5 = poly.polyval(x, P5)
    y6 = poly.polyval(x, P6)
    y7 = poly.polyval(x, P7)
    y8 = poly.polyval(x, P8)
    y9 = poly.polyval(x, P9)
    y = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9]

    def test_legval(self) :
        def f(x) :
            return x*(x**2 - 1)

        #check empty input
        assert_equal(leg.legval([], [1]).size, 0)

        #check normal input)
        for i in range(10) :
            msg = "At i=%d" % i
            ser = np.zeros
            tgt = self.y[i]
            res = leg.legval(self.x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        #check that shape is preserved
        for i in range(3) :
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(leg.legval(x, [1]).shape, dims)
            assert_equal(leg.legval(x, [1,0]).shape, dims)
            assert_equal(leg.legval(x, [1,0,0]).shape, dims)

    def test_legadd(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = leg.legadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legsub(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = leg.legsub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legmulx(self):
        assert_equal(leg.legmulx([0]), [0])
        assert_equal(leg.legmulx([1]), [0,1])
        for i in range(1, 5):
            tmp = 2*i + 1
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [i/tmp, 0, (i + 1)/tmp]
            assert_equal(leg.legmulx(ser), tgt)

    def test_legmul(self) :
        # check values of result
        for i in range(5) :
            pol1 = [0]*i + [1]
            val1 = leg.legval(self.x, pol1)
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                pol2 = [0]*j + [1]
                val2 = leg.legval(self.x, pol2)
                pol3 = leg.legmul(pol1, pol2)
                val3 = leg.legval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1*val2, err_msg=msg)

    def test_legdiv(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                ci = [0]*i + [1]
                cj = [0]*j + [1]
                tgt = leg.legadd(ci, cj)
                quo, rem = leg.legdiv(tgt, ci)
                res = leg.legadd(leg.legmul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestCalculus(TestCase) :

    def test_legint(self) :
        # check exceptions
        assert_raises(ValueError, leg.legint, [0], .5)
        assert_raises(ValueError, leg.legint, [0], -1)
        assert_raises(ValueError, leg.legint, [0], 1, [0,0])

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = leg.legint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            legpol = leg.poly2leg(pol)
            legint = leg.legint(legpol, m=1, k=[i])
            res = leg.leg2poly(legint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            legpol = leg.poly2leg(pol)
            legint = leg.legint(legpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(leg.legval(-1, legint), i)

        # check single integration with integration constant and scaling
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            legpol = leg.poly2leg(pol)
            legint = leg.legint(legpol, m=1, k=[i], scl=2)
            res = leg.leg2poly(legint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = leg.legint(tgt, m=1)
                res = leg.legint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = leg.legint(tgt, m=1, k=[k])
                res = leg.legint(pol, m=j, k=range(j))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = leg.legint(tgt, m=1, k=[k], lbnd=-1)
                res = leg.legint(pol, m=j, k=range(j), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = leg.legint(tgt, m=1, k=[k], scl=2)
                res = leg.legint(pol, m=j, k=range(j), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_legder(self) :
        # check exceptions
        assert_raises(ValueError, leg.legder, [0], .5)
        assert_raises(ValueError, leg.legder, [0], -1)

        # check that zeroth deriviative does nothing
        for i in range(5) :
            tgt = [1] + [0]*i
            res = leg.legder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = leg.legder(leg.legint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = leg.legder(leg.legint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))


class TestMisc(TestCase) :

    def test_legfromroots(self) :
        res = leg.legfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1,5) :
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            pol = leg.legfromroots(roots)
            res = leg.legval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(leg.leg2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_legroots(self) :
        assert_almost_equal(leg.legroots([1]), [])
        assert_almost_equal(leg.legroots([1, 2]), [-.5])
        for i in range(2,5) :
            tgt = np.linspace(-1, 1, i)
            res = leg.legroots(leg.legfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_legvander(self) :
        # check for 1d x
        x = np.arange(3)
        v = leg.legvander(x, 3)
        assert_(v.shape == (3,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], leg.legval(x, coef))

        # check for 2d x
        x = np.array([[1,2],[3,4],[5,6]])
        v = leg.legvander(x, 3)
        assert_(v.shape == (3,2,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], leg.legval(x, coef))

    def test_legfit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)

        # Test exceptions
        assert_raises(ValueError, leg.legfit, [1],    [1],     -1)
        assert_raises(TypeError,  leg.legfit, [[1]],  [1],      0)
        assert_raises(TypeError,  leg.legfit, [],     [1],      0)
        assert_raises(TypeError,  leg.legfit, [1],    [[[1]]],  0)
        assert_raises(TypeError,  leg.legfit, [1, 2], [1],      0)
        assert_raises(TypeError,  leg.legfit, [1],    [1, 2],   0)
        assert_raises(TypeError,  leg.legfit, [1],    [1],   0, w=[[1]])
        assert_raises(TypeError,  leg.legfit, [1],    [1],   0, w=[1,1])

        # Test fit
        x = np.linspace(0,2)
        y = f(x)
        #
        coef3 = leg.legfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(leg.legval(x, coef3), y)
        #
        coef4 = leg.legfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(leg.legval(x, coef4), y)
        #
        coef2d = leg.legfit(x, np.array([y,y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3,coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = leg.legfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = leg.legfit(x, np.array([yw,yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3,coef3]).T)

    def test_legtrim(self) :
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, leg.legtrim, coef, -1)

        # Test results
        assert_equal(leg.legtrim(coef), coef[:-1])
        assert_equal(leg.legtrim(coef, 1), coef[:-3])
        assert_equal(leg.legtrim(coef, 2), [0])

    def test_legline(self) :
        assert_equal(leg.legline(3,4), [3, 4])

    def test_leg2poly(self) :
        for i in range(10) :
            assert_almost_equal(leg.leg2poly([0]*i + [1]), Plist[i])

    def test_poly2leg(self) :
        for i in range(10) :
            assert_almost_equal(leg.poly2leg(Plist[i]), [0]*i + [1])


def assert_poly_almost_equal(p1, p2):
    assert_almost_equal(p1.coef, p2.coef)
    assert_equal(p1.domain, p2.domain)


class TestLegendreClass(TestCase) :

    p1 = leg.Legendre([1,2,3])
    p2 = leg.Legendre([1,2,3], [0,1])
    p3 = leg.Legendre([1,2])
    p4 = leg.Legendre([2,2,3])
    p5 = leg.Legendre([3,2,3])

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
        tgt = leg.Legendre([2,4,6])
        assert_(self.p1 + self.p1 == tgt)
        assert_(self.p1 + [1,2,3] == tgt)
        assert_([1,2,3] + self.p1 == tgt)

    def test_sub(self) :
        tgt = leg.Legendre([1])
        assert_(self.p4 - self.p1 == tgt)
        assert_(self.p4 - [1,2,3] == tgt)
        assert_([2,2,3] - self.p1 == tgt)

    def test_mul(self) :
        tgt = leg.Legendre([4.13333333, 8.8, 11.23809524, 7.2, 4.62857143])
        assert_poly_almost_equal(self.p1 * self.p1, tgt)
        assert_poly_almost_equal(self.p1 * [1,2,3], tgt)
        assert_poly_almost_equal([1,2,3] * self.p1, tgt)

    def test_floordiv(self) :
        tgt = leg.Legendre([1])
        assert_(self.p4 // self.p1 == tgt)
        assert_(self.p4 // [1,2,3] == tgt)
        assert_([2,2,3] // self.p1 == tgt)

    def test_mod(self) :
        tgt = leg.Legendre([1])
        assert_((self.p4 % self.p1) == tgt)
        assert_((self.p4 % [1,2,3]) == tgt)
        assert_(([2,2,3] % self.p1) == tgt)

    def test_divmod(self) :
        tquo = leg.Legendre([1])
        trem = leg.Legendre([2])
        quo, rem = divmod(self.p5, self.p1)
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod(self.p5, [1,2,3])
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod([3,2,3], self.p1)
        assert_(quo == tquo and rem == trem)

    def test_pow(self) :
        tgt = leg.Legendre([1])
        for i in range(5) :
            res = self.p1**i
            assert_(res == tgt)
            tgt = tgt*self.p1

    def test_call(self) :
        # domain = [-1, 1]
        x = np.linspace(-1, 1)
        tgt = 3*(1.5*x**2 - .5) + 2*x + 1
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
        p = leg.Legendre(coef)
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
        assert_almost_equal(p.coef, leg.legint([1,2,3], 1, 0, scl=.5))
        p = self.p2.integ(lbnd=0)
        assert_almost_equal(p(0), 0)
        p = self.p2.integ(1, 1)
        assert_almost_equal(p.coef, leg.legint([1,2,3], 1, 1, scl=.5))
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.coef, leg.legint([1,2,3], 2, [1,2], scl=.5))

    def test_deriv(self) :
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.deriv(1).coef, self.p2.integ(1, [1]).coef)
        assert_almost_equal(p.deriv(2).coef, self.p2.coef)

    def test_roots(self) :
        p = leg.Legendre(leg.poly2leg([0, -1, 0, 1]), [0, 1])
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
        p = leg.Legendre.fromroots(roots, domain=[0, 1])
        res = p.coef
        tgt = leg.poly2leg([0, -1, 0, 1])
        assert_almost_equal(res, tgt)

    def test_fit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)
        x = np.linspace(0,3)
        y = f(x)

        # test default value of domain
        p = leg.Legendre.fit(x, y, 3)
        assert_almost_equal(p.domain, [0,3])

        # test that fit works in given domains
        p = leg.Legendre.fit(x, y, 3, None)
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [0,3])
        p = leg.Legendre.fit(x, y, 3, [])
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [-1, 1])
        # test that fit accepts weights.
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        p = leg.Legendre.fit(x, yw, 3, w=w)
        assert_almost_equal(p(x), y)

    def test_identity(self) :
        x = np.linspace(0,3)
        p = leg.Legendre.identity()
        assert_almost_equal(p(x), x)
        p = leg.Legendre.identity([1,3])
        assert_almost_equal(p(x), x)
#

if __name__ == "__main__":
    run_module_suite()
