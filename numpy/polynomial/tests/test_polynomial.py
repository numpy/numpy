"""Tests for polynomial module.

"""
from __future__ import division

import numpy as np
import numpy.polynomial.polynomial as poly
from numpy.testing import *

def trim(x) :
    return poly.polytrim(x, tol=1e-6)

T0 = [ 1]
T1 = [ 0,  1]
T2 = [-1,  0,   2]
T3 = [ 0, -3,   0,    4]
T4 = [ 1,  0,  -8,    0,   8]
T5 = [ 0,  5,   0,  -20,   0,   16]
T6 = [-1,  0,  18,    0, -48,    0,   32]
T7 = [ 0, -7,   0,   56,   0, -112,    0,   64]
T8 = [ 1,  0, -32,    0, 160,    0, -256,    0, 128]
T9 = [ 0,  9,   0, -120,   0,  432,    0, -576,   0, 256]

Tlist = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]


class TestConstants(TestCase) :

    def test_polydomain(self) :
        assert_equal(poly.polydomain, [-1, 1])

    def test_polyzero(self) :
        assert_equal(poly.polyzero, [0])

    def test_polyone(self) :
        assert_equal(poly.polyone, [1])

    def test_polyx(self) :
        assert_equal(poly.polyx, [0, 1])


class TestArithmetic(TestCase) :

    def test_polyadd(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = poly.polyadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polysub(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = poly.polysub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polymulx(self):
        assert_equal(poly.polymulx([0]), [0])
        assert_equal(poly.polymulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i + 1) + [1]
            assert_equal(poly.polymulx(ser), tgt)

    def test_polymul(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(i + j + 1)
                tgt[i + j] += 1
                res = poly.polymul([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polydiv(self) :
        # check zero division
        assert_raises(ZeroDivisionError, poly.polydiv, [1], [0])

        # check scalar division
        quo, rem = poly.polydiv([2],[2])
        assert_equal((quo, rem), (1, 0))
        quo, rem = poly.polydiv([2,2],[2])
        assert_equal((quo, rem), ((1,1), 0))

        # check rest.
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                ci = [0]*i + [1,2]
                cj = [0]*j + [1,2]
                tgt = poly.polyadd(ci, cj)
                quo, rem = poly.polydiv(tgt, ci)
                res = poly.polyadd(poly.polymul(quo, ci), rem)
                assert_equal(res, tgt, err_msg=msg)

    def test_polyval(self) :
        def f(x) :
            return x*(x**2 - 1)

        #check empty input
        assert_equal(poly.polyval([], [1]).size, 0)

        #check normal input)
        x = np.linspace(-1,1)
        for i in range(5) :
            tgt = x**i
            res = poly.polyval(x, [0]*i + [1])
            assert_almost_equal(res, tgt)
        tgt = f(x)
        res = poly.polyval(x, [0, -1, 0, 1])
        assert_almost_equal(res, tgt)

        #check that shape is preserved
        for i in range(3) :
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(poly.polyval(x, [1]).shape, dims)
            assert_equal(poly.polyval(x, [1,0]).shape, dims)
            assert_equal(poly.polyval(x, [1,0,0]).shape, dims)


class TestCalculus(TestCase) :

    def test_polyint(self) :
        # check exceptions
        assert_raises(ValueError, poly.polyint, [0], .5)
        assert_raises(ValueError, poly.polyint, [0], -1)
        assert_raises(ValueError, poly.polyint, [0], 1, [0,0])

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = poly.polyint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            res = poly.polyint(pol, m=1, k=[i])
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            res = poly.polyint(pol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(poly.polyval(-1, res), i)

        # check single integration with integration constant and scaling
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            res = poly.polyint(pol, m=1, k=[i], scl=2)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = poly.polyint(tgt, m=1)
                res = poly.polyint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = poly.polyint(tgt, m=1, k=[k])
                res = poly.polyint(pol, m=j, k=range(j))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = poly.polyint(tgt, m=1, k=[k], lbnd=-1)
                res = poly.polyint(pol, m=j, k=range(j), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = poly.polyint(tgt, m=1, k=[k], scl=2)
                res = poly.polyint(pol, m=j, k=range(j), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_polyder(self) :
        # check exceptions
        assert_raises(ValueError, poly.polyder, [0], .5)
        assert_raises(ValueError, poly.polyder, [0], -1)

        # check that zeroth deriviative does nothing
        for i in range(5) :
            tgt = [1] + [0]*i
            res = poly.polyder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = poly.polyder(poly.polyint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = poly.polyder(poly.polyint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))


class TestMisc(TestCase) :

    def test_polyfromroots(self) :
        res = poly.polyfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1,5) :
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            tgt = Tlist[i]
            res = poly.polyfromroots(roots)*2**(i-1)
            assert_almost_equal(trim(res),trim(tgt))

    def test_polyroots(self) :
        assert_almost_equal(poly.polyroots([1]), [])
        assert_almost_equal(poly.polyroots([1, 2]), [-.5])
        for i in range(2,5) :
            tgt = np.linspace(-1, 1, i)
            res = poly.polyroots(poly.polyfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_polyvander(self) :
        # check for 1d x
        x = np.arange(3)
        v = poly.polyvander(x, 3)
        assert_(v.shape == (3,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], poly.polyval(x, coef))

        # check for 2d x
        x = np.array([[1,2],[3,4],[5,6]])
        v = poly.polyvander(x, 3)
        assert_(v.shape == (3,2,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], poly.polyval(x, coef))

    def test_polyfit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)

        # Test exceptions
        assert_raises(ValueError, poly.polyfit, [1],    [1],     -1)
        assert_raises(TypeError,  poly.polyfit, [[1]],  [1],      0)
        assert_raises(TypeError,  poly.polyfit, [],     [1],      0)
        assert_raises(TypeError,  poly.polyfit, [1],    [[[1]]],  0)
        assert_raises(TypeError,  poly.polyfit, [1, 2], [1],      0)
        assert_raises(TypeError,  poly.polyfit, [1],    [1, 2],   0)
        assert_raises(TypeError,  poly.polyfit, [1],    [1],   0, w=[[1]])
        assert_raises(TypeError,  poly.polyfit, [1],    [1],   0, w=[1,1])

        # Test fit
        x = np.linspace(0,2)
        y = f(x)
        #
        coef3 = poly.polyfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(poly.polyval(x, coef3), y)
        #
        coef4 = poly.polyfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(poly.polyval(x, coef4), y)
        #
        coef2d = poly.polyfit(x, np.array([y,y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3,coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        wcoef3 = poly.polyfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = poly.polyfit(x, np.array([yw,yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3,coef3]).T)

    def test_polytrim(self) :
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, poly.polytrim, coef, -1)

        # Test results
        assert_equal(poly.polytrim(coef), coef[:-1])
        assert_equal(poly.polytrim(coef, 1), coef[:-3])
        assert_equal(poly.polytrim(coef, 2), [0])

    def test_polyline(self) :
        assert_equal(poly.polyline(3,4), [3, 4])

class TestPolynomialClass(TestCase) :

    p1 = poly.Polynomial([1,2,3])
    p2 = poly.Polynomial([1,2,3], [0,1])
    p3 = poly.Polynomial([1,2])
    p4 = poly.Polynomial([2,2,3])
    p5 = poly.Polynomial([3,2,3])

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
        tgt = poly.Polynomial([2,4,6])
        assert_(self.p1 + self.p1 == tgt)
        assert_(self.p1 + [1,2,3] == tgt)
        assert_([1,2,3] + self.p1 == tgt)

    def test_sub(self) :
        tgt = poly.Polynomial([1])
        assert_(self.p4 - self.p1 == tgt)
        assert_(self.p4 - [1,2,3] == tgt)
        assert_([2,2,3] - self.p1 == tgt)

    def test_mul(self) :
        tgt = poly.Polynomial([1,4,10,12,9])
        assert_(self.p1 * self.p1 == tgt)
        assert_(self.p1 * [1,2,3] == tgt)
        assert_([1,2,3] * self.p1 == tgt)

    def test_floordiv(self) :
        tgt = poly.Polynomial([1])
        assert_(self.p4 // self.p1 == tgt)
        assert_(self.p4 // [1,2,3] == tgt)
        assert_([2,2,3] // self.p1 == tgt)

    def test_mod(self) :
        tgt = poly.Polynomial([1])
        assert_((self.p4 % self.p1) == tgt)
        assert_((self.p4 % [1,2,3]) == tgt)
        assert_(([2,2,3] % self.p1) == tgt)

    def test_divmod(self) :
        tquo = poly.Polynomial([1])
        trem = poly.Polynomial([2])
        quo, rem = divmod(self.p5, self.p1)
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod(self.p5, [1,2,3])
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod([3,2,3], self.p1)
        assert_(quo == tquo and rem == trem)

    def test_pow(self) :
        tgt = poly.Polynomial([1])
        for i in range(5) :
            res = self.p1**i
            assert_(res == tgt)
            tgt *= self.p1

    def test_call(self) :
        # domain = [-1, 1]
        x = np.linspace(-1, 1)
        tgt = (3*x + 2)*x + 1
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
        p = poly.Polynomial(coef)
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
        assert_almost_equal(p.coef, poly.polyint([1,2,3], 1, 0, scl=.5))
        p = self.p2.integ(lbnd=0)
        assert_almost_equal(p(0), 0)
        p = self.p2.integ(1, 1)
        assert_almost_equal(p.coef, poly.polyint([1,2,3], 1, 1, scl=.5))
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.coef, poly.polyint([1,2,3], 2, [1, 2], scl=.5))

    def test_deriv(self) :
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.deriv(1).coef, self.p2.integ(1, [1]).coef)
        assert_almost_equal(p.deriv(2).coef, self.p2.coef)

    def test_roots(self) :
        p = poly.Polynomial([0, -1, 0, 1], [0, 1])
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
        p = poly.Polynomial.fromroots(roots, domain=[0, 1])
        res = p.coef
        tgt = [0, -1, 0, 1]
        assert_almost_equal(res, tgt)

    def test_fit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)
        x = np.linspace(0,3)
        y = f(x)

        # test default value of domain
        p = poly.Polynomial.fit(x, y, 3)
        assert_almost_equal(p.domain, [0,3])

        # test that fit works in given domains
        p = poly.Polynomial.fit(x, y, 3, None)
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [0,3])
        p = poly.Polynomial.fit(x, y, 3, [])
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [-1, 1])
        # test that fit accepts weights.
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        p = poly.Polynomial.fit(x, yw, 3, w=w)
        assert_almost_equal(p(x), y)

    def test_identity(self) :
        x = np.linspace(0,3)
        p = poly.Polynomial.identity()
        assert_almost_equal(p(x), x)
        p = poly.Polynomial.identity([1,3])
        assert_almost_equal(p(x), x)
#

if __name__ == "__main__":
    run_module_suite()
