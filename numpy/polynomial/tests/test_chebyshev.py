"""Tests for chebyshev module.

"""
from __future__ import division

import numpy as np
import numpy.polynomial.chebyshev as ch
from numpy.testing import *

def trim(x) :
    return ch.chebtrim(x, tol=1e-6)

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


class TestPrivate(TestCase) :

    def test__cseries_to_zseries(self) :
        for i in range(5) :
            inp = np.array([2] + [1]*i, np.double)
            tgt = np.array([.5]*i + [2] + [.5]*i, np.double)
            res = ch._cseries_to_zseries(inp)
            assert_equal(res, tgt)

    def test__zseries_to_cseries(self) :
        for i in range(5) :
            inp = np.array([.5]*i + [2] + [.5]*i, np.double)
            tgt = np.array([2] + [1]*i, np.double)
            res = ch._zseries_to_cseries(inp)
            assert_equal(res, tgt)


class TestConstants(TestCase) :

    def test_chebdomain(self) :
        assert_equal(ch.chebdomain, [-1, 1])

    def test_chebzero(self) :
        assert_equal(ch.chebzero, [0])

    def test_chebone(self) :
        assert_equal(ch.chebone, [1])

    def test_chebx(self) :
        assert_equal(ch.chebx, [0, 1])


class TestArithmetic(TestCase) :

    def test_chebadd(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = ch.chebadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebsub(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = ch.chebsub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebmulx(self):
        assert_equal(ch.chebmulx([0]), [0])
        assert_equal(ch.chebmulx([1]), [0,1])
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [.5, 0, .5]
            assert_equal(ch.chebmulx(ser), tgt)

    def test_chebmul(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(i + j + 1)
                tgt[i + j] += .5
                tgt[abs(i - j)] += .5
                res = ch.chebmul([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebdiv(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                ci = [0]*i + [1]
                cj = [0]*j + [1]
                tgt = ch.chebadd(ci, cj)
                quo, rem = ch.chebdiv(tgt, ci)
                res = ch.chebadd(ch.chebmul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebval(self) :
        def f(x) :
            return x*(x**2 - 1)

        #check empty input
        assert_equal(ch.chebval([], [1]).size, 0)

        #check normal input)
        for i in range(5) :
            tgt = 1
            res = ch.chebval(1, [0]*i + [1])
            assert_almost_equal(res, tgt)
            tgt = (-1)**i
            res = ch.chebval(-1, [0]*i + [1])
            assert_almost_equal(res, tgt)
            zeros = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            tgt = 0
            res = ch.chebval(zeros,  [0]*i + [1])
            assert_almost_equal(res, tgt)
        x = np.linspace(-1,1)
        tgt = f(x)
        res = ch.chebval(x, [0, -.25, 0, .25])
        assert_almost_equal(res, tgt)

        #check that shape is preserved
        for i in range(3) :
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(ch.chebval(x, [1]).shape, dims)
            assert_equal(ch.chebval(x, [1,0]).shape, dims)
            assert_equal(ch.chebval(x, [1,0,0]).shape, dims)


class TestCalculus(TestCase) :

    def test_chebint(self) :
        # check exceptions
        assert_raises(ValueError, ch.chebint, [0], .5)
        assert_raises(ValueError, ch.chebint, [0], -1)
        assert_raises(ValueError, ch.chebint, [0], 1, [0,0])

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = ch.chebint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            chebpol = ch.poly2cheb(pol)
            chebint = ch.chebint(chebpol, m=1, k=[i])
            res = ch.cheb2poly(chebint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            chebpol = ch.poly2cheb(pol)
            chebint = ch.chebint(chebpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(ch.chebval(-1, chebint), i)

        # check single integration with integration constant and scaling
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            chebpol = ch.poly2cheb(pol)
            chebint = ch.chebint(chebpol, m=1, k=[i], scl=2)
            res = ch.cheb2poly(chebint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = ch.chebint(tgt, m=1)
                res = ch.chebint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = ch.chebint(tgt, m=1, k=[k])
                res = ch.chebint(pol, m=j, k=range(j))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = ch.chebint(tgt, m=1, k=[k], lbnd=-1)
                res = ch.chebint(pol, m=j, k=range(j), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = ch.chebint(tgt, m=1, k=[k], scl=2)
                res = ch.chebint(pol, m=j, k=range(j), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_chebder(self) :
        # check exceptions
        assert_raises(ValueError, ch.chebder, [0], .5)
        assert_raises(ValueError, ch.chebder, [0], -1)

        # check that zeroth deriviative does nothing
        for i in range(5) :
            tgt = [1] + [0]*i
            res = ch.chebder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = ch.chebder(ch.chebint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = ch.chebder(ch.chebint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))


class TestMisc(TestCase) :

    def test_chebfromroots(self) :
        res = ch.chebfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1,5) :
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            tgt = [0]*i + [1]
            res = ch.chebfromroots(roots)*2**(i-1)
            assert_almost_equal(trim(res),trim(tgt))

    def test_chebroots(self) :
        assert_almost_equal(ch.chebroots([1]), [])
        assert_almost_equal(ch.chebroots([1, 2]), [-.5])
        for i in range(2,5) :
            tgt = np.linspace(-1, 1, i)
            res = ch.chebroots(ch.chebfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_chebvander(self) :
        # check for 1d x
        x = np.arange(3)
        v = ch.chebvander(x, 3)
        assert_(v.shape == (3,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], ch.chebval(x, coef))

        # check for 2d x
        x = np.array([[1,2],[3,4],[5,6]])
        v = ch.chebvander(x, 3)
        assert_(v.shape == (3,2,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], ch.chebval(x, coef))

    def test_chebfit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)

        # Test exceptions
        assert_raises(ValueError, ch.chebfit, [1],    [1],     -1)
        assert_raises(TypeError,  ch.chebfit, [[1]],  [1],      0)
        assert_raises(TypeError,  ch.chebfit, [],     [1],      0)
        assert_raises(TypeError,  ch.chebfit, [1],    [[[1]]],  0)
        assert_raises(TypeError,  ch.chebfit, [1, 2], [1],      0)
        assert_raises(TypeError,  ch.chebfit, [1],    [1, 2],   0)
        assert_raises(TypeError,  ch.chebfit, [1],    [1],   0, w=[[1]])
        assert_raises(TypeError,  ch.chebfit, [1],    [1],   0, w=[1,1])

        # Test fit
        x = np.linspace(0,2)
        y = f(x)
        #
        coef3 = ch.chebfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(ch.chebval(x, coef3), y)
        #
        coef4 = ch.chebfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(ch.chebval(x, coef4), y)
        #
        coef2d = ch.chebfit(x, np.array([y,y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3,coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = ch.chebfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = ch.chebfit(x, np.array([yw,yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3,coef3]).T)

    def test_chebtrim(self) :
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, ch.chebtrim, coef, -1)

        # Test results
        assert_equal(ch.chebtrim(coef), coef[:-1])
        assert_equal(ch.chebtrim(coef, 1), coef[:-3])
        assert_equal(ch.chebtrim(coef, 2), [0])

    def test_chebline(self) :
        assert_equal(ch.chebline(3,4), [3, 4])

    def test_cheb2poly(self) :
        for i in range(10) :
            assert_almost_equal(ch.cheb2poly([0]*i + [1]), Tlist[i])

    def test_poly2cheb(self) :
        for i in range(10) :
            assert_almost_equal(ch.poly2cheb(Tlist[i]), [0]*i + [1])

    def test_chebpts1(self):
        #test exceptions
        assert_raises(ValueError, ch.chebpts1, 1.5)
        assert_raises(ValueError, ch.chebpts1, 0)

        #test points
        tgt = [0]
        assert_almost_equal(ch.chebpts1(1), tgt)
        tgt = [-0.70710678118654746, 0.70710678118654746]
        assert_almost_equal(ch.chebpts1(2), tgt)
        tgt = [-0.86602540378443871, 0, 0.86602540378443871]
        assert_almost_equal(ch.chebpts1(3), tgt)
        tgt = [-0.9238795325, -0.3826834323,  0.3826834323,  0.9238795325]
        assert_almost_equal(ch.chebpts1(4), tgt)


    def test_chebpts2(self):
        #test exceptions
        assert_raises(ValueError, ch.chebpts2, 1.5)
        assert_raises(ValueError, ch.chebpts2, 1)

        #test points
        tgt = [-1, 1]
        assert_almost_equal(ch.chebpts2(2), tgt)
        tgt = [-1, 0, 1]
        assert_almost_equal(ch.chebpts2(3), tgt)
        tgt = [-1, -0.5, .5, 1]
        assert_almost_equal(ch.chebpts2(4), tgt)
        tgt = [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
        assert_almost_equal(ch.chebpts2(5), tgt)




class TestChebyshevClass(TestCase) :

    p1 = ch.Chebyshev([1,2,3])
    p2 = ch.Chebyshev([1,2,3], [0,1])
    p3 = ch.Chebyshev([1,2])
    p4 = ch.Chebyshev([2,2,3])
    p5 = ch.Chebyshev([3,2,3])

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
        tgt = ch.Chebyshev([2,4,6])
        assert_(self.p1 + self.p1 == tgt)
        assert_(self.p1 + [1,2,3] == tgt)
        assert_([1,2,3] + self.p1 == tgt)

    def test_sub(self) :
        tgt = ch.Chebyshev([1])
        assert_(self.p4 - self.p1 == tgt)
        assert_(self.p4 - [1,2,3] == tgt)
        assert_([2,2,3] - self.p1 == tgt)

    def test_mul(self) :
        tgt = ch.Chebyshev([7.5, 10., 8., 6., 4.5])
        assert_(self.p1 * self.p1 == tgt)
        assert_(self.p1 * [1,2,3] == tgt)
        assert_([1,2,3] * self.p1 == tgt)

    def test_floordiv(self) :
        tgt = ch.Chebyshev([1])
        assert_(self.p4 // self.p1 == tgt)
        assert_(self.p4 // [1,2,3] == tgt)
        assert_([2,2,3] // self.p1 == tgt)

    def test_mod(self) :
        tgt = ch.Chebyshev([1])
        assert_((self.p4 % self.p1) == tgt)
        assert_((self.p4 % [1,2,3]) == tgt)
        assert_(([2,2,3] % self.p1) == tgt)

    def test_divmod(self) :
        tquo = ch.Chebyshev([1])
        trem = ch.Chebyshev([2])
        quo, rem = divmod(self.p5, self.p1)
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod(self.p5, [1,2,3])
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod([3,2,3], self.p1)
        assert_(quo == tquo and rem == trem)

    def test_pow(self) :
        tgt = ch.Chebyshev([1])
        for i in range(5) :
            res = self.p1**i
            assert_(res == tgt)
            tgt *= self.p1

    def test_call(self) :
        # domain = [-1, 1]
        x = np.linspace(-1, 1)
        tgt = 3*(2*x**2 - 1) + 2*x + 1
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
        p = ch.Chebyshev(coef)
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
        assert_almost_equal(p.coef, ch.chebint([1,2,3], 1, 0, scl=.5))
        p = self.p2.integ(lbnd=0)
        assert_almost_equal(p(0), 0)
        p = self.p2.integ(1, 1)
        assert_almost_equal(p.coef, ch.chebint([1,2,3], 1, 1, scl=.5))
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.coef, ch.chebint([1,2,3], 2, [1,2], scl=.5))

    def test_deriv(self) :
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.deriv(1).coef, self.p2.integ(1, [1]).coef)
        assert_almost_equal(p.deriv(2).coef, self.p2.coef)

    def test_roots(self) :
        p = ch.Chebyshev(ch.poly2cheb([0, -1, 0, 1]), [0, 1])
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
        p = ch.Chebyshev.fromroots(roots, domain=[0, 1])
        res = p.coef
        tgt = ch.poly2cheb([0, -1, 0, 1])
        assert_almost_equal(res, tgt)

    def test_fit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)
        x = np.linspace(0,3)
        y = f(x)

        # test default value of domain
        p = ch.Chebyshev.fit(x, y, 3)
        assert_almost_equal(p.domain, [0,3])

        # test that fit works in given domains
        p = ch.Chebyshev.fit(x, y, 3, None)
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [0,3])
        p = ch.Chebyshev.fit(x, y, 3, [])
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [-1, 1])
        # test that fit accepts weights.
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        p = ch.Chebyshev.fit(x, yw, 3, w=w)
        assert_almost_equal(p(x), y)

    def test_identity(self) :
        x = np.linspace(0,3)
        p = ch.Chebyshev.identity()
        assert_almost_equal(p(x), x)
        p = ch.Chebyshev.identity([1,3])
        assert_almost_equal(p(x), x)
#

if __name__ == "__main__":
    run_module_suite()
