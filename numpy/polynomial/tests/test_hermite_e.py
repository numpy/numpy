"""Tests for hermeendre module.

"""
from __future__ import division

import numpy as np
import numpy.polynomial.hermite_e as herme
import numpy.polynomial.polynomial as poly
from numpy.testing import *

He0 = np.array([ 1 ])
He1 = np.array([ 0 , 1 ])
He2 = np.array([ -1 ,0 , 1 ])
He3 = np.array([ 0 , -3 ,0 , 1 ])
He4 = np.array([ 3 ,0 , -6 ,0 , 1 ])
He5 = np.array([ 0 , 15 ,0 , -10 ,0 , 1 ])
He6 = np.array([ -15 ,0 , 45 ,0 , -15 ,0 , 1 ])
He7 = np.array([ 0 , -105 ,0 , 105 ,0 , -21 ,0 , 1 ])
He8 = np.array([ 105 ,0 , -420 ,0 , 210 ,0 , -28 ,0 , 1 ])
He9 = np.array([ 0 , 945 ,0 , -1260 ,0 , 378 ,0 , -36 ,0 , 1 ])

Helist = [He0, He1, He2, He3, He4, He5, He6, He7, He8, He9]

def trim(x) :
    return herme.hermetrim(x, tol=1e-6)


class TestConstants(TestCase) :

    def test_hermedomain(self) :
        assert_equal(herme.hermedomain, [-1, 1])

    def test_hermezero(self) :
        assert_equal(herme.hermezero, [0])

    def test_hermeone(self) :
        assert_equal(herme.hermeone, [1])

    def test_hermex(self) :
        assert_equal(herme.hermex, [0, 1])


class TestArithmetic(TestCase) :
    x = np.linspace(-3, 3, 100)
    y0 = poly.polyval(x, He0)
    y1 = poly.polyval(x, He1)
    y2 = poly.polyval(x, He2)
    y3 = poly.polyval(x, He3)
    y4 = poly.polyval(x, He4)
    y5 = poly.polyval(x, He5)
    y6 = poly.polyval(x, He6)
    y7 = poly.polyval(x, He7)
    y8 = poly.polyval(x, He8)
    y9 = poly.polyval(x, He9)
    y = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9]

    def test_hermeval(self) :
        def f(x) :
            return x*(x**2 - 1)

        #check empty input
        assert_equal(herme.hermeval([], [1]).size, 0)

        #check normal input)
        for i in range(10) :
            msg = "At i=%d" % i
            ser = np.zeros
            tgt = self.y[i]
            res = herme.hermeval(self.x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        #check that shape is preserved
        for i in range(3) :
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(herme.hermeval(x, [1]).shape, dims)
            assert_equal(herme.hermeval(x, [1,0]).shape, dims)
            assert_equal(herme.hermeval(x, [1,0,0]).shape, dims)

    def test_hermeadd(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = herme.hermeadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermesub(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                tgt = np.zeros(max(i,j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = herme.hermesub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermemulx(self):
        assert_equal(herme.hermemulx([0]), [0])
        assert_equal(herme.hermemulx([1]), [0,1])
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [i, 0, 1]
            assert_equal(herme.hermemulx(ser), tgt)

    def test_hermemul(self) :
        # check values of result
        for i in range(5) :
            pol1 = [0]*i + [1]
            val1 = herme.hermeval(self.x, pol1)
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                pol2 = [0]*j + [1]
                val2 = herme.hermeval(self.x, pol2)
                pol3 = herme.hermemul(pol1, pol2)
                val3 = herme.hermeval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1*val2, err_msg=msg)

    def test_hermediv(self) :
        for i in range(5) :
            for j in range(5) :
                msg = "At i=%d, j=%d" % (i,j)
                ci = [0]*i + [1]
                cj = [0]*j + [1]
                tgt = herme.hermeadd(ci, cj)
                quo, rem = herme.hermediv(tgt, ci)
                res = herme.hermeadd(herme.hermemul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestCalculus(TestCase) :

    def test_hermeint(self) :
        # check exceptions
        assert_raises(ValueError, herme.hermeint, [0], .5)
        assert_raises(ValueError, herme.hermeint, [0], -1)
        assert_raises(ValueError, herme.hermeint, [0], 1, [0,0])

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = herme.hermeint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            hermepol = herme.poly2herme(pol)
            hermeint = herme.hermeint(hermepol, m=1, k=[i])
            res = herme.herme2poly(hermeint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            hermepol = herme.poly2herme(pol)
            hermeint = herme.hermeint(hermepol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(herme.hermeval(-1, hermeint), i)

        # check single integration with integration constant and scaling
        for i in range(5) :
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            hermepol = herme.poly2herme(pol)
            hermeint = herme.hermeint(hermepol, m=1, k=[i], scl=2)
            res = herme.herme2poly(hermeint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = herme.hermeint(tgt, m=1)
                res = herme.hermeint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = herme.hermeint(tgt, m=1, k=[k])
                res = herme.hermeint(pol, m=j, k=range(j))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = herme.hermeint(tgt, m=1, k=[k], lbnd=-1)
                res = herme.hermeint(pol, m=j, k=range(j), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5) :
            for j in range(2,5) :
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j) :
                    tgt = herme.hermeint(tgt, m=1, k=[k], scl=2)
                res = herme.hermeint(pol, m=j, k=range(j), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermeder(self) :
        # check exceptions
        assert_raises(ValueError, herme.hermeder, [0], .5)
        assert_raises(ValueError, herme.hermeder, [0], -1)

        # check that zeroth deriviative does nothing
        for i in range(5) :
            tgt = [1] + [0]*i
            res = herme.hermeder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = herme.hermeder(herme.hermeint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5) :
            for j in range(2,5) :
                tgt = [1] + [0]*i
                res = herme.hermeder(herme.hermeint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))


class TestMisc(TestCase) :

    def test_hermefromroots(self) :
        res = herme.hermefromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1,5) :
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            pol = herme.hermefromroots(roots)
            res = herme.hermeval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(herme.herme2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_hermeroots(self) :
        assert_almost_equal(herme.hermeroots([1]), [])
        assert_almost_equal(herme.hermeroots([1, 1]), [-.5])
        for i in range(2,5) :
            tgt = np.linspace(-1, 1, i)
            res = herme.hermeroots(herme.hermefromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_hermevander(self) :
        # check for 1d x
        x = np.arange(3)
        v = herme.hermevander(x, 3)
        assert_(v.shape == (3,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], herme.hermeval(x, coef))

        # check for 2d x
        x = np.array([[1,2],[3,4],[5,6]])
        v = herme.hermevander(x, 3)
        assert_(v.shape == (3,2,4))
        for i in range(4) :
            coef = [0]*i + [1]
            assert_almost_equal(v[...,i], herme.hermeval(x, coef))

    def test_hermefit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)

        # Test exceptions
        assert_raises(ValueError, herme.hermefit, [1],    [1],     -1)
        assert_raises(TypeError,  herme.hermefit, [[1]],  [1],      0)
        assert_raises(TypeError,  herme.hermefit, [],     [1],      0)
        assert_raises(TypeError,  herme.hermefit, [1],    [[[1]]],  0)
        assert_raises(TypeError,  herme.hermefit, [1, 2], [1],      0)
        assert_raises(TypeError,  herme.hermefit, [1],    [1, 2],   0)
        assert_raises(TypeError,  herme.hermefit, [1],    [1],   0, w=[[1]])
        assert_raises(TypeError,  herme.hermefit, [1],    [1],   0, w=[1,1])

        # Test fit
        x = np.linspace(0,2)
        y = f(x)
        #
        coef3 = herme.hermefit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(herme.hermeval(x, coef3), y)
        #
        coef4 = herme.hermefit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(herme.hermeval(x, coef4), y)
        #
        coef2d = herme.hermefit(x, np.array([y,y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3,coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = herme.hermefit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = herme.hermefit(x, np.array([yw,yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3,coef3]).T)

    def test_hermetrim(self) :
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, herme.hermetrim, coef, -1)

        # Test results
        assert_equal(herme.hermetrim(coef), coef[:-1])
        assert_equal(herme.hermetrim(coef, 1), coef[:-3])
        assert_equal(herme.hermetrim(coef, 2), [0])

    def test_hermeline(self) :
        assert_equal(herme.hermeline(3,4), [3, 4])

    def test_herme2poly(self) :
        for i in range(10) :
            assert_almost_equal(herme.herme2poly([0]*i + [1]), Helist[i])

    def test_poly2herme(self) :
        for i in range(10) :
            assert_almost_equal(herme.poly2herme(Helist[i]), [0]*i + [1])


def assert_poly_almost_equal(p1, p2):
    assert_almost_equal(p1.coef, p2.coef)
    assert_equal(p1.domain, p2.domain)


class TestHermiteEClass(TestCase) :

    p1 = herme.HermiteE([1,2,3])
    p2 = herme.HermiteE([1,2,3], [0,1])
    p3 = herme.HermiteE([1,2])
    p4 = herme.HermiteE([2,2,3])
    p5 = herme.HermiteE([3,2,3])

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
        tgt = herme.HermiteE([2,4,6])
        assert_(self.p1 + self.p1 == tgt)
        assert_(self.p1 + [1,2,3] == tgt)
        assert_([1,2,3] + self.p1 == tgt)

    def test_sub(self) :
        tgt = herme.HermiteE([1])
        assert_(self.p4 - self.p1 == tgt)
        assert_(self.p4 - [1,2,3] == tgt)
        assert_([2,2,3] - self.p1 == tgt)

    def test_mul(self) :
        tgt = herme.HermiteE([ 23.,  28.,  46.,  12.,   9.])
        assert_poly_almost_equal(self.p1 * self.p1, tgt)
        assert_poly_almost_equal(self.p1 * [1,2,3], tgt)
        assert_poly_almost_equal([1,2,3] * self.p1, tgt)

    def test_floordiv(self) :
        tgt = herme.HermiteE([1])
        assert_(self.p4 // self.p1 == tgt)
        assert_(self.p4 // [1,2,3] == tgt)
        assert_([2,2,3] // self.p1 == tgt)

    def test_mod(self) :
        tgt = herme.HermiteE([1])
        assert_((self.p4 % self.p1) == tgt)
        assert_((self.p4 % [1,2,3]) == tgt)
        assert_(([2,2,3] % self.p1) == tgt)

    def test_divmod(self) :
        tquo = herme.HermiteE([1])
        trem = herme.HermiteE([2])
        quo, rem = divmod(self.p5, self.p1)
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod(self.p5, [1,2,3])
        assert_(quo == tquo and rem == trem)
        quo, rem = divmod([3,2,3], self.p1)
        assert_(quo == tquo and rem == trem)

    def test_pow(self) :
        tgt = herme.HermiteE([1])
        for i in range(5) :
            res = self.p1**i
            assert_(res == tgt)
            tgt = tgt*self.p1

    def test_call(self) :
        # domain = [-1, 1]
        x = np.linspace(-1, 1)
        tgt = 3*(x**2 - 1) + 2*(x) + 1
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
        p = herme.HermiteE(coef)
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
        assert_almost_equal(p.coef, herme.hermeint([1,2,3], 1, 0, scl=.5))
        p = self.p2.integ(lbnd=0)
        assert_almost_equal(p(0), 0)
        p = self.p2.integ(1, 1)
        assert_almost_equal(p.coef, herme.hermeint([1,2,3], 1, 1, scl=.5))
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.coef, herme.hermeint([1,2,3], 2, [1,2], scl=.5))

    def test_deriv(self) :
        p = self.p2.integ(2, [1, 2])
        assert_almost_equal(p.deriv(1).coef, self.p2.integ(1, [1]).coef)
        assert_almost_equal(p.deriv(2).coef, self.p2.coef)

    def test_roots(self) :
        p = herme.HermiteE(herme.poly2herme([0, -1, 0, 1]), [0, 1])
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
        p = herme.HermiteE.fromroots(roots, domain=[0, 1])
        res = p.coef
        tgt = herme.poly2herme([0, -1, 0, 1])
        assert_almost_equal(res, tgt)

    def test_fit(self) :
        def f(x) :
            return x*(x - 1)*(x - 2)
        x = np.linspace(0,3)
        y = f(x)

        # test default value of domain
        p = herme.HermiteE.fit(x, y, 3)
        assert_almost_equal(p.domain, [0,3])

        # test that fit works in given domains
        p = herme.HermiteE.fit(x, y, 3, None)
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [0,3])
        p = herme.HermiteE.fit(x, y, 3, [])
        assert_almost_equal(p(x), y)
        assert_almost_equal(p.domain, [-1, 1])
        # test that fit accepts weights.
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        p = herme.HermiteE.fit(x, yw, 3, w=w)
        assert_almost_equal(p(x), y)

    def test_identity(self) :
        x = np.linspace(0,3)
        p = herme.HermiteE.identity()
        assert_almost_equal(p(x), x)
        p = herme.HermiteE.identity([1,3])
        assert_almost_equal(p(x), x)


if __name__ == "__main__":
    run_module_suite()
