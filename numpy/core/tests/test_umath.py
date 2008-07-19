from numpy.testing import *
import numpy.core.umath as ncu
import numpy as np
import nose
from numpy import inf, nan, pi

class TestDivision(TestCase):
    def test_division_int(self):
        # int division should return the floor of the result, a la Python
        x = np.array([5, 10, 90, 100, -5, -10, -90, -100, -120])
        assert_equal(x / 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        assert_equal(x // 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        assert_equal(x % 100, [5, 10, 90, 0, 95, 90, 10, 0, 80])

class TestPower(TestCase):
    def test_power_float(self):
        x = np.array([1., 2., 3.])
        assert_equal(x**0, [1., 1., 1.])
        assert_equal(x**1, x)
        assert_equal(x**2, [1., 4., 9.])
        y = x.copy()
        y **= 2
        assert_equal(y, [1., 4., 9.])
        assert_almost_equal(x**(-1), [1., 0.5, 1./3])
        assert_almost_equal(x**(0.5), [1., ncu.sqrt(2), ncu.sqrt(3)])

    def test_power_complex(self):
        x = np.array([1+2j, 2+3j, 3+4j])
        assert_equal(x**0, [1., 1., 1.])
        assert_equal(x**1, x)
        assert_equal(x**2, [-3+4j, -5+12j, -7+24j])
        assert_almost_equal(x**(-1), [1/(1+2j), 1/(2+3j), 1/(3+4j)])
        assert_almost_equal(x**(-3), [(-11+2j)/125, (-46-9j)/2197,
                                      (-117-44j)/15625])
        assert_almost_equal(x**(0.5), [ncu.sqrt(1+2j), ncu.sqrt(2+3j),
                                       ncu.sqrt(3+4j)])
        assert_almost_equal(x**14, [-76443+16124j, 23161315+58317492j,
                                    5583548873 +  2465133864j])

class TestLog1p(TestCase):
    def test_log1p(self):
        assert_almost_equal(ncu.log1p(0.2), ncu.log(1.2))
        assert_almost_equal(ncu.log1p(1e-6), ncu.log(1+1e-6))

class TestExpm1(TestCase):
    def test_expm1(self):
        assert_almost_equal(ncu.expm1(0.2), ncu.exp(0.2)-1)
        assert_almost_equal(ncu.expm1(1e-6), ncu.exp(1e-6)-1)

class TestMaximum(TestCase):
    def test_reduce_complex(self):
        assert_equal(ncu.maximum.reduce([1,2j]),1)
        assert_equal(ncu.maximum.reduce([1+3j,2j]),1+3j)

class TestMinimum(TestCase):
    def test_reduce_complex(self):
        assert_equal(ncu.minimum.reduce([1,2j]),2j)

class TestFloatingPoint(TestCase):
    def test_floating_point(self):
        assert_equal(ncu.FLOATING_POINT_SUPPORT, 1)

class TestDegrees(TestCase):
    def test_degrees(self):
        assert_almost_equal(ncu.degrees(np.pi), 180.0)
        assert_almost_equal(ncu.degrees(-0.5*np.pi), -90.0)

class TestRadians(TestCase):
    def test_radians(self):
        assert_almost_equal(ncu.radians(180.0), np.pi)
        assert_almost_equal(ncu.radians(-90.0), -0.5*np.pi)

class TestSpecialMethods(TestCase):
    def test_wrap(self):
        class with_wrap(object):
            def __array__(self):
                return np.zeros(1)
            def __array_wrap__(self, arr, context):
                r = with_wrap()
                r.arr = arr
                r.context = context
                return r
        a = with_wrap()
        x = ncu.minimum(a, a)
        assert_equal(x.arr, np.zeros(1))
        func, args, i = x.context
        self.failUnless(func is ncu.minimum)
        self.failUnlessEqual(len(args), 2)
        assert_equal(args[0], a)
        assert_equal(args[1], a)
        self.failUnlessEqual(i, 0)

    def test_old_wrap(self):
        class with_wrap(object):
            def __array__(self):
                return np.zeros(1)
            def __array_wrap__(self, arr):
                r = with_wrap()
                r.arr = arr
                return r
        a = with_wrap()
        x = ncu.minimum(a, a)
        assert_equal(x.arr, np.zeros(1))

    def test_priority(self):
        class A(object):
            def __array__(self):
                return np.zeros(1)
            def __array_wrap__(self, arr, context):
                r = type(self)()
                r.arr = arr
                r.context = context
                return r
        class B(A):
            __array_priority__ = 20.
        class C(A):
            __array_priority__ = 40.
        x = np.zeros(1)
        a = A()
        b = B()
        c = C()
        f = ncu.minimum
        self.failUnless(type(f(x,x)) is np.ndarray)
        self.failUnless(type(f(x,a)) is A)
        self.failUnless(type(f(x,b)) is B)
        self.failUnless(type(f(x,c)) is C)
        self.failUnless(type(f(a,x)) is A)
        self.failUnless(type(f(b,x)) is B)
        self.failUnless(type(f(c,x)) is C)

        self.failUnless(type(f(a,a)) is A)
        self.failUnless(type(f(a,b)) is B)
        self.failUnless(type(f(b,a)) is B)
        self.failUnless(type(f(b,b)) is B)
        self.failUnless(type(f(b,c)) is C)
        self.failUnless(type(f(c,b)) is C)
        self.failUnless(type(f(c,c)) is C)

        self.failUnless(type(ncu.exp(a) is A))
        self.failUnless(type(ncu.exp(b) is B))
        self.failUnless(type(ncu.exp(c) is C))

    def test_failing_wrap(self):
        class A(object):
            def __array__(self):
                return np.zeros(1)
            def __array_wrap__(self, arr, context):
                raise RuntimeError
        a = A()
        self.failUnlessRaises(RuntimeError, ncu.maximum, a, a)

    def test_array_with_context(self):
        class A(object):
            def __array__(self, dtype=None, context=None):
                func, args, i = context
                self.func = func
                self.args = args
                self.i = i
                return np.zeros(1)
        class B(object):
            def __array__(self, dtype=None):
                return np.zeros(1, dtype)
        class C(object):
            def __array__(self):
                return np.zeros(1)
        a = A()
        ncu.maximum(np.zeros(1), a)
        self.failUnless(a.func is ncu.maximum)
        assert_equal(a.args[0], 0)
        self.failUnless(a.args[1] is a)
        self.failUnless(a.i == 1)
        assert_equal(ncu.maximum(a, B()), 0)
        assert_equal(ncu.maximum(a, C()), 0)


class TestChoose(TestCase):
    def test_mixed(self):
        c = np.array([True,True])
        a = np.array([True,True])
        assert_equal(np.choose(c, (a, 1)), np.array([1,1]))


class TestComplexFunctions(object):
    funcs = [np.arcsin , np.arccos , np.arctan, np.arcsinh, np.arccosh,
             np.arctanh, np.sin    , np.cos   , np.tan    , np.exp,
             np.log    , np.sqrt   , np.log10,  np.log1p]

    def test_it(self):
        for f in self.funcs:
            if f is np.arccosh :
                x = 1.5
            else :
                x = .5
            fr = f(x)
            fz = f(np.complex(x))
            assert_almost_equal(fz.real, fr, err_msg='real part %s'%f)
            assert_almost_equal(fz.imag, 0., err_msg='imag part %s'%f)

    def test_precisions_consistent(self) :
        z = 1 + 1j
        for f in self.funcs :
            fcf = f(np.csingle(z))
            fcd  = f(np.cdouble(z))
            fcl = f(np.clongdouble(z))
            assert_almost_equal(fcf, fcd, decimal=6, err_msg='fch-fcd %s'%f)
            assert_almost_equal(fcl, fcd, decimal=15, err_msg='fch-fcl %s'%f)

    def test_branch_cuts(self):
        # check branch cuts and continuity on them
        yield _check_branch_cut, np.log,   -0.5, 1j, 1, -1, True
        yield _check_branch_cut, np.log10, -0.5, 1j, 1, -1, True
        yield _check_branch_cut, np.log1p, -1.5, 1j, 1, -1, True
        yield _check_branch_cut, np.sqrt,  -0.5, 1j, 1, -1
        
        yield _check_branch_cut, np.arcsin, [ -2, 2],   [1j, -1j], 1, -1
        yield _check_branch_cut, np.arccos, [ -2, 2],   [1j, -1j], 1, -1
        yield _check_branch_cut, np.arctan, [-2j, 2j],  [1,  -1 ], -1, 1
        
        yield _check_branch_cut, np.arcsinh, [-2j,  2j], [-1,   1], -1, 1
        yield _check_branch_cut, np.arccosh, [ -1, 0.5], [1j,  1j], 1, -1
        yield _check_branch_cut, np.arctanh, [ -2,   2], [1j, -1j], 1, -1

        # check against bogus branch cuts: assert continuity between quadrants
        yield _check_branch_cut, np.arcsin, [-2j, 2j], [ 1,  1], 1, 1
        yield _check_branch_cut, np.arccos, [-2j, 2j], [ 1,  1], 1, 1
        yield _check_branch_cut, np.arctan, [ -2,  2], [1j, 1j], 1, 1

        yield _check_branch_cut, np.arcsinh, [ -2,  2, 0], [1j, 1j, 1 ], 1, 1
        yield _check_branch_cut, np.arccosh, [-2j, 2j, 2], [1,  1,  1j], 1, 1
        yield _check_branch_cut, np.arctanh, [-2j, 2j, 0], [1,  1,  1j], 1, 1

    def test_branch_cuts_failing(self):
        # XXX: signed zeros are not OK for sqrt or for the arc* functions
        yield _check_branch_cut, np.sqrt,  -0.5, 1j, 1, -1, True
        yield _check_branch_cut, np.arcsin, [ -2, 2],   [1j, -1j], 1, -1, True
        yield _check_branch_cut, np.arccos, [ -2, 2],   [1j, -1j], 1, -1, True
        yield _check_branch_cut, np.arctan, [-2j, 2j],  [1,  -1 ], -1, 1, True
        yield _check_branch_cut, np.arcsinh, [-2j,  2j], [-1,   1], -1, 1, True
        yield _check_branch_cut, np.arccosh, [ -1, 0.5], [1j,  1j], 1, -1, True
        yield _check_branch_cut, np.arctanh, [ -2,   2], [1j, -1j], 1, -1, True
    test_branch_cuts_failing = dec.skipknownfailure(test_branch_cuts_failing)
        
    def test_against_cmath(self):
        import cmath, sys

        # cmath.asinh is broken in some versions of Python, see
        # http://bugs.python.org/issue1381
        broken_cmath_asinh = False
        if sys.version_info < (2,5,3):
            broken_cmath_asinh = True
        
        points = [-2, 2j, 2, -2j, -1-1j, -1+1j, +1-1j, +1+1j]
        name_map = {'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
                    'arcsinh': 'asinh', 'arccosh': 'acosh', 'arctanh': 'atanh'}
        atol = 4*np.finfo(np.complex).eps
        for func in self.funcs:
            fname = func.__name__.split('.')[-1]
            cname = name_map.get(fname, fname)
            try: cfunc = getattr(cmath, cname)
            except AttributeError: continue
            for p in points:
                a = complex(func(np.complex_(p)))
                b = cfunc(p)
                
                if cname == 'asinh' and broken_cmath_asinh:
                    continue 

                assert abs(a - b) < atol, "%s %s: %s; cmath: %s"%(fname,p,a,b)

class TestC99(object):
    """Check special functions at special points against the C99 standard"""
    # NB: inherits from object instead of TestCase since using test generators
    
    #
    # Non-conforming results are with XXX added to the exception field.
    #
    
    def test_clog(self):
        for p, v, e in [
            ((-0., 0.), (-inf, pi), 'divide'),
            ((+0., 0.), (-inf, 0.), 'divide'),
            ((1., inf), (inf, pi/2), ''),
            ((1., nan), (nan, nan), ''),
            ((-inf, 1.), (inf, pi), ''),
            ((inf, 1.), (inf, 0.), ''),
            ((-inf, inf), (inf, 3*pi/4), ''),
            ((inf, inf), (inf, pi/4), ''),
            ((inf, nan), (inf, nan), ''),
            ((-inf, nan), (inf, nan), ''),
            ((nan, 0.), (nan, nan), ''),
            ((nan, 1.), (nan, nan), ''),
            ((nan, inf), (inf, nan), ''),
            ((+nan, nan), (nan, nan), ''),
        ]:
            yield self._check, np.log, p, v, e
    
    def test_csqrt(self):
        for p, v, e in [
            ((-0., 0.), (0.,0.),  'XXX'), # now (-0., 0.)
            ((0., 0.), (0.,0.),  ''),
            ((1., inf), (inf,inf), 'XXX invalid'), # now (inf, nan)
            ((nan, inf), (inf,inf), 'XXX'), # now (nan, nan)
            ((-inf, 1.), (0.,inf), ''),
            ((inf, 1.), (inf,0.), ''),
            ((-inf,nan), (nan, -inf), ''), # could also be +inf
            ((inf, nan), (inf, nan),  ''),
            ((nan, 1.), (nan, nan), ''),
            ((nan, nan), (nan, nan), ''),
        ]:
            yield self._check, np.sqrt, p, v, e

    def test_cacos(self):
        for p, v, e in [
            ((0., 0.), (pi/2, -0.), 'XXX'), # now (-0., 0.)
            ((-0., 0.), (pi/2, -0.), ''),
            ((0., nan), (pi/2, nan), 'XXX'), # now (nan, nan)
            ((-0., nan), (pi/2, nan), 'XXX'), # now (nan, nan)
            ((1., inf), (pi/2, -inf), 'XXX'), # now (nan, -inf)
            ((1., nan), (nan, nan), ''),
            ((-inf, 1.), (pi, -inf), 'XXX'), # now (nan, -inf)
            ((inf, 1.), (0., -inf), 'XXX'), # now (nan, -inf)
            ((-inf, inf), (3*pi/4, -inf), 'XXX'), # now (nan, nan)
            ((inf, inf), (pi/4, -inf), 'XXX'), # now (nan, nan)
            ((inf, nan), (nan, +-inf), 'XXX'), # now (nan, nan)
            ((-inf, nan), (nan, +-inf), 'XXX'), # now: (nan, nan)
            ((nan, 1.), (nan, nan), ''),
            ((nan, inf), (nan, -inf), 'XXX'), # now: (nan, nan)
            ((nan, nan), (nan, nan), ''),
        ]:
            yield self._check, np.arccos, p, v, e

    def test_cacosh(self):
        for p, v, e in [
            ((0., 0), (0, pi/2), ''),
            ((-0., 0), (0, pi/2), ''),
            ((1., inf), (inf, pi/2), 'XXX'), # now: (nan, nan)
            ((1., nan), (nan, nan), ''),
            ((-inf, 1.), (inf, pi), 'XXX'), # now: (inf, nan)
            ((inf, 1.), (inf, 0.), 'XXX'), # now: (inf, nan)
            ((-inf, inf), (inf, 3*pi/4), 'XXX'), # now: (nan, nan)
            ((inf, inf), (inf, pi/4), 'XXX'), # now: (nan, nan)
            ((inf, nan), (inf, nan), 'XXX'), # now: (nan, nan)
            ((-inf, nan), (inf, nan), 'XXX'), # now: (nan, nan)
            ((nan, 1.), (nan, nan), ''),
            ((nan, inf), (inf, nan), 'XXX'), # now: (nan, nan)
            ((nan, nan), (nan, nan), '')
        ]:
            yield self._check, np.arccosh, p, v, e

    def test_casinh(self):
        for p, v, e in [
            ((0., 0), (0, 0), ''),
            ((1., inf), (inf, pi/2), 'XXX'), # now: (inf, nan)
            ((1., nan), (nan, nan), ''),
            ((inf, 1.), (inf, 0.), 'XXX'), # now: (inf, nan)
            ((inf, inf), (inf, pi/4), 'XXX'), # now: (nan, nan)
            ((inf, nan), (nan, nan), 'XXX'), # now: (nan, nan)
            ((nan, 0.), (nan, 0.), 'XXX'), # now: (nan, nan)
            ((nan, 1.), (nan, nan), ''),
            ((nan, inf), (+-inf, nan), 'XXX'), # now: (nan, nan)
            ((nan, nan), (nan, nan), ''),
        ]:
            yield self._check, np.arcsinh, p, v, e

    def test_catanh(self):
        for p, v, e in [
            ((0., 0), (0, 0), ''),
            ((0., nan), (0., nan), 'XXX'), # now: (nan, nan)
            ((1., 0.), (inf, 0.), 'XXX divide'), # now: (nan, nan)
            ((1., inf), (inf, 0.), 'XXX'), # now: (nan, nan)
            ((1., nan), (nan, nan), ''),
            ((inf, 1.), (0., pi/2), 'XXX'), # now: (nan, nan)
            ((inf, inf), (0, pi/2), 'XXX'), # now: (nan, nan)
            ((inf, nan), (0, nan), 'XXX'), # now: (nan, nan)
            ((nan, 1.), (nan, nan), ''),
            ((nan, inf), (+0, pi/2), 'XXX'), # now: (nan, nan)
            ((nan, nan), (nan, nan), ''),
        ]:
            yield self._check, np.arctanh, p, v, e

    def _check(self, func, point, value, exc=''):
        if 'XXX' in exc:
            raise nose.SkipTest
        if isinstance(point, tuple): point = complex(*point)
        if isinstance(value, tuple): value = complex(*value)
        v = dict(divide='ignore', invalid='ignore',
                 over='ignore', under='ignore')
        old_err = np.seterr(**v)
        try:
            # check sign of zero, nan, etc.
            got = complex(func(point))
            got = "(%s, %s)" % (repr(got.real), repr(got.imag))
            expected = "(%s, %s)" % (repr(value.real), repr(value.imag))
            assert got == expected, (got, expected)
            
            # check exceptions
            if exc in ('divide', 'invalid', 'over', 'under'):
                v[exc] = 'raise'
                np.seterr(**v)
                assert_raises(FloatingPointError, func, point)
            else:
                for k in v.keys(): v[k] = 'raise'
                np.seterr(**v)
                func(point)
        finally:
            np.seterr(**old_err)

class TestAttributes(TestCase):
    def test_attributes(self):
        add = ncu.add
        assert_equal(add.__name__, 'add')
        assert add.__doc__.startswith('y = add(x1,x2)\n\n')
        self.failUnless(add.ntypes >= 18) # don't fail if types added
        self.failUnless('ii->i' in add.types)
        assert_equal(add.nin, 2)
        assert_equal(add.nout, 1)
        assert_equal(add.identity, 0)

def _check_branch_cut(f, x0, dx, re_sign=1, im_sign=-1, sig_zero_ok=False,
                      dtype=np.complex):
    """
    Check for a branch cut in a function.

    Assert that `x0` lies on a branch cut of function `f` and `f` is
    continuous from the direction `dx`.

    Parameters
    ----------
    f : func
        Function to check
    x0 : array-like
        Point on branch cut
    dx : array-like
        Direction to check continuity in
    re_sign, im_sign : {1, -1}
        Change of sign of the real or imaginary part expected
    sig_zero_ok : bool
        Whether to check if the branch cut respects signed zero (if applicable)
    dtype : dtype
        Dtype to check (should be complex)

    """
    x0 = np.atleast_1d(x0).astype(dtype)
    dx = np.atleast_1d(dx).astype(dtype)
    
    scale = np.finfo(dtype).eps * 1e3
    atol  = 1e-4
    
    y0 = f(x0)
    yp = f(x0 + dx*scale*np.absolute(x0)/np.absolute(dx))
    ym = f(x0 - dx*scale*np.absolute(x0)/np.absolute(dx))
    
    assert np.all(np.absolute(y0.real - yp.real) < atol), (y0, yp)
    assert np.all(np.absolute(y0.imag - yp.imag) < atol), (y0, yp)
    assert np.all(np.absolute(y0.real - ym.real*re_sign) < atol), (y0, ym)
    assert np.all(np.absolute(y0.imag - ym.imag*im_sign) < atol), (y0, ym)
    
    if sig_zero_ok:
        # check that signed zeros also work as a displacement
        jr = (x0.real == 0) & (dx.real != 0)
        ji = (x0.imag == 0) & (dx.imag != 0)
        
        x = -x0
        x.real[jr] = 0.*dx.real
        x.imag[ji] = 0.*dx.imag
        x = -x
        ym = f(x)
        ym = ym[jr | ji]
        y0 = y0[jr | ji]
        assert np.all(np.absolute(y0.real - ym.real*re_sign) < atol), (y0, ym)
        assert np.all(np.absolute(y0.imag - ym.imag*im_sign) < atol), (y0, ym)

if __name__ == "__main__":
    run_module_suite()
