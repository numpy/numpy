from numpy.testing import *
set_package_path()
from numpy.core.umath import minimum, maximum, exp
import numpy.core.umath as ncu
from numpy import zeros, ndarray, array, choose, pi
import numpy as np
restore_path()

class TestDivision(NumpyTestCase):
    def check_division_int(self):
        # int division should return the floor of the result, a la Python
        x = array([5, 10, 90, 100, -5, -10, -90, -100, -120])
        assert_equal(x / 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        assert_equal(x // 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        assert_equal(x % 100, [5, 10, 90, 0, 95, 90, 10, 0, 80])

class TestPower(NumpyTestCase):
    def check_power_float(self):
        x = array([1., 2., 3.])
        assert_equal(x**0, [1., 1., 1.])
        assert_equal(x**1, x)
        assert_equal(x**2, [1., 4., 9.])
        y = x.copy()
        y **= 2
        assert_equal(y, [1., 4., 9.])
        assert_almost_equal(x**(-1), [1., 0.5, 1./3])
        assert_almost_equal(x**(0.5), [1., ncu.sqrt(2), ncu.sqrt(3)])

    def check_power_complex(self):
        x = array([1+2j, 2+3j, 3+4j])
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

class TestLog1p(NumpyTestCase):
    def check_log1p(self):
        assert_almost_equal(ncu.log1p(0.2), ncu.log(1.2))
        assert_almost_equal(ncu.log1p(1e-6), ncu.log(1+1e-6))

class TestExpm1(NumpyTestCase):
    def check_expm1(self):
        assert_almost_equal(ncu.expm1(0.2), ncu.exp(0.2)-1)
        assert_almost_equal(ncu.expm1(1e-6), ncu.exp(1e-6)-1)

class TestMaximum(NumpyTestCase):
    def check_reduce_complex(self):
        assert_equal(maximum.reduce([1,2j]),1)
        assert_equal(maximum.reduce([1+3j,2j]),1+3j)

class TestMinimum(NumpyTestCase):
    def check_reduce_complex(self):
        assert_equal(minimum.reduce([1,2j]),2j)

class TestFloatingPoint(NumpyTestCase):
    def check_floating_point(self):
        assert_equal(ncu.FLOATING_POINT_SUPPORT, 1)

def TestDegrees(NumpyTestCase):
    def check_degrees(self):
        assert_almost_equal(ncu.degrees(pi), 180.0)
        assert_almost_equal(ncu.degrees(-0.5*pi), -90.0)

def TestRadians(NumpyTestCase):
    def check_radians(self):
        assert_almost_equal(ncu.radians(180.0), pi)
        assert_almost_equal(ncu.degrees(-90.0), -0.5*pi)

class TestSpecialMethods(NumpyTestCase):
    def check_wrap(self):
        class with_wrap(object):
            def __array__(self):
                return zeros(1)
            def __array_wrap__(self, arr, context):
                r = with_wrap()
                r.arr = arr
                r.context = context
                return r
        a = with_wrap()
        x = minimum(a, a)
        assert_equal(x.arr, zeros(1))
        func, args, i = x.context
        self.failUnless(func is minimum)
        self.failUnlessEqual(len(args), 2)
        assert_equal(args[0], a)
        assert_equal(args[1], a)
        self.failUnlessEqual(i, 0)

    def check_old_wrap(self):
        class with_wrap(object):
            def __array__(self):
                return zeros(1)
            def __array_wrap__(self, arr):
                r = with_wrap()
                r.arr = arr
                return r
        a = with_wrap()
        x = minimum(a, a)
        assert_equal(x.arr, zeros(1))

    def check_priority(self):
        class A(object):
            def __array__(self):
                return zeros(1)
            def __array_wrap__(self, arr, context):
                r = type(self)()
                r.arr = arr
                r.context = context
                return r
        class B(A):
            __array_priority__ = 20.
        class C(A):
            __array_priority__ = 40.
        x = zeros(1)
        a = A()
        b = B()
        c = C()
        f = minimum
        self.failUnless(type(f(x,x)) is ndarray)
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

        self.failUnless(type(exp(a) is A))
        self.failUnless(type(exp(b) is B))
        self.failUnless(type(exp(c) is C))

    def check_failing_wrap(self):
        class A(object):
            def __array__(self):
                return zeros(1)
            def __array_wrap__(self, arr, context):
                raise RuntimeError
        a = A()
        self.failUnlessRaises(RuntimeError, maximum, a, a)

    def check_array_with_context(self):
        class A(object):
            def __array__(self, dtype=None, context=None):
                func, args, i = context
                self.func = func
                self.args = args
                self.i = i
                return zeros(1)
        class B(object):
            def __array__(self, dtype=None):
                return zeros(1, dtype)
        class C(object):
            def __array__(self):
                return zeros(1)
        a = A()
        maximum(zeros(1), a)
        self.failUnless(a.func is maximum)
        assert_equal(a.args[0], 0)
        self.failUnless(a.args[1] is a)
        self.failUnless(a.i == 1)
        assert_equal(maximum(a, B()), 0)
        assert_equal(maximum(a, C()), 0)

class TestChoose(NumpyTestCase):
    def check_mixed(self):
        c = array([True,True])
        a = array([True,True])
        assert_equal(choose(c, (a, 1)), array([1,1]))


class TestComplexFunctions(NumpyTestCase):
    funcs = [np.arcsin , np.arccos , np.arctan, np.arcsinh, np.arccosh,
             np.arctanh, np.sin    , np.cos   , np.tan    , np.exp,
             np.log    , np.sqrt   , np.log10]

    def check_it(self):
        for f in self.funcs:
            if f is np.arccosh :
                x = 1.5
            else :
                x = .5
            fr = f(x)
            fz = f(np.complex(x))
            assert_almost_equal(fr, fz.real, err_msg='real part %s'%f)
            assert_almost_equal(0., fz.imag, err_msg='imag part %s'%f)

    def check_precisions_consistent(self) :
        z = 1 + 1j
        for f in self.funcs :
            fcf = f(np.csingle(z))
            fcd  = f(np.cdouble(z))
            fcl = f(np.clongdouble(z))
            assert_almost_equal(fcf, fcd, decimal=6, err_msg='fch-fcd %s'%f)
            assert_almost_equal(fcf, fcl, decimal=6, err_msg='fch-fcl %s'%f)


class TestChoose(NumpyTestCase):
    def check_attributes(self):
        add = ncu.add
        assert_equal(add.__name__, 'add')
        assert_equal(add.__doc__, 'y = add(x1,x2) adds the arguments elementwise.')
        self.failUnless(add.ntypes >= 18) # don't fail if types added
        self.failUnless('ii->i' in add.types)
        assert_equal(add.nin, 2)
        assert_equal(add.nout, 1)
        assert_equal(add.identity, 0)

if __name__ == "__main__":
    NumpyTest().run()
