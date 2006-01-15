
from numpy.testing import *
set_package_path()
from numpy.core.umath import minimum, maximum
import numpy.core.umath as ncu
from numpy import zeros
restore_path()

class test_log1p(ScipyTestCase):
    def check_log1p(self):
        assert_almost_equal(ncu.log1p(0.2), ncu.log(1.2))
        assert_almost_equal(ncu.log1p(1e-6), ncu.log(1+1e-6))

class test_expm1(ScipyTestCase):
    def check_expm1(self):
        assert_almost_equal(ncu.expm1(0.2), ncu.exp(0.2)-1)
        assert_almost_equal(ncu.expm1(1e-6), ncu.exp(1e-6)-1)

class test_maximum(ScipyTestCase):
    def check_reduce_complex(self):
        assert_equal(maximum.reduce([1,2j]),1)
        assert_equal(maximum.reduce([1+3j,2j]),1+3j)

class test_minimum(ScipyTestCase):
    def check_reduce_complex(self):
        assert_equal(minimum.reduce([1,2j]),2j)

class test_floating_point(ScipyTestCase):
    def check_floating_point(self):
        assert_equal(ncu.FLOATING_POINT_SUPPORT, 1)

class test_special_methods(ScipyTestCase):
    def test_wrap(self):
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

    def test_old_wrap(self):
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

    def test_priority(self):
        class A(object):
            def __array__(self):
                return zeros(1)
            def __array_wrap__(self, arr, context):
                r = type(self)()
                r.arr = arr
                r.context = context
                return r
        class B(A):
            __array_priority__ = 20
        class C(A):
            __array_priority__ = 40
        a = A()
        b = B()
        c = C()
        f = minimum
        self.failUnless(isinstance(f(a,a), A))
        self.failUnless(isinstance(f(a,b), B))
        self.failUnless(isinstance(f(b,a), B))
        self.failUnless(isinstance(f(b,b), B))
        self.failUnless(isinstance(f(b,c), C))
        self.failUnless(isinstance(f(c,b), C))
        self.failUnless(isinstance(f(c,c), C))

if __name__ == "__main__":
    ScipyTest().run()
