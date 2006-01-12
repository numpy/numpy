
from numpy.testing import *
set_package_path()
from numpy.core.umath import minimum, maximum
import numpy.core.umath as ncu
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

if __name__ == "__main__":
    ScipyTest().run()
