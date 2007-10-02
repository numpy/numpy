import sys
from numpy.testing import *

set_package_path()
from pyrex_ext.primes import primes
restore_path()

class TestPrimes(NumpyTestCase):
    def check_simple(self, level=1):
        l = primes(10)
        assert_equal(l, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
if __name__ == "__main__":
    NumpyTest().run()
