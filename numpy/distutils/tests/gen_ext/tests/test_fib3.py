from __future__ import division, absolute_import, print_function

from gen_ext import fib3
from numpy.testing import TestCase, run_module_suite, assert_array_equal

class TestFib3(TestCase):
    def test_fib(self):
        assert_array_equal(fib3.fib(6), [0, 1, 1, 2, 3, 5])

if __name__ == "__main__":
    run_module_suite()
