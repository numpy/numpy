from __future__ import division, absolute_import, print_function

from numpy.testing import TestCase, run_module_suite, assert_equal
from f2py_f90_ext import foo

class TestFoo(TestCase):
    def test_foo_free(self):
        assert_equal(foo.foo_free.bar13(), 13)

if __name__ == "__main__":
    run_module_suite()
