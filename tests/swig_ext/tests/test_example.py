import sys
from numpy.testing import *
set_package_path()
from swig_ext import example
restore_path()

class test_example(NumpyTestCase):

    def check_fact(self):
        assert_equal(example.fact(10),3628800)

    def check_cvar(self):
        assert_equal(example.cvar.My_variable,3.0)
        example.cvar.My_variable = 5
        assert_equal(example.cvar.My_variable,5.0)

if __name__ == "__main__":
    NumpyTest().run()
