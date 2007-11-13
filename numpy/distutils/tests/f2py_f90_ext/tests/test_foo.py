import sys
from numpy.testing import *
set_package_path()
from f2py_f90_ext import foo
del sys.path[0]

class TestFoo(NumpyTestCase):

    def check_foo_free(self):
        assert_equal(foo.foo_free.bar13(),13)

if __name__ == "__main__":
    NumpyTest().run()
