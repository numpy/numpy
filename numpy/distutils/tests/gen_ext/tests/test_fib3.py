import sys
from numpy.testing import *
set_package_path()
from gen_ext import fib3
del sys.path[0]

class TestFib3(TestCase):

    def test_fib(self):
        assert_array_equal(fib3.fib(6),[0,1,1,2,3,5])

if __name__ == "__main__":
    nose.run(argv=['', __file__])
