from numpy.testing import *

set_package_path()
import numpy as N
restore_path()

class TestUfunc(NumpyTestCase):
    def test_reduceat_shifting_sum(self):
        L = 6
        x = N.arange(L)
        idx = N.array(zip(N.arange(L-2),N.arange(L-2)+2)).ravel()
        assert_array_equal(N.add.reduceat(x,idx)[::2],
                           [1,3,5,7])

if __name__ == "__main__":
    NumpyTest().run()
