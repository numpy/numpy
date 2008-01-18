import numpy as np
from numpy.testing import *

class TestUfunc(NumpyTestCase):
    def test_reduceat_shifting_sum(self):
        L = 6
        x = np.arange(L)
        idx = np.array(zip(np.arange(L-2), np.arange(L-2)+2)).ravel()
        assert_array_equal(np.add.reduceat(x,idx)[::2],
                           [1,3,5,7])

if __name__ == "__main__":
    NumpyTest().run()
