import numpy as np
from numpy.testing import *

class TestArrayRepr(object):
    def test_nan_inf(self):
        x = np.array([np.nan, np.inf])
        assert_equal(repr(x), 'array([ nan,  inf])')

if __name__ == "__main__":
    run_module_suite()
