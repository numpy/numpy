import sys
from decimal import Decimal

import numpy as np
from numpy.testing import *
from numpy.testing.utils import WarningManager
import warnings


class (TestCase):
    def test_int(self):
        
		a = arange(12).reshape((3,4))
		index = ([1,1,2,0], [0,0,2,3])
		inc = [50,50, 30,16]
		
		np.inplace_increment(a, index, inc)
		
        assert_array_almost_equal(a, array([[   0,    1,    2,   19],
       [ 104,    5,    6,    7],
       [   8,    9,   40,   11]]))
	   
		

if __name__ == "__main__":
    run_module_suite()
