import sys
from decimal import Decimal

import numpy as np
from numpy.testing import *
from numpy.testing.utils import WarningManager
import warnings


class (TestCase):
    def test_dtypes(self):
        def test(dt):
			a = arange(12).reshape((3,4)).astype(dt)
			index = ([1,1,2,0], [0,0,2,3])
			inc = [50,50, 30,16]
			
			np.inplace_increment(a, index, inc)
			
			assert_array_almost_equal(a, array([[   0,    1,    2,   19],
		   [ 104,    5,    6,    7],
		   [   8,    9,   40,   11]]).astype(dt))
		   
		test(float)
		test(int)
		test(complex)
		
	def test_slice(self):
		a = arange(12).reshape((3,4)).astype(float)
		index = (1, Slice(None, None))
		inc = [50,50, 30,16]
		
		np.inplace_increment(a, index, inc)
		
		assert_array_almost_equal(a, array([[   0,    1,    2,   3],
											[ 54,    55,    36,  23],
											[   8,    9,   10,   11]]).astype(float))
											
	def test_boolean(self):
		a = arange(12).reshape((3,4)).astype(float)
		index = (1, [ False, True, False, True])
		inc = [50, 30,16]
		
		np.inplace_increment(a, index, inc)
		
		assert_array_almost_equal(a, array([[   0,    1,    2,   3],
											[   4,   55,   36,  23],
											[   8,    9,   10,   11]]).astype(float))

		

if __name__ == "__main__":
    run_module_suite()
