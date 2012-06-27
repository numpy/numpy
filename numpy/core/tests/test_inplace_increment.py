import sys
from decimal import Decimal

import numpy as np
from numpy.testing import *
from numpy.testing.utils import WarningManager
import warnings


class TestInplaceIncrement(TestCase):
	def test_dtypes(self):
		def testdt(dt):
			a = np.arange(12).reshape((3,4)).astype(dt)
			index = ([1,1,2,0], [0,0,2,3])
			inc = [50,50, 30,16]
			
			np.inplace_increment(a, index, inc)
			
			assert_array_almost_equal(a, np.array([[   0,    1,    2,   19],
											[ 104,    5,    6,    7],
											[   8,    9,   40,   11]]).astype(dt))
		
		testdt(np.float)
		testdt(np.int)
		testdt(np.complex)
		
	def test_slice(self):
		a = np.arange(12).reshape((3,4)).astype(float)
		index = (1, slice(None, None))
		inc = [50,50, 30,16]

		np.inplace_increment(a, index, inc)

		assert_array_almost_equal(a, np.array([[   0,    1,    2,   3],
											[ 54,    55,    36,  23],
											[   8,    9,   10,   11]]).astype(float))
	
	@dec.knownfailureif( True, "not sure what's wrong here")										
	def test_boolean(self):
		a = np.arange(12).reshape((3,4)).astype(float)
		index = (1, [ False, True, False, True])
		inc = [50, 30,16, -30]
		
		np.inplace_increment(a, index, inc)
		
		assert_array_almost_equal(a, np.array([[   0,    1,    2,   3],
											[   4,   55,   36,  23],
											[   8,    9,   10,   11]]).astype(float))

		

if __name__ == "__main__":
	run_module_suite()
