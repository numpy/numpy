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
										
	def test_boolean(self):
		a = np.arange(12).reshape((3,4)).astype(float)
		index = (1, np.array([ False, True, False, True], dtype = bool))
		inc = [30,16]
		
		np.inplace_increment(a, index, inc)
		
		assert_array_almost_equal(a, np.array([[   0,    1,    2,   3],
											[   4,   35,    6,  23],
											[   8,    9,   10,   11]]).astype(float))

		

if __name__ == "__main__":
	run_module_suite()
