"""Test deprecation and future warnings.

"""
from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import TestCase, run_module_suite, assert_warns
from numpy.ma.testutils import assert_equal


class TestArgsort(TestCase):
	""" gh-8701 """
	def _test_base(self, argsort, cls):
		arr_0d = np.array(1).view(cls)
		argsort(arr_0d)

		arr_1d = np.array([1, 2, 3]).view(cls)
		argsort(arr_1d)

		# argsort has a bad default for >1d arrays
		arr_2d = np.array([[1, 2], [3, 4]]).view(cls)
		result = assert_warns(
			np.ma.core.MaskedArrayFutureWarning, argsort, arr_2d)
		assert_equal(result, argsort(arr_2d, axis=None))

		# should be no warnings for explictly specifiying it
		argsort(arr_2d, axis=None)
		argsort(arr_2d, axis=-1)

	def test_function_ndarray(self):
		return self._test_base(np.ma.argsort, np.ndarray)

	def test_function_maskedarray(self):
		return self._test_base(np.ma.argsort, np.ma.MaskedArray)

	def test_method(self):
		return self._test_base(np.ma.MaskedArray.argsort, np.ma.MaskedArray)


class TestMinimumMaximum(TestCase):
    def test_minimum(self):
        assert_warns(DeprecationWarning, np.ma.minimum, np.ma.array([1, 2]))

    def test_maximum(self):
        assert_warns(DeprecationWarning, np.ma.maximum, np.ma.array([1, 2]))


if __name__ == "__main__":
    run_module_suite()
