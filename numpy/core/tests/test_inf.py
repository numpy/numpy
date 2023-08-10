import numpy as np
from numpy import float64
from numpy.testing import assert_equal

class TestInf:
    def assert_equal_zero(self, expected, actual):
        is_same_sign = np.copysign(1, expected) == np.copysign(1, actual)
        are_zeroes = expected == 0 and actual == 0

        if not is_same_sign and are_zeroes:
            raise AssertionError("Arrays are not equal {} vs {}",
                                 repr(expected), repr(actua;l))

    
    def assert_equal_zeroes(self, expected_list, actual_list):
        for expected, actual in zip(expected_list, actual_list):
            self.assert_equal_zero(expected, actual)

    def test_inf_ufuncs(self):
        """Test the various ufuncs"""

        dividend = np.array(
            [np.inf] * 5
            + [-np.inf] * 5
            + [324434.4354] * 2
            + [-434354.45543] * 2, dtype=float64)
        
        divisor = np.array(
            [np.inf, -np.inf, 8657463.3435, -59420603.453265, 0] * 2
            + [np.inf, -np.inf] * 2, dtype=float64)
        
        expected = np.array(
            [np.nan, np.nan, np.inf, -np.inf, np.nan,
             np.nan, np.nan, -np.inf, np.inf, np.nan,
             0., -0., -0., 0.])
        
        with np.errstate(invalid='ignore'):
            actual = np.floor_divide(dividend, divisor)

            expected_subarray_1 = expected[:10]
            actual_subarray_1 = actual[:10]
            assert_equal(expected_subarray_1, actual_subarray_1)

            expected_subarray_2 = expected[10:]
            actual_subarray_2 = actual[10:]
            self.assert_equal_zeroes(
                expected_subarray_2, actual_subarray_2)
