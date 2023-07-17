import numpy as np
from numpy import float64
from numpy.testing import assert_equal

class TestInf:
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
             0, -0, -0, 0])
        
        with np.errstate(invalid='ignore'):
            assert_equal(np.floor_divide(dividend, divisor), expected)
