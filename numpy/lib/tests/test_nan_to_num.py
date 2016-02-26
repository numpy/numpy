# tests nan_to_num for fixing bug #1447.

from numpy import nan_to_num, float64, int64
from numpy.testing import assert_equal

def test_scalar():
    '''
    Test whether nan_to_num returns scalar when called with scaler.

    Due to the bug #1447 nan_to_num returns an array when called with a scalar.
    '''
    x = 1.0
    y = nan_to_num(x)
    assert_equal(type(y), float64)

    x = 1
    y = nan_to_num(x)
    assert_equal(type(y), int64)
