import numpy as np

def test_cumsum_reverse_1d():
    x = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(np.cumsum(x, reverse=True), np.array([10, 9, 7, 4]))

def test_cumprod_reverse_basic():
    x = np.array([2, 3, 4])
    np.testing.assert_array_equal(np.cumprod(x, reverse=True), np.array([24, 12, 4]))
