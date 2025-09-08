import numpy as np

def test_nancumsum_reverse_simple():
    x = np.array([1.0, np.nan, 2.0])
    np.testing.assert_allclose(np.nancumsum(x, reverse=True), np.array([3.0, 2.0, 2.0]))
