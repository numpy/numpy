import pytest

import numpy as np
from numpy.ma.testutils import assert_almost_equal


def test_assert_almost_equal_decimal_6_fails():
    actual = np.array([0.0])
    desired = np.array([7.7e-7])
    with pytest.raises(AssertionError):
        assert_almost_equal(actual, desired, decimal=6)
