import pytest

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, assert_raises, assert_raises_regex,
    )
from numpy.lib import jacobian

class TestJacobian:
    def test_linear_function(self):
        def linear_func(x):
            return np.array([2*x[0]+x[1], x[2] - x[1]])
        
        x0 = np.array([1.0,2.0,3.0])
        J = jacobian(linear_func, x0)