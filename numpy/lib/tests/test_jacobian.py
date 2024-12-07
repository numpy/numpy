import numpy as np
from numpy.testing import (
    assert_equal, assert_array_equal,assert_allclose
    )
from numpy.lib.jacobian import jacobian

class TestJacobian:
    def test_linear_function(self):
        def linear_func(x):
            return np.array([2*x[0]+x[1], x[2] - x[1]])
        
        x0 = np.array([1.0,2.0,3.0])
        J = jacobian(linear_func, x0)
        expected = np.array([
            [2, 1, 0],
            [0, -1, 1]
        ])

        assert_equal(J.shape, expected.shape)
        assert_allclose(J, expected, atol=1e-6)

    def test_non_linear_function(self):
        def non_linear_function(x):
            return np.array([x[0]**2, np.sin(x[1]), np.exp(x[2])])

        x0 = np.array([1.0, np.pi / 4, 0.5])
        J = jacobian(non_linear_function, x0)
        expected = np.array([
            [2 * x0[0], 0, 0],
            [0, np.cos(x0[1]), 0],
            [0, 0, np.exp(x0[2])]
        ])
        assert_equal(J.shape, expected.shape)
        assert_allclose(J, expected, atol=1e-6)
    
    def test_multidimensional_output(self):
        def multi_output_func(x):
            return np.array([
                x[0] + x[1] * x[2],
                x[1]**2,
                x[2] + np.log(1 + x[0])
            ])

        x0 = np.array([1.0, 2.0, 3.0])
        J = jacobian(multi_output_func, x0)
        expected = np.array([
            [1, x0[2], x0[1]],
            [0, 2 * x0[1], 0],
            [1 / (1 + x0[0]), 0, 1]
        ])
        assert_equal(J.shape, expected.shape)
        assert_allclose(J, expected, atol=1e-6)

    def test_constant_function(self):
        def constant_func(x):
            return np.array([5.0, 5.0, 5.0])

        x0 = np.array([1.0, 2.0, 3.0])
        J = jacobian(constant_func, x0)
        expected = np.zeros((3, 3))
        assert_equal(J.shape, expected.shape)
        assert_array_equal(J, expected)

    def test_zero_input(self):
        def zero_input_func(x):
            return np.array([x[0]**2 + x[1], np.sin(x[1] * x[2]), x[2]])

        x0 = np.array([0.0, 0.0, 0.0])
        J = jacobian(zero_input_func, x0)
        expected = np.array([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1]
        ])
        assert_equal(J.shape, expected.shape)
        assert_allclose(J, expected, atol=1e-6)