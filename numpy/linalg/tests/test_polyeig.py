"""Test functions for polynomial eigenvalue problem solver."""

import numpy as np
import pytest
from numpy import linalg
from numpy.testing import (
    assert_,
    assert_allclose,
    assert_raises,
)

class TestPolyEig:
    def test_basic(self):
        # Test basic polynomial eigenvalue problem
        A0 = np.array([[1, 0], [0, 1]])
        A1 = np.array([[0, 1], [-1, 0]])
        A2 = np.array([[1, 0], [0, 1]])
        eigenvalues, eigenvectors = linalg.polyeig(A0, A1, A2)
        # Check that eigenvalues and eigenvectors satisfy the equation
        for i in range(len(eigenvalues)):
            lambda_val = eigenvalues[i]
            v = eigenvectors[:, i]
            result = (A0 + lambda_val * A1 + lambda_val ** 2 * A2) @ v
            assert_allclose(result, np.zeros_like(result), rtol=1e-10)

    def test_empty_input(self):
        # Test that empty input raises ValueError
        assert_raises(ValueError, linalg.polyeig)

    def test_non_square_matrix(self):
        # Test that non-square matrices raise ValueError
        A0 = np.array([[1, 2, 3], [4, 5, 6]])
        A1 = np.array([[1, 2], [3, 4]])
        assert_raises(ValueError, linalg.polyeig, A0, A1)

    def test_inconsistent_shapes(self):
        # Test that matrices with inconsistent shapes raise ValueError
        A0 = np.array([[1, 0], [0, 1]])
        A1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert_raises(ValueError, linalg.polyeig, A0, A1)

    def test_complex_matrices(self):
        # Test with complex matrices
        A0 = np.array([[1 + 1j, 2j], [-2j, 1 - 1j]])
        A1 = np.array([[0, 1], [-1, 0]])
        eigenvalues, eigenvectors = linalg.polyeig(A0, A1)
        # Check that eigenvalues and eigenvectors satisfy the equation
        for i in range(len(eigenvalues)):
            lambda_val = eigenvalues[i]
            v = eigenvectors[:, i]
            result = (A0 + lambda_val * A1) @ v
            assert_allclose(result, np.zeros_like(result), rtol=1e-10)

    def test_high_degree_polynomial(self):
        # Test with higher degree polynomial
        A0 = np.array([[1, 0], [0, 1]])
        A1 = np.array([[0, 1], [-1, 0]])
        A2 = np.array([[1, 0], [0, 1]])
        A3 = np.array([[0, 1], [-1, 0]])
        eigenvalues, eigenvectors = linalg.polyeig(A0, A1, A2, A3)
        # Check that eigenvalues and eigenvectors satisfy the equation
        for i in range(len(eigenvalues)):
            lambda_val = eigenvalues[i]
            v = eigenvectors[:, i]
            result = (
                A0 + lambda_val * A1 + lambda_val ** 2 * A2 + lambda_val ** 3 * A3
            ) @ v
            assert_allclose(result, np.zeros_like(result), rtol=1e-10)

    def test_singular_matrices(self):
        # Test with singular matrices
        A0 = np.array([[1, 0], [0, 0]])
        A1 = np.array([[0, 1], [-1, 0]])
        eigenvalues, eigenvectors = linalg.polyeig(A0, A1)
        # Check that eigenvalues and eigenvectors satisfy the equation
        for i in range(len(eigenvalues)):
            lambda_val = eigenvalues[i]
            v = eigenvectors[:, i]
            result = (A0 + lambda_val * A1) @ v
            assert_allclose(result, np.zeros_like(result), rtol=1e-10)

    def test_zero_matrices(self):
        # Test with zero matrices
        A0 = np.zeros((2, 2))
        A1 = np.zeros((2, 2))
        eigenvalues, eigenvectors = linalg.polyeig(A0, A1)
        # All eigenvalues should be zero
        assert_allclose(eigenvalues, np.zeros_like(eigenvalues))

    def test_identity_matrices(self):
        # Test with identity matrices
        A0 = np.eye(2)
        A1 = np.eye(2)
        eigenvalues, eigenvectors = linalg.polyeig(A0, A1)
        # Check that eigenvalues and eigenvectors satisfy the equation
        for i in range(len(eigenvalues)):
            lambda_val = eigenvalues[i]
            v = eigenvectors[:, i]
            result = (A0 + lambda_val * A1) @ v
            assert_allclose(result, np.zeros_like(result), rtol=1e-10)

    def test_return_type(self):
        # Test that the return type is correct
        A0 = np.array([[1, 0], [0, 1]])
        A1 = np.array([[0, 1], [-1, 0]])
        result = linalg.polyeig(A0, A1)
        assert_(isinstance(result, linalg.PolyEigResult))
        assert_(hasattr(result, 'eigenvalues'))
        assert_(hasattr(result, 'eigenvectors'))

    @pytest.mark.parametrize(
        'dtype', [np.float32, np.float64, np.complex64, np.complex128]
    )
    def test_dtypes(self, dtype):
        # Test with different data types
        A0 = np.array([[1, 0], [0, 1]], dtype=dtype)
        A1 = np.array([[0, 1], [-1, 0]], dtype=dtype)
        eigenvalues, eigenvectors = linalg.polyeig(A0, A1)
        # Check that eigenvalues and eigenvectors satisfy the equation
        for i in range(len(eigenvalues)):
            lambda_val = eigenvalues[i]
            v = eigenvectors[:, i]
            result = (A0 + lambda_val * A1) @ v
            assert_allclose(result, np.zeros_like(result), rtol=1e-10)

    def test_large_matrices(self):
        # Test with larger matrices
        n = 10
        A0 = np.eye(n)
        A1 = np.eye(n)
        A2 = np.eye(n)
        eigenvalues, eigenvectors = linalg.polyeig(A0, A1, A2)
        # Check that eigenvalues and eigenvectors satisfy the equation
        for i in range(len(eigenvalues)):
            lambda_val = eigenvalues[i]
            v = eigenvectors[:, i]
            result = (A0 + lambda_val * A1 + lambda_val ** 2 * A2) @ v
            assert_allclose(result, np.zeros_like(result), rtol=1e-10) 
            