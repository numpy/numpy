import numpy as np
import pytest

from numpy.matrixlib.modular import matrix_inv_mod


def test_inv_mod_2x2_mod26():
    A = np.array([[3, 3],
                  [2, 5]])
    invA = matrix_inv_mod(A, 26)

    I = np.eye(2, dtype=int)
    result = (A @ invA) % 26

    assert np.array_equal(result, I)


def test_inv_mod_3x3_mod29():
    A = np.array([[2, 5, 7],
                  [6, 3, 4],
                  [5, -2, -3]])

    invA = matrix_inv_mod(A, 29)

    I = np.eye(3, dtype=int)
    result = (A @ invA) % 29

    assert np.array_equal(result, I)


def test_inv_mod_identity():
    A = np.eye(4, dtype=int)
    invA = matrix_inv_mod(A, 17)

    assert np.array_equal(invA, A)


def test_inv_mod_non_invertible():
    A = np.array([[2, 4],
                  [1, 2]])

    with pytest.raises(ValueError):
        matrix_inv_mod(A, 26)


def test_inv_mod_non_square():
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])

    with pytest.raises(ValueError):
        matrix_inv_mod(A, 11)


def test_inv_mod_negative_entries():
    A = np.array([[1, -1],
                  [1,  2]])

    invA = matrix_inv_mod(A, 7)

    I = np.eye(2, dtype=int)
    result = (A @ invA) % 7

    assert np.array_equal(result, I)
