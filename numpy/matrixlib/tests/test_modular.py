import numpy as np
import pytest

def test_inv_mod_2x2():
    A = np.array([[3, 3], [2, 5]])
    inv = np.matrixlib.inv_mod(A, 26)
    assert np.all((A @ inv) % 26 == np.eye(2) % 26)
