"""Test deprecation and future warnings.

"""
import pytest

import numpy as np


def test_qr_mode_full_removed():
    """Check mode='full' and mode='economic' raise ValueError."""
    a = np.eye(2)
    with pytest.raises(ValueError, match="deprecated in NumPy 1.8"):
        np.linalg.qr(a, mode='full')
    with pytest.raises(ValueError, match="deprecated in NumPy 1.8"):
        np.linalg.qr(a, mode='f')
    with pytest.raises(ValueError, match="deprecated in NumPy 1.8"):
        np.linalg.qr(a, mode='economic')
    with pytest.raises(ValueError, match="deprecated in NumPy 1.8"):
        np.linalg.qr(a, mode='e')
