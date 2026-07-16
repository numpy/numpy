"""Test deprecation and future warnings.

"""
import subprocess
import sys

import pytest

import numpy as np
from numpy.testing import IS_WASM


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


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
def test_lapack_lite_deprecation():
    """Check that importing lapack_lite raises a DeprecationWarning."""
    code = "import numpy.linalg.lapack_lite"
    res = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", code],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "The numpy.linalg.lapack_lite module is deprecated" in res.stderr
