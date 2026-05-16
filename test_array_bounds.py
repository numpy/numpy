"""Test file for array bounds validation improvements in NumPy."""
import numpy as np
import pytest


def test_diagonal_offset_bounds():
    """Test that diagonal offset is validated."""
    # Fix: Validate offset is within array bounds
    # Before: large offset → out-of-bounds read
    # After: raises ValueError with clear message
    
    arr = np.eye(3)  # 3x3 identity
    
    # Large offset out of bounds
    with pytest.raises(ValueError, match="Offset|bounds"):
        np.diagonal(arr, offset=100)
    
    # Negative offset out of bounds
    with pytest.raises(ValueError, match="Offset|bounds"):
        np.diagonal(arr, offset=-100)


def test_reshape_size_mismatch():
    """Test that reshape validates size compatibility."""
    # Fix: Check that old size == new size
    # Before: reshape(3,5) on 12-element array → silent corruption
    # After: raises ValueError
    
    arr = np.arange(12)  # 12 elements
    
    # Reshape to incompatible size (should fail)
    with pytest.raises(ValueError, match="shape|size"):
        arr.reshape(3, 5)  # 15 elements != 12


def test_view_dtype_mismatch():
    """Test that view validates dtype compatibility."""
    # Fix: Check that byte count is divisible by new itemsize
    # Before: uint32[3] → uint64 → invalid shape
    # After: raises ValueError
    
    arr = np.array([1, 2, 3], dtype=np.uint32)  # 12 bytes
    
    # View as larger dtype without proper size
    with pytest.raises(ValueError, match="divisible|dtype"):
        arr.view(np.uint64)  # 12 bytes not divisible by 8
