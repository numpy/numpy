"""
Tests for same_value casting support in np.can_cast
"""

import numpy as np
import pytest


class TestCanCastSameValue:
    """Test same_value casting functionality in np.can_cast"""

    def test_identity_casts(self):
        """Test that same_value casting works for identity casts"""
        dtypes = [np.int32, np.float64, np.complex128, np.bool_]
        for dtype in dtypes:
            assert np.can_cast(dtype, dtype, casting='same_value')

    def test_widening_integer_casts(self):
        """Test same_value casting for safe widening integer casts"""
        # These should work because no precision is lost
        assert np.can_cast(np.int8, np.int16, casting='same_value')
        assert np.can_cast(np.int16, np.int32, casting='same_value')
        assert np.can_cast(np.int32, np.int64, casting='same_value')
        assert np.can_cast(np.uint8, np.uint16, casting='same_value')
        assert np.can_cast(np.uint16, np.uint32, casting='same_value')

    def test_narrowing_casts_fail(self):
        """Test that narrowing casts that could lose data fail with same_value"""
        # These should fail because data could be lost
        assert not np.can_cast(np.int64, np.int32, casting='same_value')
        assert not np.can_cast(np.float64, np.float32, casting='same_value')
        assert not np.can_cast(np.int32, np.uint32, casting='same_value')  # Could lose sign

    def test_float_to_integer_fails(self):
        """Test that float to integer casts fail with same_value"""
        # These should fail because decimal part would be lost
        assert not np.can_cast(np.float32, np.int32, casting='same_value')
        assert not np.can_cast(np.float64, np.int64, casting='same_value')

    def test_complex_to_real_fails(self):
        """Test that complex to real casts fail with same_value"""
        # These should fail because imaginary part would be lost
        assert not np.can_cast(np.complex64, np.float32, casting='same_value')
        assert not np.can_cast(np.complex128, np.float64, casting='same_value')

    def test_comparison_with_other_modes(self):
        """Test that same_value is more restrictive than unsafe but may allow some safe casts"""
        # same_value should be more restrictive than unsafe
        assert not np.can_cast(np.float64, np.int32, casting='same_value')
        assert np.can_cast(np.float64, np.int32, casting='unsafe')
        
        # But should allow some safe casts
        assert np.can_cast(np.int32, np.int64, casting='same_value')
        assert np.can_cast(np.int32, np.int64, casting='safe')

    def test_parameter_validation(self):
        """Test that invalid casting parameters are rejected"""
        with pytest.raises(ValueError):
            np.can_cast(np.int32, np.int64, casting='invalid_mode')
