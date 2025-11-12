"""
Tests for same_value casting support in np.can_cast

Note: For can_cast, same_value casting can only do type-level checking,
not value-level checking. It should be conservative and only return True
for casts that are guaranteed to preserve all possible values.
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

    def test_safe_widening_casts_only(self):
        """Test same_value casting for absolutely safe widening casts only"""
        # These should work because ALL values are guaranteed to be preserved
        assert np.can_cast(np.int8, np.int16, casting='same_value')
        assert np.can_cast(np.int16, np.int32, casting='same_value')
        assert np.can_cast(np.int32, np.int64, casting='same_value')
        assert np.can_cast(np.uint8, np.uint16, casting='same_value')
        assert np.can_cast(np.uint16, np.uint32, casting='same_value')
        assert np.can_cast(np.uint32, np.uint64, casting='same_value')

    def test_all_unsafe_casts_fail(self):
        """Test that all potentially unsafe casts fail with same_value"""
        # These should fail because some values could be lost
        assert not np.can_cast(np.int64, np.int32, casting='same_value')
        assert not np.can_cast(np.float64, np.float32, casting='same_value')
        # Could lose sign
        assert not np.can_cast(np.int32, np.uint32, casting='same_value')

        # Any float to integer conversion could lose decimal part
        assert not np.can_cast(np.float32, np.int32, casting='same_value')
        assert not np.can_cast(np.float64, np.int64, casting='same_value')

        # Integer to float could lose precision for large integers
        assert not np.can_cast(np.int64, np.float64, casting='same_value')
        assert not np.can_cast(np.int32, np.float32, casting='same_value')

    def test_complex_to_real_fails(self):
        """Test that complex to real casts fail with same_value"""
        # These should fail because imaginary part would be lost
        assert not np.can_cast(np.complex64, np.float32, casting='same_value')
        assert not np.can_cast(np.complex128, np.float64, casting='same_value')

    def test_comparison_with_other_modes(self):
        """Test that same_value is more restrictive than other modes"""
        # same_value should be more restrictive than unsafe
        assert not np.can_cast(np.float64, np.int32, casting='same_value')
        assert np.can_cast(np.float64, np.int32, casting='unsafe')

        # same_value should be more restrictive than safe
        # for questionable cases
        assert not np.can_cast(np.int64, np.float64, casting='same_value')
        # Note: This might be True for safe casting,
        # but same_value is more conservative

    def test_parameter_validation(self):
        """Test that invalid casting parameters are rejected"""
        with pytest.raises(ValueError):
            np.can_cast(np.int32, np.int64, casting='invalid_mode')

    def test_bool_conversions(self):
        """Test boolean conversion rules with same_value"""
        # Bool to integer should be safe (True->1, False->0)
        assert np.can_cast(np.bool_, np.int8, casting='same_value')
        assert np.can_cast(np.bool_, np.int32, casting='same_value')
        assert np.can_cast(np.bool_, np.int64, casting='same_value')

        # Integer to bool is potentially lossy (non-zero values)
        assert not np.can_cast(np.int8, np.bool_, casting='same_value')
        assert not np.can_cast(np.int32, np.bool_, casting='same_value')
