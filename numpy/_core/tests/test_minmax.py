"""Tests for np.minmax function."""

import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_equal


class TestMinMax:
    """Tests for np.minmax."""

    def test_basic(self):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        mn, mx = np.minmax(a)
        assert_equal(mn, 1)
        assert_equal(mx, 9)

    def test_scalar(self):
        mn, mx = np.minmax(42)
        assert_equal(mn, 42)
        assert_equal(mx, 42)

    def test_2d_no_axis(self):
        a = np.array([[3, 1], [4, 5]])
        mn, mx = np.minmax(a)
        assert_equal(mn, 1)
        assert_equal(mx, 5)

    def test_2d_axis0(self):
        a = np.array([[3, 1], [4, 5]])
        mn, mx = np.minmax(a, axis=0)
        assert_array_equal(mn, [3, 1])
        assert_array_equal(mx, [4, 5])

    def test_2d_axis1(self):
        a = np.array([[3, 1], [4, 5]])
        mn, mx = np.minmax(a, axis=1)
        assert_array_equal(mn, [1, 4])
        assert_array_equal(mx, [3, 5])

    def test_3d_axis(self):
        a = np.arange(24).reshape((2, 3, 4))
        mn, mx = np.minmax(a, axis=2)
        assert_array_equal(mn, np.min(a, axis=2))
        assert_array_equal(mx, np.max(a, axis=2))

    def test_3d_axis_tuple(self):
        a = np.arange(24).reshape((2, 3, 4))
        mn, mx = np.minmax(a, axis=(0, 2))
        assert_array_equal(mn, np.min(a, axis=(0, 2)))
        assert_array_equal(mx, np.max(a, axis=(0, 2)))

    def test_keepdims(self):
        a = np.array([[3, 1], [4, 5]])
        mn, mx = np.minmax(a, axis=0, keepdims=True)
        assert mn.shape == (1, 2)
        assert mx.shape == (1, 2)
        assert_array_equal(mn, [[3, 1]])
        assert_array_equal(mx, [[4, 5]])

    def test_float(self):
        a = np.array([1.5, -2.3, 4.7, 0.1])
        mn, mx = np.minmax(a)
        assert_equal(mn, -2.3)
        assert_equal(mx, 4.7)

    def test_nan_propagation(self):
        a = np.array([1.0, np.nan, 3.0])
        mn, mx = np.minmax(a)
        assert np.isnan(mn)
        assert np.isnan(mx)

    def test_all_same(self):
        a = np.array([5, 5, 5, 5])
        mn, mx = np.minmax(a)
        assert_equal(mn, 5)
        assert_equal(mx, 5)

    def test_single_element(self):
        a = np.array([42])
        mn, mx = np.minmax(a)
        assert_equal(mn, 42)
        assert_equal(mx, 42)

    def test_negative_axis(self):
        a = np.arange(12).reshape((3, 4))
        mn, mx = np.minmax(a, axis=-1)
        assert_array_equal(mn, np.min(a, axis=-1))
        assert_array_equal(mx, np.max(a, axis=-1))

    def test_empty_with_initial(self):
        a = np.array([])
        mn, mx = np.minmax(a, initial=(np.inf, -np.inf))
        assert_equal(mn, np.inf)
        assert_equal(mx, -np.inf)

    def test_where(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mask = np.array([True, False, True, False, True])
        mn, mx = np.minmax(a, initial=(np.inf, -np.inf), where=mask)
        assert_equal(mn, 1.0)
        assert_equal(mx, 5.0)

    def test_integer_dtypes(self):
        for dtype in [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ]:
            a = np.array([10, 3, 7, 1, 9], dtype=dtype)
            mn, mx = np.minmax(a)
            assert_equal(mn, dtype(1))
            assert_equal(mx, dtype(10))

    def test_float_dtypes(self):
        for dtype in [np.float32, np.float64]:
            a = np.array([1.5, -2.3, 4.7, 0.1], dtype=dtype)
            mn, mx = np.minmax(a)
            assert_equal(mn, dtype(-2.3))
            assert_equal(mx, dtype(4.7))

    def test_bool(self):
        a = np.array([True, False, True])
        mn, mx = np.minmax(a)
        assert_equal(mn, False)
        assert_equal(mx, True)

    def test_datetime(self):
        for dtype in ("m8[s]", "m8[Y]"):
            a = np.arange(10).astype(dtype)
            mn, mx = np.minmax(a)
            assert_equal(mn, a[0])
            assert_equal(mx, a[9])

    def test_datetime_nat_propagation(self):
        for dtype in ("m8[s]", "m8[Y]"):
            a = np.arange(10).astype(dtype)
            a[3] = "NaT"
            mn, mx = np.minmax(a)
            assert_equal(mn, a[3])
            assert_equal(mx, a[3])

    def test_complex(self):
        a = np.array([1 + 2j, 3 + 0j, 0 + 1j])
        mn, mx = np.minmax(a)
        assert_equal(mn, np.min(a))
        assert_equal(mx, np.max(a))

    def test_consistency_with_min_max(self):
        """Verify minmax matches separate min/max calls."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            shape = rng.integers(1, 10, size=rng.integers(1, 4))
            a = rng.standard_normal(shape)
            mn, mx = np.minmax(a)
            assert_equal(mn, np.min(a))
            assert_equal(mx, np.max(a))

    def test_consistency_with_min_max_axis(self):
        """Verify minmax matches separate min/max calls along axes."""
        rng = np.random.default_rng(123)
        a = rng.standard_normal((5, 7, 3))
        for axis in [0, 1, 2, -1, (0, 2)]:
            mn, mx = np.minmax(a, axis=axis)
            assert_array_equal(mn, np.min(a, axis=axis))
            assert_array_equal(mx, np.max(a, axis=axis))

    def test_list_input(self):
        mn, mx = np.minmax([3, 1, 4, 1, 5])
        assert_equal(mn, 1)
        assert_equal(mx, 5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            np.minmax(np.array([]))
