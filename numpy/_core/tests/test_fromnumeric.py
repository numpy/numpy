import numpy as np
import pytest

def test_cumsum_reverse_1d():
    x = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(np.cumsum(x, reverse=True), np.array([10, 9, 7, 4]))

def test_cumsum_reverse_axis_and_out():
    x = np.arange(6).reshape(2, 3)
    out = np.empty_like(x)
    r = np.cumsum(x, axis=1, out=out, reverse=True)
    assert r is out
    # explicit expected
    exp = np.flip(np.cumsum(np.flip(x, axis=1), axis=1), axis=1)
    np.testing.assert_array_equal(out, exp)
    # Check specific expected values
    np.testing.assert_array_equal(out, np.array([[3, 3, 2],
                                                 [12, 9, 5]]))

def test_cumprod_reverse_basic():
    x = np.array([2, 3, 4])
    np.testing.assert_array_equal(np.cumprod(x, reverse=True), np.array([24, 12, 4]))

def test_cumsum_reverse_dtype():
    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.cumsum(x, dtype=np.int64, reverse=True)
    assert y.dtype == np.int64

def test_cumsum_reverse_none_axis_equiv():
    x = np.arange(12).reshape(3,4)
    r1 = np.cumsum(x, axis=None, reverse=True)
    r2 = np.flip(np.cumsum(np.flip(x.ravel()), axis=0))
    np.testing.assert_array_equal(r1, r2)
