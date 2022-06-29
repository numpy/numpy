"""
This file adds basic tests to test the NEP 50 style promotion compatibility
mode.  Most of these test are likely to be simply deleted again once NEP 50
is adopted in the main test suite.  A few may be moved elsewhere.
"""

import numpy as np
import pytest


@pytest.fixture(scope="module", autouse=True)
def _weak_promotion_enabled():
    state = np._get_promotion_state()
    np._set_promotion_state("weak_and_warn")
    yield
    np._set_promotion_state(state)


def test_nep50_examples():
    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.uint8(1) + 2
    assert res.dtype == np.uint8

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([1], np.uint8) + np.int64(1)
    assert res.dtype == np.int64

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([1], np.uint8) + np.array(1, dtype=np.int64)
    assert res.dtype == np.int64

    with pytest.warns(UserWarning, match="result dtype changed"):
        # Note: Overflow would be nice, but does not warn with change warning
        with np.errstate(over="raise"):
            res = np.uint8(100) + 200
    assert res.dtype == np.uint8

    with pytest.warns(Warning) as recwarn:
        res = np.float32(1) + 3e100

    # Check that both warnings were given in the one call:
    warning = str(recwarn.pop(UserWarning).message)
    assert warning.startswith("result dtype changed")
    warning = str(recwarn.pop(RuntimeWarning).message)
    assert warning.startswith("overflow")
    assert len(recwarn) == 0  # no further warnings
    assert np.isinf(res)
    assert res.dtype == np.float32

    # Changes, but we don't warn for it (too noisy)
    res = np.array([0.1], np.float32) == np.float64(0.1)
    assert res[0] == False

    # Additional test, since the above silences the warning:
    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([0.1], np.float32) + np.float64(0.1)
    assert res.dtype == np.float64

    with pytest.warns(UserWarning, match="result dtype changed"):
        res = np.array([1.], np.float32) + np.int64(3)
    assert res.dtype == np.float64


def test_nep50_without_warnings():
    # Test that avoid the "warn" method, since that may lead to different
    # code paths in some cases.
    # Set promotion to weak (no warning), the auto-fixture will reset it.
    np._set_promotion_state("weak")
    with np.errstate(over="warn"):
        with pytest.warns(RuntimeWarning):
            res = np.uint8(100) + 200
    assert res.dtype == np.uint8

    with pytest.warns(RuntimeWarning):
        res = np.float32(1) + 3e100
    assert res.dtype == np.float32


def test_nep50_integer_conversion_errors():
    # Do not worry about warnings here (auto-fixture will reset).
    np._set_promotion_state("weak")
    # Implementation for error paths is mostly missing (as of writing)
    with pytest.raises(OverflowError, match=".*uint8"):
        np.array([1], np.uint8) + 300

    with pytest.raises(OverflowError, match=".*uint8"):
        np.uint8(1) + 300

    with pytest.raises(OverflowError, match=".*unsigned int"):
        np.uint8(1) + -1
