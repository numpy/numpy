import numpy as np
import pytest

array_info = np.array_info


# ---------------------------------------------------------------------
# Helper: verify required keys exist in every result
# ---------------------------------------------------------------------
REQUIRED_KEYS = {
    "shape", "dtype", "size", "ndim", "itemsize",
    "contiguous", "aligned", "min", "max", "mean", "std",
    "nan_count", "inf_count", "none_count", "num_unique", "example_values"
}


def _check_keys(info):
    missing = REQUIRED_KEYS - info.keys()
    assert not missing, f"array_info missing keys: {missing}"


# ---------------------------------------------------------------------
# 1. Small numeric int array
# ---------------------------------------------------------------------
def test_small_int():
    a = np.arange(5)
    info = array_info(a)
    _check_keys(info)
    assert info["dtype"].startswith("int")
    assert info["min"] == 0 and info["max"] == 4
    assert info["nan_count"] == info["inf_count"] == 0
    assert info["mean"] == 2


# ---------------------------------------------------------------------
# 2. NaN / Inf
# ---------------------------------------------------------------------
def test_nan_inf():
    a = np.array([1.0, np.nan, np.inf, -np.inf])
    info = array_info(a)
    _check_keys(info)
    assert info["nan_count"] == 1
    assert info["inf_count"] == 2
    assert np.isfinite(info["mean"])


# ---------------------------------------------------------------------
# 3. Complex numbers
# ---------------------------------------------------------------------
def test_complex():
    a = np.array([1+2j, 3+4j])
    info = array_info(a)
    _check_keys(info)
    assert "complex" in info["dtype"]
    assert info["nan_count"] == 0
    assert info["num_unique"] == 2


# ---------------------------------------------------------------------
# 4. Boolean array
# ---------------------------------------------------------------------
def test_bool():
    a = np.array([True, False, True, True])
    info = array_info(a)
    assert info["dtype"] == "bool"
    assert info["mean"] == pytest.approx(0.75)


# ---------------------------------------------------------------------
# 5. Empty array
# ---------------------------------------------------------------------
def test_empty():
    a = np.array([])
    info = array_info(a)
    assert info["size"] == 0
    assert info["num_unique"] == 0
    assert info["min"] is None and info["max"] is None


# ---------------------------------------------------------------------
# 6. Scalar (0-D)
# ---------------------------------------------------------------------
def test_scalar():
    a = np.array(42)
    info = array_info(a)
    assert info["shape"] == ()
    assert info["size"] == 1
    assert info["mean"] == 42


# ---------------------------------------------------------------------
# 7. High-dim (>2)
# ---------------------------------------------------------------------
def test_high_dim():
    a = np.ones((2, 3, 4, 5))
    info = array_info(a)
    assert info["ndim"] == 4
    assert info["contiguous"] is True


# ---------------------------------------------------------------------
# 8. Non-contiguous view
# ---------------------------------------------------------------------
def test_non_contiguous():
    base = np.arange(20)
    view = base[::2]  
    info = array_info(view)
    assert info["contiguous"] is False
    assert info["aligned"] is True


# ---------------------------------------------------------------------
# 9. Large array → sampling path
# ---------------------------------------------------------------------
def test_sampling_path():
    big = np.arange(0, 2_000_000, dtype=np.int32)
    info = array_info(big, sample_unique_threshold=1_000_000)
    assert isinstance(info["num_unique"], str) and info["num_unique"].startswith(">=")


# ---------------------------------------------------------------------
# 10. Object array
# ---------------------------------------------------------------------
def test_object_array():
    a = np.array(["apple", "banana", None, "banana"], dtype=object)
    info = array_info(a)
    assert info["dtype"] == "object"
    assert info["none_count"] == 1
    assert info["min"] is None and info["mean"] is None


# ---------------------------------------------------------------------
# 11. Unhashable objects force 'n/a' unique
# ---------------------------------------------------------------------
def test_unhashable_objects():
    a = np.array([[1, 2], [1, 2]], dtype=object) 
    info = array_info(a)
    assert info["num_unique"] in ("n/a", "n/a (sample)")

# ---------------------------------------------------------------------
# 12. Pure string array (incompatible with numeric ops)
# ---------------------------------------------------------------------
def test_string_array():
    a = np.array(["foo", "bar", "baz", "foo"])
    info = array_info(a)
    assert info["dtype"] == "<U3" or info["dtype"].startswith("<U")  
    for key in ("min", "max", "mean", "std"):
        assert info[key] is None
    assert info["num_unique"] == 3


# ---------------------------------------------------------------------
# 13. Performance sanity on 1-million-element array
# ---------------------------------------------------------------------
def test_performance_1m(monkeypatch):
    import time

    a = np.random.rand(1_000_000) 
    start = time.perf_counter()
    _ = array_info(a)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"array_info too slow: {elapsed:.3f}s"

# ---------------------------------------------------------------------
# 14. Axis-wise mean check
# ---------------------------------------------------------------------
def test_axis_mean():
    a = np.arange(6).reshape(2, 3)
    stats = np.array_info(a, axis=0)["axis_stats"]
    assert (stats["mean"] == np.array([1.5, 2.5, 3.5])).all()

# ---------------------------------------------------------------------
# 15. Percentile & IQR on flattened data
# ---------------------------------------------------------------------
def test_percentiles():
    a = np.arange(1, 5)         
    info = np.array_info(a, percentiles=(25, 50, 75))
    assert info["p25"] == 1.75
    assert info["p75"] == 3.25
    assert info["iqr"] == 1.5
