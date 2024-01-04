import os
import pytest

import numpy as np

from . import util


@pytest.fixture(scope="module")
def param_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestParameters",
        sources=[
            util.getpath("tests", "src", "parameter", "constant_real.f90"),
            util.getpath("tests", "src", "parameter", "constant_integer.f90"),
            util.getpath("tests", "src", "parameter", "constant_both.f90"),
            util.getpath("tests", "src", "parameter", "constant_compound.f90"),
            util.getpath("tests", "src", "parameter", "constant_non_compound.f90"),
            util.getpath("tests", "src", "parameter", "constant_array.f90"),
        ],
    )
    return spec


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_real_single(_mod):
    # non-contiguous should raise error
    x = np.arange(6, dtype=np.float32)[::2]
    pytest.raises(ValueError, _mod.foo_single, x)

    # check values with contiguous array
    x = np.arange(3, dtype=np.float32)
    _mod.foo_single(x)
    assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_real_double(_mod):
    # non-contiguous should raise error
    x = np.arange(6, dtype=np.float64)[::2]
    pytest.raises(ValueError, _mod.foo_double, x)

    # check values with contiguous array
    x = np.arange(3, dtype=np.float64)
    _mod.foo_double(x)
    assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_compound_int(_mod):
    # non-contiguous should raise error
    x = np.arange(6, dtype=np.int32)[::2]
    pytest.raises(ValueError, _mod.foo_compound_int, x)

    # check values with contiguous array
    x = np.arange(3, dtype=np.int32)
    _mod.foo_compound_int(x)
    assert np.allclose(x, [0 + 1 + 2 * 6, 1, 2])


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_non_compound_int(_mod):
    # check values
    x = np.arange(4, dtype=np.int32)
    _mod.foo_non_compound_int(x)
    assert np.allclose(x, [0 + 1 + 2 + 3 * 4, 1, 2, 3])


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_integer_int(_mod):
    # non-contiguous should raise error
    x = np.arange(6, dtype=np.int32)[::2]
    pytest.raises(ValueError, _mod.foo_int, x)

    # check values with contiguous array
    x = np.arange(3, dtype=np.int32)
    _mod.foo_int(x)
    assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_integer_long(_mod):
    # non-contiguous should raise error
    x = np.arange(6, dtype=np.int64)[::2]
    pytest.raises(ValueError, _mod.foo_long, x)

    # check values with contiguous array
    x = np.arange(3, dtype=np.int64)
    _mod.foo_long(x)
    assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_both(_mod):
    # non-contiguous should raise error
    x = np.arange(6, dtype=np.float64)[::2]
    pytest.raises(ValueError, _mod.foo, x)

    # check values with contiguous array
    x = np.arange(3, dtype=np.float64)
    _mod.foo(x)
    assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_no(_mod):
    # non-contiguous should raise error
    x = np.arange(6, dtype=np.float64)[::2]
    pytest.raises(ValueError, _mod.foo_no, x)

    # check values with contiguous array
    x = np.arange(3, dtype=np.float64)
    _mod.foo_no(x)
    assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_sum(_mod):
    # non-contiguous should raise error
    x = np.arange(6, dtype=np.float64)[::2]
    pytest.raises(ValueError, _mod.foo_sum, x)

    # check values with contiguous array
    x = np.arange(3, dtype=np.float64)
    _mod.foo_sum(x)
    assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_array(_mod):
    x = np.arange(3, dtype=np.float64)
    y = np.arange(5, dtype=np.float64)
    z = _mod.foo_array(x, y)
    assert np.allclose(x, [0.0, 1.0 / 10, 2.0 / 10])
    assert np.allclose(y, [0.0, 1.0 * 10, 2.0 * 10, 3.0 * 10, 4.0 * 10])
    assert np.allclose(z, 19.0)


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_array_any_index(_mod):
    x = np.arange(6, dtype=np.float64)
    y = _mod.foo_array_any_index(x)
    assert np.allclose(y, x.reshape((2, 3), order="F"))


@pytest.mark.parametrize("_mod", ["param_spec"], indirect=True)
def test_constant_array_delims(_mod):
    x = _mod.foo_array_delims()
    assert x == 9
