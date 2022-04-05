import pytest
from pytest import param

import numpy as np


def values_and_dtypes():
    """
    Generate value+dtype pairs that generate floating point errors during
    casts.  The invalid casts to integers will generate "invalid" value
    warnings, the float casts all generate "overflow".

    (The Python int/float paths don't need to get tested in all the same
    situations, but it does not hurt.)
    """
    # Casting to float16:
    yield param(70000, "float16", id="int-to-f2")
    yield param("70000", "float16", id="str-to-f2")
    yield param(70000.0, "float16", id="float-to-f2")
    yield param(np.longdouble(70000.), "float16", id="longdouble-to-f2")
    yield param(np.float64(70000.), "float16", id="double-to-f2")
    yield param(np.float32(70000.), "float16", id="float-to-f2")
    # Casting to float32:
    yield param(10**100, "float32", id="int-to-f4")
    yield param(1e100, "float32", id="float-to-f2")
    yield param(np.longdouble(1e300), "float32", id="longdouble-to-f2")
    yield param(np.float64(1e300), "float32", id="double-to-f2")
    # Casting to float64:
    if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
        yield param(np.finfo(np.longdouble).max, "float64",
                    id="longdouble-to-f4")

    # Invalid float to integer casts:
    with np.errstate(over="ignore"):
        for to_dt in np.typecodes["AllInteger"]:
            for value in [np.inf, np.nan]:
                for from_dt in np.typecodes["AllFloat"]:
                    from_dt = np.dtype(from_dt)
                    from_val = from_dt.type(value)

                    yield param(from_val, to_dt, id=f"{from_val}-to-{to_dt}")


def check_operations(dtype, value):
    """
    There are many dedicated paths in NumPy which cast and should check for
    floating point errors which occurred during those casts.
    """
    if dtype.kind != 'i':
        def assignment():
            arr = np.empty(3, dtype=dtype)
            arr[0] = value

        yield assignment

        def fill():
            arr = np.empty(3, dtype=dtype)
            arr.fill(value)

        yield fill

    def copyto_scalar():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, value, casting="unsafe")

    yield copyto_scalar

    def copyto():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, np.array([value, value, value]), casting="unsafe")

    yield copyto

    def copyto_scalar_masked():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, value, casting="unsafe",
                  where=[True, False, True])

    yield copyto_scalar_masked

    def copyto_masked():
        arr = np.empty(3, dtype=dtype)
        np.copyto(arr, np.array([value, value, value]), casting="unsafe",
                  where=[True, False, True])

    yield copyto_masked

    def direct_cast():
        np.array([value, value, value]).astype(dtype)

    yield direct_cast

    def direct_cast_nd_strided():
        arr = np.full((5, 5, 5), fill_value=value)[:, ::2, :]
        arr.astype(dtype)

    yield direct_cast_nd_strided

    def boolean_array_assignment():
        arr = np.empty(3, dtype=dtype)
        arr[[True, False, True]] = np.array([value, value])

    yield boolean_array_assignment

    def integer_array_assignment():
        arr = np.empty(3, dtype=dtype)
        values = np.array([value, value])

        arr[[0, 1]] = values

    #yield integer_array_assignment

    def integer_array_assignment_with_subspace():
        arr = np.empty((5, 3), dtype=dtype)
        values = np.array([value, value, value])

        arr[[0, 2]] = values

    yield integer_array_assignment_with_subspace


@pytest.mark.parametrize(["value", "dtype"], values_and_dtypes())
@pytest.mark.filterwarnings("ignore::numpy.ComplexWarning")
def test_floatingpoint_errors_casting(dtype, value):
    dtype = np.dtype(dtype)
    for operation in check_operations(dtype, value):
        dtype = np.dtype(dtype)

        match = "invalid" if dtype.kind in 'iu' else "overflow"
        with pytest.warns(RuntimeWarning, match=match):
            operation()

        with np.errstate(all="raise"):
            with pytest.raises(FloatingPointError, match=match):
                operation()
