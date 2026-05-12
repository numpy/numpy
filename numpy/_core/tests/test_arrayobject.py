import sys

import pytest

import numpy as np
from numpy._core._rational_tests import rational
from numpy.testing import HAS_REFCOUNT, assert_array_equal


def test_matrix_transpose_raises_error_for_1d():
    msg = "matrix transpose with ndim < 2 is undefined"
    arr = np.arange(48)
    with pytest.raises(ValueError, match=msg):
        arr.mT


def test_matrix_transpose_equals_transpose_2d():
    arr = np.arange(48).reshape((6, 8))
    assert_array_equal(arr.T, arr.mT)


ARRAY_SHAPES_TO_TEST = (
    (5, 2),
    (5, 2, 3),
    (5, 2, 3, 4),
)


@pytest.mark.parametrize("shape", ARRAY_SHAPES_TO_TEST)
def test_matrix_transpose_equals_swapaxes(shape):
    num_of_axes = len(shape)
    vec = np.arange(shape[-1])
    arr = np.broadcast_to(vec, shape)
    tgt = np.swapaxes(arr, num_of_axes - 2, num_of_axes - 1)
    mT = arr.mT
    assert_array_equal(tgt, mT)


class MyArr(np.ndarray):
    def __array_wrap__(self, arr, context=None, return_scalar=None):
        return super().__array_wrap__(arr, context, return_scalar)


class MyArrNoWrap(np.ndarray):
    pass


@pytest.mark.parametrize("subclass_self", [np.ndarray, MyArr, MyArrNoWrap])
@pytest.mark.parametrize("subclass_arr", [np.ndarray, MyArr, MyArrNoWrap])
def test_array_wrap(subclass_self, subclass_arr):
    # NumPy should allow `__array_wrap__` to be called on arrays, it's logic
    # is designed in a way that:
    #
    # * Subclasses never return scalars by default (to preserve their
    #   information).  They can choose to if they wish.
    # * NumPy returns scalars, if `return_scalar` is passed as True to allow
    #   manual calls to `arr.__array_wrap__` to do the right thing.
    # * The type of the input should be ignored (it should be a base-class
    #   array, but I am not sure this is guaranteed).

    arr = np.arange(3).view(subclass_self)

    arr0d = np.array(3, dtype=np.int8).view(subclass_arr)
    # With third argument True, ndarray allows "decay" to scalar.
    # (I don't think NumPy would pass `None`, but it seems clear to support)
    if subclass_self is np.ndarray:
        assert type(arr.__array_wrap__(arr0d, None, True)) is np.int8
    else:
        assert type(arr.__array_wrap__(arr0d, None, True)) is type(arr)

    # Otherwise, result should be viewed as the subclass
    assert type(arr.__array_wrap__(arr0d)) is type(arr)
    assert type(arr.__array_wrap__(arr0d, None, None)) is type(arr)
    assert type(arr.__array_wrap__(arr0d, None, False)) is type(arr)

    # Non 0-D array can't be converted to scalar, so we ignore that
    arr1d = np.array([3], dtype=np.int8).view(subclass_arr)
    assert type(arr.__array_wrap__(arr1d, None, True)) is type(arr)


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_cleanup_with_refs_non_contig():
    # Regression test, leaked the dtype (but also good for rest)
    dtype = np.dtype("O,i")
    obj = object()
    expected_ref_dtype = sys.getrefcount(dtype)
    expected_ref_obj = sys.getrefcount(obj)
    proto = np.full((3, 4, 5, 6, 7), np.array((obj, 2), dtype=dtype))
    # Give array a non-trivial order to exercise more cleanup paths.
    arr = proto.transpose((2, 0, 3, 1, 4)).copy("K")
    del proto, arr

    actual_ref_dtype = sys.getrefcount(dtype)
    actual_ref_obj = sys.getrefcount(obj)
    assert actual_ref_dtype == expected_ref_dtype
    assert actual_ref_obj == actual_ref_dtype


@pytest.mark.parametrize("dtype",
    list("?bhilqnpBHILQNPefdgSUV") + ["M8[ns]", "m8[ns]", rational])
def test_real_imag_attributes_non_complex(dtype):
    dtype = np.dtype(dtype)

    a = np.array([[1, 2, 3], [4, 5, 6]]).astype(dtype)
    assert a.real is a
    # One could imagine broadcasting, but doesn't right now:
    imag = a.imag
    assert imag.strides == a.strides
    assert imag.dtype == a.dtype
    # This part is rather unclear:
    assert (imag == np.zeros((), dtype=a.dtype)).all()
    assert imag.flags.writeable is False

    class myarr(np.ndarray):
        def __array_finalize__(self, obj):
            self.finalized_with = obj

    ma = a.view(myarr)
    assert ma.real is ma
    assert type(ma.imag) is myarr
    assert ma.imag.finalized_with is ma


@pytest.mark.parametrize("dtype,real_dt",
    [(">c8", ">f4"), ("c16", "f8"), ("clongdouble", "longdouble")])
@pytest.mark.parametrize("variation", ["transpose", "set_writeable"])
def test_real_imag_attributes_complex(dtype, real_dt, variation):
    a = np.array([[1, 2j, 3], [4, 5j, 6]]).astype(dtype)
    real = np.array([[1, 0, 3], [4, 0, 6]], dtype=real_dt)
    imag = np.array([[0, 2, 0], [0, 5, 0]], dtype=real_dt)

    if variation == "transpose":
        a = a.T
        real = real.T
        imag = imag.T
    elif variation == "set_writeable":
        a.flags.writeable = False

    assert_array_equal(a.real, real)
    assert_array_equal(a.imag, imag)
    assert a.real.dtype == real_dt
    assert a.imag.dtype == real_dt
    assert np.may_share_memory(a.real, a)
    assert np.may_share_memory(a.imag, a)
    assert a.real.flags.writeable == a.flags.writeable
    assert a.imag.flags.writeable == a.flags.writeable

    class myarr(np.ndarray):
        def __array_finalize__(self, obj):
            self.finalized_with = obj

    ma = a.view(myarr)
    assert ma.real.finalized_with is ma
    assert ma.imag.finalized_with is ma


def test_real_imag_attributes_object():
    a = np.array([[1, 0.5 + 2j, 3, int], [4, 5j, "string", {}]], dtype=object)

    # NOTE(seberg): doing something for non-numbers is guesswork...
    real = np.array([[1, 0.5, 3, int.real], [4, 0, "string", {}]], dtype=object)
    imag = np.array([[0, 2, 0, int.imag], [0, 5, 0, 0]], dtype=object)

    assert_array_equal(a.real, real)
    assert_array_equal(a.imag, imag)
    assert a.real.dtype == object
    assert a.imag.dtype == object
    assert not np.may_share_memory(a.real, a)
    assert not np.may_share_memory(a.imag, a)
    assert not a.real.flags.writeable
    assert not a.imag.flags.writeable

    # Object returns new arrays via ufuncs, so call wrap
    class myarr(np.ndarray):
        def __array_wrap__(self, *args, **kwargs):
            ret = super().__array_wrap__(*args, **kwargs)
            ret.wrap_called = True
            return ret

    ma = a.view(myarr)
    assert ma.real.wrap_called
    assert ma.imag.wrap_called


@pytest.mark.parametrize("ufunc,attr", [
    (np._core.umath.real, "real"), (np._core.umath.imag, "imag")])
def test_real_imag_ufunc_minimal(ufunc, attr):
    with pytest.raises(TypeError):
        ufunc(np.array([1, 2, 3]))  # non-complex or object raises

    arr = np.array([1 + 2j, 3 + 4j])
    res = ufunc(arr)
    assert_array_equal(res, getattr(arr, attr), strict=True)

    arr = np.array([1 + 2j, 3 + 4j], dtype=object)
    res = ufunc(arr)
    assert_array_equal(res, getattr(arr, attr), strict=True)
