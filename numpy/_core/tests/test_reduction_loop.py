import itertools

import pytest

import numpy as np
from numpy._core._reduction_loop_tests import (
    minimum_intp_maximum,
    minimum_object_maximum,
    minimummaximum as mm,
    minimummaximum_with_identity as mmi,
)

SHAPES = [(12,), (3, 4), (2, 3, 4)]
EMPTY_SHAPES = [(0,), (0, 3), (3, 0), (2, 0, 4)]
SPECIALS = {
    "nan": [np.nan],
    "inf": [np.inf, -np.inf],
    "nan_inf": [np.nan, np.inf, -np.inf],
}


def make_array(shape, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 10).astype(np.float64)


def check_minmax(method, *args, ufunc=mm, max_dtype=None, **kwargs):
    """
    Compare ``ufunc.<method>`` against ``np.minimum.<method>`` and
    ``np.maximum.<method>``. `out`, if passed, only goes to `ufunc`. The
    references are computed first, since it may overlap the input.
    """
    ref_kwargs = {k: v for k, v in kwargs.items() if k != "out"}
    ref_min = getattr(np.minimum, method)(*args, **ref_kwargs)
    ref_max = getattr(np.maximum, method)(*args, **ref_kwargs)
    got = getattr(ufunc, method)(*args, **kwargs)
    out = kwargs.get("out")
    assert isinstance(got, tuple) and len(got) == 2
    if out is not None:
        for res, given in zip(got, out):
            assert given is None or res is given
    got_min, got_max = got
    if max_dtype is not None:
        assert got_min.dtype == np.float64 and got_max.dtype == max_dtype
        ref_max = ref_max.astype(max_dtype)
    np.testing.assert_array_equal(got_min, ref_min)
    np.testing.assert_array_equal(got_max, ref_max)


def reduce_axes(ndim):
    axes = [None]
    for r in range(1, ndim + 1):
        axes.extend(itertools.combinations(range(ndim), r))
    if ndim >= 1:
        axes.append(-1)
    return [ax[0] if isinstance(ax, tuple) and len(ax) == 1 else ax
            for ax in axes]


def method_axes(method, ndim):
    if method == "reduce":
        return reduce_axes(ndim)
    return [*range(ndim), -1]


def method_args(method, a, axis=0):
    """
    Positional and keyword arguments for ``ufunc.<method>`` over `a`.
    `reduceat` gets two segments along `axis`, or one if it has length 1.
    """
    if method in ("reduce", "accumulate"):
        return (a,), {"axis": axis}
    n = a.shape[axis]
    return (a, [0] if n == 1 else [0, n // 2]), {"axis": axis}


def method_out_shape(method, args, kwargs):
    return np.shape(getattr(np.minimum, method)(*args, **kwargs))


def reduces_over_empty(shape, axis):
    ndim = len(shape)
    if axis is None:
        reduced = set(range(ndim))
    elif isinstance(axis, tuple):
        reduced = {ax % ndim for ax in axis}
    else:
        reduced = {axis % ndim}
    empty_reduced = any(shape[ax] == 0 for ax in reduced)
    result_nonempty = all(shape[i] > 0 for i in range(ndim) if i not in reduced)
    return empty_reduced and result_nonempty


class TestReductionLoop:
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_forward(self, shape):
        a = make_array(shape, seed=1)
        b = make_array(shape, seed=2)
        check_minmax("__call__", a, b)

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_forward_strided(self, shape):
        a = make_array(shape, seed=1)[::-1]
        b = make_array(shape, seed=2)[::-1]
        check_minmax("__call__", a, b)

    def test_forward_mixed_dtype_inputs(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([2.5, 1.5, 0.5], dtype=np.float64)
        check_minmax("__call__", a, b, max_dtype=np.float64)

    @pytest.mark.parametrize("kind", list(SPECIALS))
    def test_forward_specials(self, kind):
        vals = [1.0, -2.0, 3.5, 0.0, -1.0] + SPECIALS[kind]
        a = np.array(vals, dtype=np.float64)
        b = np.array(vals[::-1], dtype=np.float64)
        check_minmax("__call__", a, b)

    def test_forward_python_scalars(self):
        got_min, got_max = mm(3.0, 5.0)
        assert (got_min, got_max) == (3.0, 5.0)

    def test_forward_python_ints(self):
        got_min, got_max = mm(3, 5)
        assert (got_min, got_max) == (3.0, 5.0)
        assert got_min.dtype == np.float64

    def test_forward_dtype_propagates_to_inputs(self):
        got_min, got_max = mm(1, 5, dtype=np.float64)
        assert got_min == 1.0 and got_max == 5.0

    def test_scalar_and_0d(self):
        v1 = make_array((), seed=13).item()
        v2 = make_array((), seed=14).item()
        for a, b in ((np.array(v1), np.array(v2)),
                     (np.float64(v1), np.float64(v2))):
            check_minmax("__call__", a, b)
            check_minmax("reduce", a)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_reduce_like(self, method, shape):
        a = make_array(shape, seed=3)
        for axis in method_axes(method, a.ndim):
            args, kwargs = method_args(method, a, axis)
            check_minmax(method, *args, **kwargs)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_strided(self, method, shape):
        a = make_array(shape, seed=15)[::-1]
        for axis in method_axes(method, a.ndim):
            args, kwargs = method_args(method, a, axis)
            check_minmax(method, *args, **kwargs)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    @pytest.mark.parametrize("kind", list(SPECIALS))
    def test_specials(self, method, kind):
        vals = [1.0, -2.0, 3.5, 0.0, -1.0] + SPECIALS[kind]
        args, kwargs = method_args(method, np.array(vals, dtype=np.float64))
        check_minmax(method, *args, **kwargs)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    def test_single_element(self, method):
        args, kwargs = method_args(method, np.array([7.0]))
        check_minmax(method, *args, **kwargs)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_out_tuple(self, method, shape):
        a = make_array(shape, seed=8)
        for axis in method_axes(method, a.ndim):
            args, kwargs = method_args(method, a, axis)
            out_shape = method_out_shape(method, args, kwargs)
            out = (np.empty(out_shape), np.empty(out_shape))
            check_minmax(method, *args, **kwargs, out=out)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    def test_out_overlapping_input(self, method):
        # `out` views into the reduced array make the iterator use a writeback
        # temporary, but the arrays that were passed in must still be the ones
        # returned and updated.
        flat = np.arange(72, dtype=np.float64)
        a = flat[:24].reshape(6, 4)
        args, kwargs = method_args(method, a)
        out_shape = method_out_shape(method, args, kwargs)
        n = int(np.prod(out_shape))
        out = (flat[1:n + 1].reshape(out_shape),
               flat[n + 1:2 * n + 1].reshape(out_shape))
        check_minmax(method, *args, **kwargs, out=out)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    @pytest.mark.parametrize("out", [None, (None, None)])
    def test_out_none(self, method, out):
        # `out=None` means no output was given, as for single-output methods.
        args, kwargs = method_args(method, make_array((4,), seed=9))
        check_minmax(method, *args, **kwargs, out=out)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    @pytest.mark.parametrize("which", [0, 1])
    def test_out_partial(self, method, which):
        # Only some entries of the `out` tuple given, the rest allocated.
        args, kwargs = method_args(method, make_array((3, 4), seed=9))
        out = [None, None]
        out[which] = np.empty(method_out_shape(method, args, kwargs))
        check_minmax(method, *args, **kwargs, out=tuple(out))

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    def test_out_bare_array_raises(self, method):
        args, kwargs = method_args(method, make_array((4,), seed=9))
        with pytest.raises(TypeError, match="must be a tuple of arrays"):
            getattr(mm, method)(*args, **kwargs, out=np.empty(2))

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    def test_out_wrong_length_raises(self, method):
        args, kwargs = method_args(method, make_array((4,), seed=10))
        with pytest.raises(ValueError, match="exactly one entry per ufunc output"):
            getattr(mm, method)(*args, **kwargs, out=(np.empty(2),))

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    def test_dtype_same(self, method):
        args, kwargs = method_args(method, make_array((4,), seed=16))
        check_minmax(method, *args, **kwargs, dtype=np.float64,
                     max_dtype=np.float64)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    def test_dtype_forced_no_loop_raises(self, method):
        args, kwargs = method_args(method, make_array((4,), seed=17))
        with pytest.raises(TypeError, match="did not contain a loop"):
            getattr(mm, method)(*args, **kwargs, dtype=np.int64)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    def test_dtype_mismatched_tuple_raises(self, method):
        args, kwargs = method_args(method, make_array((4,), seed=18))
        with pytest.raises(ValueError, match="mismatch in size"):
            getattr(mm, method)(*args, **kwargs, dtype=(np.int32, np.int64))

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    def test_unsupported_dtype_raises(self, method):
        args, kwargs = method_args(method, np.array(["a", "b", "c"]))
        with pytest.raises(ValueError, match="could not convert string to float"):
            getattr(mm, method)(*args, **kwargs)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    def test_no_reduction_loop_raises(self, method):
        args, kwargs = method_args(method, np.array([1, 2, 3]))
        with pytest.raises(
                TypeError, match="resolved loop does not register a reduction loop"):
            getattr(np.divmod, method)(*args, **kwargs)

    # accumulate, where acc and out differ, must skip the in-place SIMD paths.
    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    @pytest.mark.parametrize("dtype", ["f8", "f4", "i8", "i1"])
    @pytest.mark.parametrize("step", [1, 2])
    def test_builtin_minimummaximum(self, method, dtype, step):
        a = np.random.default_rng(46).integers(-50, 51, size=(4, 48))
        a = a.astype(dtype)[:, ::step]
        for axis in method_axes(method, a.ndim):
            args, kwargs = method_args(method, a, axis)
            check_minmax(method, *args, **kwargs,
                         ufunc=np._core.umath.minimummaximum)

    # `minimummaximum` also registers an object loop, so the reduction
    # machinery is exercised with refcounted (NPY_ITEM_REFCOUNT) descriptors.
    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_object(self, method, shape):
        a = np.random.default_rng(30).integers(-50, 51, size=shape).astype(object)
        for axis in method_axes(method, a.ndim):
            args, kwargs = method_args(method, a, axis)
            check_minmax(method, *args, **kwargs)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    def test_object_incomparable_raises(self, method):
        args, kwargs = method_args(method, np.array([1, "x", 2], dtype=object))
        with pytest.raises(TypeError):
            getattr(mm, method)(*args, **kwargs)

    # The second output of the mixed ufuncs is the maximum as intp/object, so
    # the first element of each reduction is cast from float64 into it.
    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    @pytest.mark.parametrize("ufunc, max_dtype", [
        (minimum_intp_maximum, np.intp), (minimum_object_maximum, object)],
        ids=["intp", "object"])
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_mixed(self, method, ufunc, max_dtype, shape):
        a = make_array(shape, seed=42)
        for axis in method_axes(method, a.ndim):
            args, kwargs = method_args(method, a, axis)
            if method == "reduce":
                # A full object reduction returns the Python object, which
                # has no dtype to check.
                kwargs["keepdims"] = True
            check_minmax(method, *args, **kwargs, ufunc=ufunc,
                         max_dtype=max_dtype)

    @pytest.mark.parametrize("method", ["reduce", "reduceat", "accumulate"])
    @pytest.mark.parametrize("ufunc, max_dtype", [
        (minimum_intp_maximum, np.intp), (minimum_object_maximum, object)],
        ids=["intp", "object"])
    def test_mixed_out_strided(self, method, ufunc, max_dtype):
        args, kwargs = method_args(method, make_array((12,), seed=45)[::-1])
        out_shape = method_out_shape(method, args, kwargs)
        out = (np.empty(out_shape, np.float64), np.empty(out_shape, max_dtype))
        check_minmax(method, *args, **kwargs, ufunc=ufunc, max_dtype=max_dtype,
                     out=out)

    @pytest.mark.parametrize("ufunc, max_dtype", [
        (minimum_intp_maximum, np.intp), (minimum_object_maximum, object)],
        ids=["intp", "object"])
    def test_mixed_forward(self, ufunc, max_dtype):
        a, b = make_array((12,), seed=40), make_array((12,), seed=41)
        check_minmax("__call__", a, b, ufunc=ufunc, max_dtype=max_dtype)

    def test_at_raises(self):
        a = make_array((4,), seed=22)
        with pytest.raises(ValueError, match="single output"):
            mm.at(a, [0], a)

    def test_outer(self):
        a = make_array((3,), seed=23)
        b = make_array((4,), seed=24)
        check_minmax("outer", a, b)

    def test_object_forward(self):
        a = np.array([3, -7, 12], dtype=object)
        b = np.array([1, 9, -2], dtype=object)
        got_min, got_max = mm(a, b)
        assert got_min.tolist() == [1, -7, -2]
        assert got_max.tolist() == [3, 9, 12]


class TestReduce:
    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_reduce_keepdims(self, shape):
        a = make_array(shape, seed=3)
        for axis in reduce_axes(a.ndim):
            kwargs = {"axis": axis, "keepdims": True}
            out_shape = method_out_shape("reduce", (a,), kwargs)
            check_minmax("reduce", a, **kwargs)
            check_minmax("reduce", a, **kwargs,
                         out=(np.empty(out_shape), np.empty(out_shape)))

    @pytest.mark.parametrize("initial_kind", ["small", "large"])
    def test_reduce_initial_scalar(self, initial_kind):
        a = make_array((3, 4), seed=5)
        initial = float(a.min()) - 1.0 if initial_kind == "small" \
            else float(a.max()) + 1.0
        for axis in reduce_axes(a.ndim):
            check_minmax("reduce", a, axis=axis, initial=initial)

    def test_reduce_initial_tuple(self):
        a = make_array((3, 4), seed=6)
        small = float(a.min()) - 1.0
        large = float(a.max()) + 1.0
        for axis in reduce_axes(a.ndim):
            got_min, got_max = mm.reduce(a, axis=axis, initial=(small, large))
            np.testing.assert_array_equal(
                got_min, np.minimum.reduce(a, axis=axis, initial=small))
            np.testing.assert_array_equal(
                got_max, np.maximum.reduce(a, axis=axis, initial=large))

    def test_reduce_initial_wrong_length_raises(self):
        a = make_array((4,), seed=7)
        with pytest.raises(ValueError, match="one entry per reduction output"):
            mm.reduce(a, initial=(1.0, 2.0, 3.0))

    @pytest.mark.parametrize("initial", [(None, 5.0), (5.0, None), (None, None)])
    def test_reduce_initial_tuple_none_raises(self, initial):
        # `initial=None` means "no initial value", which cannot be expressed
        # per-output, so it must not be packed as a value (NaN for floats).
        a = make_array((4,), seed=7)
        with pytest.raises(ValueError, match="cannot be None"):
            mm.reduce(a, initial=initial)

    def test_reduce_initial_scalar_none(self):
        # A scalar None is still "no initial value", as for single-output
        # reductions, so it falls back to seeding from the first element.
        check_minmax("reduce", make_array((4,), seed=7), initial=None)

    def test_reduce_out_and_initial(self):
        check_minmax("reduce", make_array((4,), seed=19), initial=100.0,
                     out=(np.empty(()), np.empty(())))

    def test_object_reduce_initial(self):
        # `initial` is packed into refcounted buffers, which the reduction
        # machinery has to clear again.
        a = np.random.default_rng(31).integers(-50, 51, size=12).astype(object)
        got_min, got_max = mm.reduce(a, initial=(7, -7))
        assert got_min == np.minimum.reduce(a, initial=7)
        assert got_max == np.maximum.reduce(a, initial=-7)

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_reduce_where(self, shape):
        a = make_array(shape, seed=11)
        mask = np.random.default_rng(12).integers(0, 2, size=shape).astype(bool)
        initial = float(a.max()) + 1.0
        for axis in reduce_axes(a.ndim):
            check_minmax("reduce", a, axis=axis, where=mask, initial=initial)

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_reduce_where_uses_registered_identity(self, shape):
        # Without `initial`, the registered identity seeds the masked-out
        # slots, including slots where the mask excludes everything.
        a = make_array(shape, seed=11)
        mask = np.random.default_rng(12).integers(0, 2, size=shape).astype(bool)
        for axis in reduce_axes(a.ndim):
            got_min, got_max = mmi.reduce(a, axis=axis, where=mask)
            np.testing.assert_array_equal(
                got_min,
                np.minimum.reduce(a, axis=axis, where=mask, initial=np.inf))
            np.testing.assert_array_equal(
                got_max,
                np.maximum.reduce(a, axis=axis, where=mask, initial=-np.inf))

    def test_reduce_where_fully_masked_gives_identity(self):
        a = make_array((3, 4), seed=11)
        mask = np.array([[True] * 4, [False] * 4, [True] * 4])
        got_min, got_max = mmi.reduce(a, axis=1, where=mask)
        assert got_min[1] == np.inf and got_max[1] == -np.inf

    @pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=str)
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_reduce_empty_no_initial_raises(self, shape, keepdims):
        a = np.zeros(shape, np.float64)
        for axis in reduce_axes(a.ndim):
            if not reduces_over_empty(shape, axis):
                continue
            for op in (np.minimum, np.maximum, mm):
                with pytest.raises(
                        ValueError, match="zero-size array to reduction operation"):
                    op.reduce(a, axis=axis, keepdims=keepdims)

    @pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=str)
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_reduce_empty_uses_registered_identity(self, shape, keepdims):
        a = np.zeros(shape, np.float64)
        for axis in reduce_axes(a.ndim):
            if not reduces_over_empty(shape, axis):
                continue
            got_min, got_max = mmi.reduce(a, axis=axis, keepdims=keepdims)
            np.testing.assert_array_equal(
                got_min,
                np.minimum.reduce(a, axis=axis, keepdims=keepdims, initial=np.inf))
            np.testing.assert_array_equal(
                got_max,
                np.maximum.reduce(a, axis=axis, keepdims=keepdims, initial=-np.inf))

    def test_reduce_empty_returns_identity(self):
        got_min, got_max = mmi.reduce(np.array([], dtype=np.float64))
        assert got_min == np.inf
        assert got_max == -np.inf

    @pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=str)
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_reduce_empty_axis_survives(self, shape, keepdims):
        a = np.zeros(shape, np.float64)
        for axis in reduce_axes(a.ndim):
            if reduces_over_empty(shape, axis):
                continue
            check_minmax("reduce", a, axis=axis, keepdims=keepdims)

    @pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=str)
    def test_reduce_empty_scalar_initial(self, shape):
        a = np.zeros(shape, np.float64)
        for axis in reduce_axes(a.ndim):
            check_minmax("reduce", a, axis=axis, initial=0.0)

    @pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=str)
    def test_reduce_empty_tuple_initial(self, shape):
        a = np.zeros(shape, np.float64)
        for axis in reduce_axes(a.ndim):
            got_min, got_max = mm.reduce(a, axis=axis, initial=(0.0, 1.0))
            np.testing.assert_array_equal(
                got_min, np.minimum.reduce(a, axis=axis, initial=0.0))
            np.testing.assert_array_equal(
                got_max, np.maximum.reduce(a, axis=axis, initial=1.0))


class TestReduceat:
    def test_reduceat_empty(self):
        check_minmax("reduceat", np.array([], np.float64), [])

    def test_reduceat_identity_ignored(self):
        # reduceat seeds each segment with its first element, so a registered
        # identity must not change the result.
        check_minmax("reduceat", make_array((10,), seed=25), [0, 4, 7], ufunc=mmi)

    @pytest.mark.parametrize("idx", [[0, 0, 3], [3, 1], [0, 5], [5],
                                     [0, 2, 2, 4], [4, 0]])
    def test_reduceat_repeated_and_unordered_indices(self, idx):
        # Segments that are empty or run backwards return the element at the
        # index itself, which the reduction loop must not fold over.
        check_minmax("reduceat", make_array((6,), seed=29), idx)

    @pytest.mark.parametrize("ufunc, max_dtype", [
        (minimum_intp_maximum, np.intp), (minimum_object_maximum, object)],
        ids=["intp", "object"])
    @pytest.mark.parametrize("idx", [[0, 0, 3], [3, 1], [5], [4, 0]])
    def test_mixed_reduceat_repeated_and_unordered_indices(
            self, ufunc, max_dtype, idx):
        # Empty and backwards segments are only their first element, cast to
        # the second output's dtype.
        check_minmax("reduceat", make_array((6,), seed=44), idx, ufunc=ufunc,
                     max_dtype=max_dtype)

    def test_reduceat_error_stops_later_segments(self):
        # The loop error in the first segment must stop the reduction, so
        # the second segment of `out` is never written.
        a = np.array([1, "x", 2, 3], dtype=object)
        out = (np.full(2, None, object), np.full(2, None, object))
        with pytest.raises(TypeError):
            mm.reduceat(a, [0, 2], out=out)
        assert out[0][1] is None and out[1][1] is None

    def test_reduceat_out_of_bounds_index_raises(self):
        a = make_array((4,), seed=21)
        with pytest.raises(IndexError, match="out-of-bounds"):
            mm.reduceat(a, [0, 9])


class TestAccumulate:
    def test_accumulate_empty(self):
        check_minmax("accumulate", np.array([], np.float64))

    def test_accumulate_identity_ignored(self):
        # accumulate always seeds with the first element, so a registered
        # identity must not change the result.
        check_minmax("accumulate", make_array((10,), seed=25), ufunc=mmi)

    def test_accumulate_error_stops_later_rows(self):
        # The loop error in the first row must stop the accumulation, so
        # the second row of `out` is never written.
        a = np.array([[1, "x"], [2, 3]], dtype=object)
        out = (np.full((2, 2), None, object), np.full((2, 2), None, object))
        with pytest.raises(TypeError):
            mm.accumulate(a, axis=1, out=out)
        assert out[0][1, 0] is None and out[1][1, 0] is None
