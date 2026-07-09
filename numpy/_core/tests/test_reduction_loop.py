import itertools

import pytest

import numpy as np
from numpy._core._reduction_loop_tests import minimummaximum as mm

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


def reduce_axes(ndim):
    axes = [None]
    for r in range(1, ndim + 1):
        axes.extend(itertools.combinations(range(ndim), r))
    if ndim >= 1:
        axes.append(-1)
    return [ax[0] if isinstance(ax, tuple) and len(ax) == 1 else ax
            for ax in axes]


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
        got_min, got_max = mm(a, b)
        np.testing.assert_array_equal(got_min, np.minimum(a, b))
        np.testing.assert_array_equal(got_max, np.maximum(a, b))

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_forward_strided(self, shape):
        a = make_array(shape, seed=1)[::-1]
        b = make_array(shape, seed=2)[::-1]
        got_min, got_max = mm(a, b)
        np.testing.assert_array_equal(got_min, np.minimum(a, b))
        np.testing.assert_array_equal(got_max, np.maximum(a, b))

    def test_forward_mixed_dtype_inputs(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([2.5, 1.5, 0.5], dtype=np.float64)
        got_min, got_max = mm(a, b)
        assert got_min.dtype == np.float64
        np.testing.assert_array_equal(got_min, np.minimum(a, b))
        np.testing.assert_array_equal(got_max, np.maximum(a, b))

    @pytest.mark.parametrize("kind", list(SPECIALS))
    def test_forward_specials(self, kind):
        vals = [1.0, -2.0, 3.5, 0.0, -1.0] + SPECIALS[kind]
        a = np.array(vals, dtype=np.float64)
        b = np.array(vals[::-1], dtype=np.float64)
        got_min, got_max = mm(a, b)
        np.testing.assert_array_equal(got_min, np.minimum(a, b))
        np.testing.assert_array_equal(got_max, np.maximum(a, b))

    def test_scalar_and_0d(self):
        v1 = make_array((), seed=13).item()
        v2 = make_array((), seed=14).item()
        for a, b in ((np.array(v1), np.array(v2)),
                     (np.float64(v1), np.float64(v2))):
            got_min, got_max = mm(a, b)
            np.testing.assert_array_equal(got_min, np.minimum(a, b))
            np.testing.assert_array_equal(got_max, np.maximum(a, b))
            red_min, red_max = mm.reduce(a)
            np.testing.assert_array_equal(red_min, np.minimum.reduce(a))
            np.testing.assert_array_equal(red_max, np.maximum.reduce(a))

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_reduce(self, shape, keepdims):
        a = make_array(shape, seed=3)
        for axis in reduce_axes(a.ndim):
            got_min, got_max = mm.reduce(a, axis=axis, keepdims=keepdims)
            np.testing.assert_array_equal(
                got_min, np.minimum.reduce(a, axis=axis, keepdims=keepdims))
            np.testing.assert_array_equal(
                got_max, np.maximum.reduce(a, axis=axis, keepdims=keepdims))

    def test_reduce_returns_tuple(self):
        a = make_array((5,), seed=4)
        result = mm.reduce(a)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reduce_single_element(self):
        a = np.array([7.0])
        got_min, got_max = mm.reduce(a)
        assert got_min == 7.0 and got_max == 7.0

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_reduce_strided(self, shape):
        a = make_array(shape, seed=15)[::-1]
        for axis in reduce_axes(a.ndim):
            got_min, got_max = mm.reduce(a, axis=axis)
            np.testing.assert_array_equal(got_min, np.minimum.reduce(a, axis=axis))
            np.testing.assert_array_equal(got_max, np.maximum.reduce(a, axis=axis))

    @pytest.mark.parametrize("kind", list(SPECIALS))
    def test_reduce_specials(self, kind):
        vals = [1.0, -2.0, 3.5, 0.0, -1.0] + SPECIALS[kind]
        a = np.array(vals, dtype=np.float64)
        got_min, got_max = mm.reduce(a)
        np.testing.assert_array_equal(got_min, np.minimum.reduce(a))
        np.testing.assert_array_equal(got_max, np.maximum.reduce(a))

    @pytest.mark.parametrize("initial_kind", ["small", "large"])
    def test_reduce_initial_scalar(self, initial_kind):
        a = make_array((3, 4), seed=5)
        initial = float(a.min()) - 1.0 if initial_kind == "small" \
            else float(a.max()) + 1.0
        for axis in reduce_axes(a.ndim):
            got_min, got_max = mm.reduce(a, axis=axis, initial=initial)
            np.testing.assert_array_equal(
                got_min, np.minimum.reduce(a, axis=axis, initial=initial))
            np.testing.assert_array_equal(
                got_max, np.maximum.reduce(a, axis=axis, initial=initial))

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

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_reduce_out_tuple(self, shape, keepdims):
        a = make_array(shape, seed=8)
        for axis in reduce_axes(a.ndim):
            ref_min = np.minimum.reduce(a, axis=axis, keepdims=keepdims)
            ref_max = np.maximum.reduce(a, axis=axis, keepdims=keepdims)
            omin = np.empty(np.shape(ref_min), np.float64)
            omax = np.empty(np.shape(ref_max), np.float64)
            got_min, got_max = mm.reduce(
                a, axis=axis, keepdims=keepdims, out=(omin, omax))
            assert got_min is omin and got_max is omax
            np.testing.assert_array_equal(omin, ref_min)
            np.testing.assert_array_equal(omax, ref_max)

    def test_reduce_out_bare_array_raises(self):
        a = make_array((4,), seed=9)
        with pytest.raises(TypeError, match="must be a tuple of arrays"):
            mm.reduce(a, out=np.empty(()))

    def test_reduce_out_wrong_length_raises(self):
        a = make_array((4,), seed=10)
        with pytest.raises(ValueError, match="exactly one entry per ufunc output"):
            mm.reduce(a, out=(np.empty(()),))

    def test_reduce_out_and_initial(self):
        a = make_array((4,), seed=19)
        omin = np.empty(())
        omax = np.empty(())
        got_min, got_max = mm.reduce(a, initial=100.0, out=(omin, omax))
        assert got_min is omin and got_max is omax
        np.testing.assert_array_equal(omin, np.minimum.reduce(a, initial=100.0))
        np.testing.assert_array_equal(omax, np.maximum.reduce(a, initial=100.0))

    @pytest.mark.parametrize("shape", SHAPES, ids=str)
    def test_reduce_where(self, shape):
        a = make_array(shape, seed=11)
        mask = np.random.default_rng(12).integers(0, 2, size=shape).astype(bool)
        initial = float(a.max()) + 1.0
        for axis in reduce_axes(a.ndim):
            got_min, got_max = mm.reduce(a, axis=axis, where=mask, initial=initial)
            np.testing.assert_array_equal(
                got_min, np.minimum.reduce(a, axis=axis, where=mask, initial=initial))
            np.testing.assert_array_equal(
                got_max, np.maximum.reduce(a, axis=axis, where=mask, initial=initial))

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
    def test_reduce_empty_axis_survives(self, shape, keepdims):
        a = np.zeros(shape, np.float64)
        for axis in reduce_axes(a.ndim):
            if reduces_over_empty(shape, axis):
                continue
            got_min, got_max = mm.reduce(a, axis=axis, keepdims=keepdims)
            np.testing.assert_array_equal(
                got_min, np.minimum.reduce(a, axis=axis, keepdims=keepdims))
            np.testing.assert_array_equal(
                got_max, np.maximum.reduce(a, axis=axis, keepdims=keepdims))

    @pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=str)
    def test_reduce_empty_scalar_initial(self, shape):
        a = np.zeros(shape, np.float64)
        for axis in reduce_axes(a.ndim):
            got_min, got_max = mm.reduce(a, axis=axis, initial=0.0)
            np.testing.assert_array_equal(
                got_min, np.minimum.reduce(a, axis=axis, initial=0.0))
            np.testing.assert_array_equal(
                got_max, np.maximum.reduce(a, axis=axis, initial=0.0))

    @pytest.mark.parametrize("shape", EMPTY_SHAPES, ids=str)
    def test_reduce_empty_tuple_initial(self, shape):
        a = np.zeros(shape, np.float64)
        for axis in reduce_axes(a.ndim):
            got_min, got_max = mm.reduce(a, axis=axis, initial=(0.0, 1.0))
            np.testing.assert_array_equal(
                got_min, np.minimum.reduce(a, axis=axis, initial=0.0))
            np.testing.assert_array_equal(
                got_max, np.maximum.reduce(a, axis=axis, initial=1.0))

    def test_unsupported_dtype_raises(self):
        a = np.array(["a", "b", "c"])
        with pytest.raises(ValueError, match="could not convert string to float"):
            mm.reduce(a)

    def test_no_reduction_loop_raises(self):
        with pytest.raises(
                ValueError, match="resolved loop does not register a reduction loop"):
            np.divmod.reduce([1, 2, 3])

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

    def test_reduce_dtype_same(self):
        a = make_array((4,), seed=16)
        got_min, got_max = mm.reduce(a, dtype=np.float64)
        assert got_min.dtype == np.float64 and got_max.dtype == np.float64
        np.testing.assert_array_equal(got_min, np.minimum.reduce(a, dtype=np.float64))
        np.testing.assert_array_equal(got_max, np.maximum.reduce(a, dtype=np.float64))

    def test_reduce_dtype_forced_no_loop_raises(self):
        a = make_array((4,), seed=17)
        with pytest.raises(TypeError, match="did not contain a loop"):
            mm.reduce(a, dtype=np.int64)

    def test_reduce_dtype_mismatched_tuple_raises(self):
        a = make_array((4,), seed=18)
        with pytest.raises(ValueError, match="mismatch in size"):
            mm.reduce(a, dtype=(np.int32, np.int64))

    def test_accumulate_raises(self):
        a = make_array((4,), seed=20)
        with pytest.raises(
                ValueError, match="only supported for functions returning a single value"):
            mm.accumulate(a)

    def test_reduceat_raises(self):
        a = make_array((4,), seed=21)
        with pytest.raises(
                ValueError, match="only supported for functions returning a single value"):
            mm.reduceat(a, [0, 2])

    def test_at_raises(self):
        a = make_array((4,), seed=22)
        with pytest.raises(ValueError, match="single output"):
            mm.at(a, [0], a)

    def test_outer(self):
        a = make_array((3,), seed=23)
        b = make_array((4,), seed=24)
        got_min, got_max = mm.outer(a, b)
        np.testing.assert_array_equal(got_min, np.minimum.outer(a, b))
        np.testing.assert_array_equal(got_max, np.maximum.outer(a, b))
