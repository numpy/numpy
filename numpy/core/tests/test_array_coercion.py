"""
Tests for array coercion, mainly through testing `np.array` results directly.
Note that other such tests exist e.g. in `test_api.py` and many corner-cases
are tested (sometimes indirectly) elsewhere.
"""

import pytest

from itertools import product

import numpy as np
from numpy.core._rational_tests import rational

from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_raises, assert_warns,
    HAS_REFCOUNT
)

all_scalars = set(np.typeDict.values())
all_scalars.add(rational)

def arraylikes():
    """
    Generator for functions converting an array into various array-likes.
    If full is True (default) includes array-likes not capable of handling
    all dtypes
    """
    # base array:
    def ndarray(a):
        return a

    yield ndarray

    # subclass:
    class MyArr(np.ndarray):
        pass

    def subclass(a):
        return a.view(MyArr)

    yield subclass

    # Array-interface
    class ArrayDunder:
        def __init__(self, a):
            self.a = a

        def __array__(self, dtype=None):
            return self.a

    yield ArrayDunder

    # memory-view
    yield memoryview

    # Array-interface
    class ArrayInterface:
        def __init__(self, a):
            self.a = a  # need to hold on to keep interface valid
            self.__array_interface__ = a.__array_interface__

    yield ArrayInterface

    # Array-Struct
    class ArrayStruct:
        def __init__(self, a):
            self.a = a  # need to hold on to keep struct valid
            self.__array_struct__ = a.__array_struct__

    yield ArrayStruct


def scalar_instances(times=True):
    for scalar_type in all_scalars:
        if isinstance(scalar_instances(), np.complexfloating):
            yield scalar_type(2, 3)**scalar_type(0.5)
        elif issubclass(scalar_type, np.flexible):
            yield scalar_type(b"string")
        elif scalar_type is np.timedelta64:
            if times:
                yield np.timedelta64(2)  # generic units
                yield np.timedelta64(23, "s")
        elif scalar_type is np.datetime64:
            if times:
                yield np.datetime64("NaT")  # generic units
                yield np.datetime64("2020-06-07 12:43", "ms")
        elif issubclass(scalar_type, np.number):
            # Make we have an irrational scalar which needs full precision
            yield scalar_type(2)**scalar_type(0.5)

    # Cannot create a structured void scalar directly:
    structured = np.array((1, 3), "i,i")[()]
    assert isinstance(structured, np.void)
    assert structured.dtype == np.dtype("i,i")
    yield structured


def is_parametric_dtype(dtype):
    """Returns True if the the dtype is a parametric legacy dtype (itemsize
    is 0, or a datetime without units)
    """
    if dtype.itemsize == 0:
        return True
    if issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        if dtype.name.endswith("64"):
            # Generic time units
            return True
    return False


class TestStringDiscovery:
    @pytest.mark.parametrize("obj",
            [object(), 1.2, 10**43, None, "string"])
    def test_basic_stringlength(self, obj):
        if not isinstance(obj, (str, int)):
            pytest.xfail(
                "The Single object (first assert) uses a different branch "
                "and thus gives a different result (either wrong or longer"
                "string than normally discovered).")

        length = len(str(obj))
        expected = np.dtype(f"S{length}")

        assert np.array(obj, dtype="S").dtype == expected
        assert np.array([obj], dtype="S").dtype == expected

        # A nested array is also discovered correctly
        arr = np.array(obj, dtype="O")
        assert np.array(arr, dtype="S").dtype == expected

    @pytest.mark.xfail(reason="Only single array unpacking is supported")
    @pytest.mark.parametrize("obj",
            [object(), 1.2, 10**43, None, "string"])
    def test_nested_arrays_stringlength(self, obj):
        length = len(str(obj))
        expected = np.dtype(f"S{length}")
        arr = np.array(obj, dtype="O")
        assert np.array([arr, arr], dtype="S").dtype == expected

    @pytest.mark.xfail(reason="Only single array unpacking is supported")
    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_unpack_first_level(self, arraylike):
        # We unpack exactly one level of array likes
        obj = np.array([None])
        obj[0] = np.array(1.2)
        # the length of the included item, not of the float dtype
        length = len(str(obj[0]))
        expected = np.dtype(f"S{length}")

        obj = arraylike(obj)
        # casting to string usually calls str(obj)
        arr = np.array([obj], dtype="S")
        assert arr.shape == (1, 1)
        assert arr.dtype == expected

class TestScalarDiscovery:
    def test_void_special_case(self):
        # Void dtypes with structures discover tuples as elements
        arr = np.array((1, 2, 3), dtype="i,i,i")
        assert arr.shape == ()
        arr = np.array([(1, 2, 3)], dtype="i,i,i")
        assert arr.shape == (1,)

    def test_char_special_case(self):
        arr = np.array("string", dtype="c")
        assert arr.shape == (6,)
        assert arr.dtype.char == "c"
        arr = np.array(["string"], dtype="c")
        assert arr.shape == (1, 6)
        assert arr.dtype.char == "c"

    def test_unknown_object(self):
        arr = np.array(object())
        assert arr.shape == ()
        assert arr.dtype == np.dtype("O")

    @pytest.mark.parametrize("scalar", scalar_instances())
    def test_scalar(self, scalar):
        arr = np.array(scalar)
        assert arr.shape == ()
        assert arr.dtype == scalar.dtype

        if type(scalar) is np.bytes_:
            pytest.xfail("Nested bytes use len(str(scalar)) currently.")

        arr = np.array([[scalar, scalar]])
        assert arr.shape == (1, 2)
        assert arr.dtype == scalar.dtype

    # Additionally to string this test also runs into a corner case
    # with datetime promotion (the difference is the promotion order).
    @pytest.mark.xfail(reason="Coercion to string is not symmetric")
    def test_scalar_promotion(self):
        for sc1, sc2 in product(scalar_instances(), scalar_instances()):
            # test all combinations:
            arr = np.array([sc1, sc2])
            assert arr.shape == (2,)
            try:
                dt1, dt2 = sc1.dtype, sc2.dtype
                expected_dtype = np.promote_types(dt1, dt2)
                assert arr.dtype == expected_dtype
            except TypeError as e:
                # Will currently always go to object dtype
                assert arr.dtype == np.dtype("O")

    @pytest.mark.parametrize("scalar", scalar_instances())
    def test_scalar_coercion(self, scalar):
        # This tests various scalar coercion paths, mainly for the numerical
        # types.  It includes some paths not directly related to `np.array`
        if isinstance(scalar, np.inexact):
            # Ensure we have a full-precision number if available
            scalar = type(scalar)((scalar * 2)**0.5)

        if is_parametric_dtype(scalar.dtype):
            # datetime with unit will be named "datetime64[unit]"
            pytest.xfail("0-D object array to a unit-less datetime cast fails")

        # Use casting from object:
        arr = np.array(scalar, dtype=object).astype(scalar.dtype)

        # Test various ways to create an array containing this scalar:
        arr1 = np.array(scalar).reshape(1)
        arr2 = np.array([scalar])
        arr3 = np.empty(1, dtype=scalar.dtype)
        arr3[0] = scalar
        arr4 = np.empty(1, dtype=scalar.dtype)
        arr4[:] = [scalar]
        # All of these methods should yield the same results
        assert_array_equal(arr, arr1)
        assert_array_equal(arr, arr2)
        assert_array_equal(arr, arr3)
        assert_array_equal(arr, arr4)

    @pytest.mark.filterwarnings("ignore::numpy.ComplexWarning")
    # After change, can enable times here, and below and it will work,
    # Right now times are too complex, so map out some details below.
    @pytest.mark.parametrize("cast_to", scalar_instances(times=False))
    def test_scalar_coercion_same_as_cast_and_assignment(self, cast_to):
        """
        Test that in most cases:
           * `np.array(scalar, dtype=dtype)`
           * `np.empty((), dtype=dtype)[()] = scalar`
           * `np.array(scalar).astype(dtype)`
        should behave the same.  The only exceptions are paramteric dtypes
        (mainly datetime/timedelta without unit) and void without fields.
        """
        dtype = cast_to.dtype  # use to parametrize only the target dtype

        for scalar in scalar_instances(times=False):
            if is_parametric_dtype(dtype) and dtype.type is type(scalar):
                # This skips datetime and timedelta when the target has no
                # unit. Because for them `np.array()` adapts the unit.
                continue
            if dtype.type == np.void:
               if scalar.dtype.fields is not None and dtype.fields is None:
                    # Here, coercion to "V6" works, but the cast fails.
                    # Since the types are identical, SETITEM takes care of
                    # this, but has different rules than the cast.
                    with assert_raises(TypeError):
                        np.array(scalar).astype(dtype)
                    with pytest.xfail("raises, unlike the second one"):
                        np.array(scalar, dtype=dtype)
                    np.array([scalar], dtype=dtype)
                    continue

            # The main test, we first try to use casting and if it succeeds
            # continue below testing that things are the same, otherwise
            # test that the alternative paths at least also fail.
            try:
                cast = np.array(scalar).astype(dtype)
            except (TypeError, ValueError, RuntimeError):
                # coercion should also raise (error type may change)
                with assert_raises(Exception):
                    np.array(scalar, dtype=dtype)
                # assignment should also raise
                res = np.zeros((), dtype=dtype)
                with assert_raises(Exception):
                    res[()] = scalar

                return

            # Non error path:
            arr = np.array(scalar, dtype=dtype)
            assert_array_equal(arr, cast)
            # assignment behaves the same
            ass = np.zeros((), dtype=dtype)
            ass[()] = scalar
            assert_array_equal(ass, cast)


class TestTimeScalars:
    @pytest.mark.parametrize("dtype", [np.int64, np.float32])
    @pytest.mark.parametrize("scalar",
            [np.timedelta64("NaT", "s"), np.timedelta64(123, "s"),
             np.datetime64("NaT", "generic"), np.datetime64(1, "D")])
    @pytest.mark.xfail(
            reason="This uses int(scalar) or float(scalar) to assign, which "
                   "fails.  However, casting currently does not fail.")
    def test_coercion_basic(self, dtype, scalar):
        arr = np.array(scalar, dtype=dtype)
        cast = np.array(scalar).astype(dtype)
        ass = np.ones((), dtype=dtype)
        ass[()] = scalar  # raises, as would np.array([scalar], dtype=dtype)

        assert_array_equal(arr, cast)
        assert_array_equal(cast, cast)

    @pytest.mark.parametrize("dtype", [np.int64, np.float32])
    @pytest.mark.parametrize("scalar",
             [np.timedelta64(123, "ns"), np.timedelta64(12, "generic")])
    def test_coercion_timedelta_convert_to_number(self, dtype, scalar):
        # Only "ns" and "generic" timedeltas can be converted to numbers
        # so these are slightly special.
        arr = np.array(scalar, dtype=dtype)
        cast = np.array(scalar).astype(dtype)
        ass = np.ones((), dtype=dtype)
        ass[()] = scalar  # raises, as would np.array([scalar], dtype=dtype)

        assert_array_equal(arr, cast)
        assert_array_equal(cast, cast)

    @pytest.mark.parametrize("scalar_type", [np.datetime64, np.timedelta64])
    @pytest.mark.parametrize(["val", "unit"],
            [(123, "s"), (123, "D")])
    @pytest.mark.xfail(reason="Error not raised for assignment")
    def test_coercion_assignment_times(self, scalar_type, val, unit):
        scalar = scalar_type(val, unit)

        # The error type is not ideal, fails because string is too short:
        with pytest.raises(RuntimeError):
            np.array(scalar, dtype="S6")
        with pytest.raises(RuntimeError):
            cast = np.array(scalar).astype("S6")
        ass = np.ones((), dtype="S6")
        with pytest.raises(RuntimeError):
            ass[()] = scalar


class TestNested:
    @pytest.mark.xfail(reason="No deprecation warning given.")
    def test_nested_simple(self):
        initial = [1.2]
        nested = initial
        for i in range(np.MAXDIMS - 1):
            nested = [nested]

        arr = np.array(nested, dtype="float64")
        assert arr.shape == (1,) * np.MAXDIMS
        with assert_raises(ValueError):
            np.array([nested], dtype="float64")

        # We discover object automatically at this time:
        with assert_warns(np.VisibleDeprecationWarning):
            arr = np.array([nested])
        assert arr.dtype == np.dtype("O")
        assert arr.shape == (1,) * np.MAXDIMS
        assert arr.item() is initial

    @pytest.mark.xfail(
            reason="For arrays and memoryview, this used to not complain "
                   "and assign to a too small array instead. For other "
                   "array-likes the error is different because fewer (only "
                   "MAXDIM-1) dimensions are found, failing the last test.")
    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_nested_arraylikes(self, arraylike):
        # We try storing an array like into an array, but the array-like
        # will have too many dimensions.  This means the shape discovery
        # decides that the array-like must be treated as an object (a special
        # case of ragged discovery).  The result will be an array with one
        # dimension less than the maximum dimensions, and the array being
        # assigned to it (which does work for object or if `float(arraylike)`
        # works).
        initial = arraylike(np.ones((1, 1)))
        #if not isinstance(initial, (np.ndarray, memoryview)):
        #    pytest.xfail(
        #        "When coercing to object, these cases currently discover "
        #        "fewer dimensions than ndarray failing the second part.")

        nested = initial
        for i in range(np.MAXDIMS - 1):
            nested = [nested]

        with assert_raises(ValueError):
            # It will refuse to assign the array into
            np.array(nested, dtype="float64")

        # If this is object, we end up assigning a (1, 1) array into (1,)
        # (due to running out of dimensions), this is currently supported but
        # a special case which is not ideal.
        arr = np.array(nested, dtype=object)
        assert arr.shape == (1,) * np.MAXDIMS
        assert arr.item() == np.array(initial).item()

    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_uneven_depth_ragged(self, arraylike):
        arr = np.arange(4).reshape((2, 2))
        arr = arraylike(arr)

        # Array is ragged in the second dimension already:
        out = np.array([arr, [arr]], dtype=object)
        assert out.shape == (2,)
        assert out[0] is arr
        assert type(out[1]) is list

        if not isinstance(arr, (np.ndarray, memoryview)):
            pytest.xfail(
                "does not raise ValueError below, because it discovers "
                "the dimension as (2,) and not (2, 2, 2)")

        # Array is ragged in the third dimension:
        with pytest.raises(ValueError):
            # This is a broadcast error during assignment, because
            # the array shape would be (2, 2, 2) but `arr[0, 0] = arr` fails.
            np.array([arr, [arr, arr]], dtype=object)

    def test_empty_sequence(self):
        arr = np.array([[], [1], [[1]]], dtype=object)
        assert arr.shape == (3,)

        # The empty sequence stops further dimension discovery, so the
        # result shape will be (0,) which leads to an error during:
        with pytest.raises(ValueError):
            np.array([[], np.empty((0, 1))], dtype=object)


class TestBadSequences:
    # These are tests for bad objects passed into `np.array`, in general
    # these should raise some error, although even returning undefined
    # behaviour is fine.  But they should not crash.

    def test_growing_list(self):
        # List to coerce, `mylist` will append to it during coercion
        obj = []
        class mylist(list):
            def __len__(self):
                obj.append([1, 2])
                return super().__len__()

        obj.append(mylist([1, 2]))

        with pytest.raises(ValueError):
            np.array(obj)

    @pytest.mark.skip(reason="segfaults currently")
    def test_shrinking_list(self):
        # List to coerce, `mylist` will delete to it during coercion
        obj = []
        class mylist(list):
            def __len__(self):
                obj.pop()
                return super().__len__()

        obj.append(mylist([1, 2]))
        obj.append([2, 3])
        with pytest.raises(ValueError):
            np.array(obj)

    @pytest.mark.skip(reason="segfaults currently")
    def test_mutated_list(self):
        # List to coerce, `mylist` will mutate the first element
        obj = []
        class mylist(list):
            def __len__(self):
                obj[0] = [2, 3]
                return super().__len__()

        obj.append(mylist([1, 2]))
        obj.append([2, 3])
        with pytest.raises(ValueError):
            np.array(obj)
