"""Tests for the bytes-specific behavior of ByteStringDType.

Coverage shared with StringDType lives in test_stringdtype.py
(TestVariableWidthShared).
"""

import copy
import pickle

import pytest

import numpy as np
from numpy._core.tests.test_stringdtype import INVALID_UTF8
from numpy.dtypes import ByteStringDType, StringDType
from numpy.testing import assert_array_equal


def R(*values):
    return np.array(list(values), dtype=ByteStringDType())


NUL_AND_HIGH_BYTE_VALUES = [
    b"hello world",
    b"",
    b"x\x00",            # trailing NUL
    b"a\x00b",           # embedded NUL
    b"\x00\x00",         # all NULs
    b"\xff\xfe\x80",     # invalid UTF-8 lead/continuation bytes
    b"caf\xc3\xa9",      # valid UTF-8 (must still be treated as raw bytes)
    b"  spaced  ",
    b"\x00\xff" * 12,           # arena-length (>15 bytes), NULs + high bytes
    b"long " * 5 + b"tail\x00",  # arena-length, trailing NUL
]


@pytest.fixture
def dtype():
    return ByteStringDType()


class TestConstruction:
    def test_default(self, dtype):
        assert not hasattr(dtype, "coerce")
        with pytest.raises(AttributeError):
            dtype.na_object

    def test_no_coerce_kwarg(self):
        with pytest.raises(TypeError):
            ByteStringDType(coerce=False)
        with pytest.raises(TypeError):
            ByteStringDType(coerce=True)

    @pytest.mark.parametrize("kwargs,flags", [
        ({}, (False, False, False)),
        ({"na_object": None}, (True, False, False)),
        ({"na_object": np.nan}, (True, True, False)),
        ({"na_object": np.float32("nan")}, (True, True, False)),
        ({"na_object": b"\x00"}, (True, False, True)),
    ])
    def test_cached_na_flags(self, kwargs, flags):
        # numpy/numpy#32693 consumers use these cached classifications.
        dtype = ByteStringDType(**kwargs)
        names = ("_has_na", "_has_nan_na", "_has_string_na")
        assert tuple(getattr(dtype, name) for name in names) == flags
        for name in names:
            with pytest.raises(AttributeError):
                setattr(dtype, name, False)

    @pytest.mark.parametrize("na", [b"", b"\x00", b"NA", np.nan, None])
    def test_na_object(self, na):
        dt = ByteStringDType(na_object=na)
        if na is np.nan:
            assert dt.na_object is np.nan
        else:
            assert dt.na_object == na
        assert dt != ByteStringDType()
        assert dt == ByteStringDType(na_object=na)
        assert repr(dt) == f"ByteStringDType(na_object={na!r})"

    def test_from_char_and_letter(self, dtype):
        assert np.dtype("R") == dtype
        assert dtype.char == "R"
        assert dtype.kind == "R"
        assert np.dtype("R").name == f"ByteStringDType{dtype.itemsize * 8}"

    def test_pickle_roundtrip(self, dtype):
        for dt in [dtype, ByteStringDType(na_object=b"\x00"),
                   ByteStringDType(na_object=None)]:
            assert pickle.loads(pickle.dumps(dt)) == dt

    def test_bytes_inference_unchanged(self):
        # registering ByteStringDType must not change default inference
        assert np.array([b"x"]).dtype == np.dtype("S1")
        assert np.array(b"x").dtype == np.dtype("S1")


class TestStrictBytesInput:
    @pytest.mark.parametrize("value", [
        "text",
        1,
        1.5,
        None,
        object(),
    ])
    def test_rejected_with_typeerror(self, dtype, value):
        with pytest.raises(TypeError,
                           match="only allows bytes data"):
            np.array([value], dtype=dtype)
        arr = np.empty(1, dtype=dtype)
        with pytest.raises(TypeError, match="only allows bytes data"):
            arr[0] = value

    @pytest.mark.parametrize("value", [
        bytearray(b"buffer"),
        memoryview(b"buffer"),
    ])
    def test_buffer_protocol_rejected(self, dtype, value):
        # setitem rejects buffer-protocol objects with the clear message;
        # array coercion reads them as uint8 buffers first, so np.array
        # fails on the (unregistered) uint8 cast instead
        arr = np.empty(1, dtype=dtype)
        with pytest.raises(TypeError, match="only allows bytes data"):
            arr[0] = value
        with pytest.raises(TypeError):
            np.array([value], dtype=dtype)

    def test_str_rejection_mentions_encode(self, dtype):
        with pytest.raises(TypeError, match="str.encode"):
            np.array(["x"], dtype=dtype)
        with pytest.raises(TypeError) as exc_info:
            np.array([1], dtype=dtype)
        assert "str.encode" not in str(exc_info.value)

    def test_bytes_and_np_bytes_accepted(self, dtype):
        arr = np.array([b"x", np.bytes_(b"y")], dtype=dtype)
        assert arr.tolist() == [b"x", b"y"]

    def test_bytes_subclasses_accepted(self, dtype):
        # matches the fixed-width S dtype's input domain
        class MyBytes(bytes):
            pass
        arr = np.empty(2, dtype=dtype)
        arr[0] = MyBytes(b"q\x00")
        arr[1] = np.bytes_(b"a\x00b")
        assert arr.tolist() == [b"q\x00", b"a\x00b"]
        assert type(arr[0]) is np.vbytes
        arr2 = np.array([MyBytes(b"x\x00")], dtype=dtype)
        assert arr2.tolist() == [b"x\x00"]


class TestSetGet:
    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    def test_roundtrip(self, dtype, value):
        arr = np.array([value], dtype=dtype)
        assert arr[0] == value
        arr2 = np.empty(1, dtype=dtype)
        arr2[0] = value
        assert arr2[0] == value

    def test_np_bytes_trailing_nul_preserved(self, dtype):
        # np.bytes_ must be a known scalar type: discovered through its
        # fixed-width 'S' descriptor instead, the S->R cast strips the NULs
        value = np.bytes_(b"q\x00")
        arr = np.empty(1, dtype=dtype)
        arr[0] = value
        assert arr[0] == b"q\x00"
        arr[0:1] = [value]
        assert arr[0] == b"q\x00"
        assert np.array([value], dtype=dtype)[0] == b"q\x00"

    def test_zeros_is_empty_bytes(self, dtype):
        assert np.zeros(3, dtype=dtype).tolist() == [b""] * 3

    @pytest.mark.parametrize("value", [b"x\x00", b"\x00", b"x" * 30 + b"\x00"])
    @pytest.mark.parametrize("scalar_type", [bytes, np.vbytes])
    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.filterwarnings("error")
    def test_append_preserves_scalar_bytes(self, value, scalar_type, reverse):
        # numpy/numpy#32642 must retain the original scalar until the
        # ByteStringDType descriptor is resolved, without an S intermediate.
        dtype = ByteStringDType(na_object=None)
        arr = np.array([[b"a", value]], dtype=dtype)
        scalar = scalar_type(value)
        args = (scalar, arr) if reverse else (arr, scalar)
        expected = [value, b"a", value] if reverse else [b"a", value, value]
        result = np.append(*args)
        assert result.dtype == dtype
        assert result.shape == (3,)
        assert result.tolist() == expected

    @pytest.mark.parametrize("other_dtype", ["R", "S2"])
    def test_append_explicit_axis(self, dtype, other_dtype):
        arr = np.array([[b"x\x00"]], dtype=dtype)
        other = np.array([[b"\xff\x00"]], dtype=other_dtype)
        result = np.append(arr, other, axis=0)
        assert result.dtype == dtype
        # Fixed-width padding is not data, even when appending to R.
        expected = b"\xff\x00" if other_dtype == "R" else b"\xff"
        assert result.tolist() == [[b"x\x00"], [expected]]


class TestSortingAndSelection:
    def test_sort_high_bytes(self, dtype):
        # bytewise, not codepoint, ordering
        arr = np.array([b"\xff", b"a", b"\x80"], dtype=dtype)
        assert np.sort(arr).tolist() == [b"a", b"\x80", b"\xff"]

    @pytest.mark.parametrize("na", [b"", b"x", b"\x00"])
    def test_null_behaves_like_bytes_sentinel(self, na):
        # a bytes na_object takes the string-NA path: a null is truthy
        # exactly when the sentinel is a nonempty bytes
        arr = np.array([na, b"y"], dtype=ByteStringDType(na_object=na))
        assert arr.astype(bool).tolist() == [bool(na), True]
        assert arr.nonzero()[0].tolist() == ([0, 1] if na else [1])
        assert np.sort(arr[::-1]).tolist() == sorted([na, b"y"])


class TestCasts:
    def test_self_cast(self, dtype):
        arr = np.array([b"x" * 30, b"a\x00b"], dtype=dtype)
        assert arr.astype(dtype).tolist() == arr.tolist()
        assert arr.astype(ByteStringDType(na_object=b"")).tolist() == \
            arr.tolist()

    def test_self_cast_null_to_no_na(self, dtype):
        # the null becomes the NA's bytes, not its repr (b"b'NAA'")
        arr = np.array([b"yo", b"NAA"], dtype=ByteStringDType(na_object=b"NAA"))
        assert arr.astype(dtype, casting="unsafe").tolist() == [b"yo", b"NAA"]
        sarr = np.array(["yo", "NAA"], dtype=StringDType(na_object="NAA"))
        assert sarr.astype(StringDType(), casting="unsafe").tolist() == \
            ["yo", "NAA"]

    def test_fixed_width_roundtrip_high_bytes(self, dtype):
        # no ASCII gate in either direction, unlike StringDType
        arr = np.array([b"ab", b"\xff\xfe", b""], dtype=dtype)
        fixed = arr.astype("S4")
        assert fixed.tolist() == [b"ab", b"\xff\xfe", b""]
        assert fixed.astype(dtype).tolist() == [b"ab", b"\xff\xfe", b""]

    @pytest.mark.parametrize("bad", INVALID_UTF8)
    def test_fixed_width_source_skips_utf8_validation(self, dtype, bad):
        # the same input is rejected by the S -> StringDType cast
        arr = np.array([bad], dtype=f"S{len(bad)}")
        assert arr.astype(dtype).tolist() == [bad]

    def test_fixed_width_source_strips_trailing_nuls(self, dtype):
        # 'S' cannot represent trailing NULs, so they are already gone in
        # the source of the S -> ByteStringDType cast
        fixed = np.array([b"x\x00"], dtype="S4")
        assert fixed.astype(dtype).tolist() == [b"x"]

    def test_fixed_width_source_is_safe_cast(self, dtype):
        assert np.can_cast("S4", dtype, casting="safe")
        assert not np.can_cast(dtype, "S4", casting="safe")
        assert not np.can_cast("V4", dtype, casting="safe")

    def test_void_roundtrip_preserves_nuls(self, dtype):
        # void -> ByteStringDType is length-explicit, with no UTF-8 validation
        arr = np.array([b"ab", b"\xff", b""], dtype=dtype)
        v = arr.astype("V4")
        assert v.astype(dtype).tolist() == [
            b"ab\x00\x00", b"\xff\x00\x00\x00", b"\x00\x00\x00\x00"]

    def test_bool_casts(self, dtype):
        arr = np.array([b"x", b"", b"\x00"], dtype=dtype)
        assert arr.astype(bool).tolist() == [True, False, True]
        assert np.array([True, False]).astype(dtype).tolist() == \
            [b"True", b"False"]

    def test_object_roundtrip(self, dtype):
        values = [b"ab", b"\xff", b"a\x00b"]
        obj = np.array(values, dtype=object)
        arr = obj.astype(dtype)
        assert arr.tolist() == values
        assert arr.astype(object).tolist() == values

    @pytest.mark.parametrize("other", [
        "int64", "uint64", "float64", "complex128", "datetime64[s]",
        "timedelta64[s]", "U4", StringDType()])
    def test_no_cast_to_or_from(self, dtype, other):
        # text and numeric conversions are deliberately unregistered;
        # text goes through the explicit encode/decode ufuncs
        arr = np.array([b"1"], dtype=dtype)
        with pytest.raises(TypeError):
            arr.astype(other)
        if isinstance(other, str):
            other_arr = np.array(["1"], dtype=other) \
                if other == "U4" else np.zeros(1, dtype=other)
        else:
            other_arr = np.array(["1"], dtype=other)
        with pytest.raises(TypeError):
            other_arr.astype(dtype)

    def test_structured_field_rejected(self, dtype):
        with pytest.raises(TypeError, match="not currently supported"):
            np.dtype([("field", dtype)])
        with pytest.raises(TypeError, match="not currently supported"):
            np.dtype([("field", dtype, 2)])
        with pytest.raises(TypeError, match="not currently supported"):
            np.dtype({"names": ["a"], "formats": [dtype]})
        with pytest.raises(TypeError, match="not currently supported"):
            np.dtype("R,i4")

    def test_subarray_dtype_rejected(self, dtype):
        with pytest.raises(TypeError,
                           match="not currently supported within subarray"):
            np.dtype((dtype, 2))
        # (dtype, ()) is equivalent to the dtype itself and remains allowed
        assert np.dtype((dtype, ())) == dtype


class TestUnsizedFixedWidthCasts:
    """Converting to an unsized "S" or "V" dtype infers the width, counted
    in bytes, from the values in the array being converted."""

    @pytest.mark.parametrize(
        "convert",
        [pytest.param(lambda arr, req: arr.astype(req), id="astype"),
         pytest.param(lambda arr, req: np.array(arr, dtype=req),
                      id="np.array")])
    @pytest.mark.parametrize("kind", ["S", "V"])
    @pytest.mark.parametrize(
        "values,width",
        [
            ([b"this", b"is", b"an", b"array"], 5),
            ([b"a" * 100, b"", b"b"], 100),
            # embedded and trailing NULs count as data
            ([b"x\0", b"y\0\0z", b""], 4),
            ([b"\xff\xfe", b"\x00" * 3], 3),
            # empty arrays and all-empty entries produce width 1
            ([], 1),
            ([b"", b"", b""], 1),
        ],
    )
    def test_infer_width(self, convert, kind, values, width, dtype):
        arr = np.array(values, dtype=dtype)
        res = convert(arr, kind)
        assert res.dtype == np.dtype(f"{kind}{width}")
        assert_array_equal(res, np.array(values, dtype=f"{kind}{width}"))

    @pytest.mark.parametrize(
        "na,width",
        [
            (None, 4),           # missing entries count as the repr "None"
            (np.nan, 3),
            (b"", 1),
            (b"miss\0", 5),
        ],
    )
    def test_missing_values(self, na, width):
        dt = ByteStringDType(na_object=na)
        arr = np.array([b"ab", na], dtype=dt)
        assert arr.astype("S").dtype == np.dtype(f"S{max(width, 2)}")
        all_null = np.array([na, na], dtype=dt)
        assert all_null.astype("S").dtype == np.dtype(f"S{width}")

    def test_explicit_width_still_truncates(self, dtype):
        arr = np.array([b"abcdef"], dtype=dtype)
        assert_array_equal(arr.astype("S3"), np.array([b"abc"], dtype="S3"))
        assert_array_equal(arr.astype("V3"), np.array([b"abc"], dtype="V3"))

    def test_descriptor_only_resolution_still_fails(self, dtype):
        # functions that adapt descriptors without inspecting array values
        # cannot infer a width
        arr = np.array([b"abc"], dtype=dtype)
        with pytest.raises(TypeError, match="cast"):
            np.concatenate([arr, arr], dtype="S")

    @pytest.mark.parametrize("unicode_dtype", ["U", "U3"])
    def test_unicode_target_fails(self, unicode_dtype, dtype):
        # no R to U cast, sized or not; the width discovery must not run
        arr = np.array([b"\xff\xfe"], dtype=dtype)
        with pytest.raises(TypeError, match="cast"):
            arr.astype(unicode_dtype)


class TestNAObject:
    def test_scalar_unpickle_guard(self, dtype):
        arr = np.array([b"x"], dtype=dtype)
        # the list-pickle path stores full arrays
        import numpy._core.multiarray as mu
        with pytest.raises(TypeError, match="Cannot unpickle"):
            mu.scalar(dtype, b"x")


class TestScalar:
    def test_type_registration(self, dtype):
        assert ByteStringDType.type is np.vbytes
        assert issubclass(np.vbytes, bytes)
        assert issubclass(np.vbytes, np.generic)
        assert np.dtype(np.vbytes) == dtype
        assert np.dtype("vbytes") == dtype
        assert np.vbytes(b"x").dtype == dtype
        assert np.sctypeDict["vbytes"] is np.vbytes
        assert np.vbytes in np.ScalarType
        assert np.issubdtype(dtype, np.generic)
        assert np.issubdtype(dtype, np.vbytes)
        assert not np.issubdtype(dtype, np.bytes_)

    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    def test_bytes_api(self, value):
        x = np.vbytes(value)
        assert type(x) is np.vbytes
        assert x == value
        assert type(x == value) is bool
        assert bytes(x) == value
        assert len(x) == len(value)
        assert hash(x) == hash(value)
        assert x[:] == value
        assert x.item() == value
        assert type(x.item()) is bytes

    def test_constructor_arguments(self):
        assert np.vbytes() == b""
        assert np.vbytes("caf\xe9", "utf-8") == b"caf\xc3\xa9"
        with pytest.raises(TypeError):
            np.vbytes("text")

    def test_repr_str(self):
        x = np.vbytes(b"x\x00")
        assert repr(x) == "np.vbytes(b'x\\x00')"
        assert str(x) == "b'x\\x00'"
        with np.printoptions(legacy="1.25"):
            assert repr(x) == "b'x\\x00'"

    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    def test_element_access(self, dtype, value):
        arr = np.array([value], dtype=dtype)
        for x in (arr[0], arr.flat[0], next(iter(arr))):
            assert type(x) is np.vbytes
            assert x == value
        assert type(arr.tolist()[0]) is bytes
        assert arr.tolist() == [value]
        assert arr.item() == value
        assert type(arr.item()) is bytes
        assert np.array(value, dtype=dtype)[()] == value
        assert type(np.array(value, dtype=dtype)[()]) is np.vbytes

    def test_inference(self, dtype):
        arr = np.array([np.vbytes(b"x\x00")])
        assert arr.dtype == dtype
        assert arr[0] == b"x\x00"
        assert np.array(np.vbytes(b"a\x00"))[()] == b"a\x00"
        assert np.array([b"x"]).dtype == np.dtype("S1")
        assert np.full(2, np.vbytes(b"x\x00")).tolist() == [b"x\x00"] * 2

    def test_mixed_inference(self, dtype):
        mixed = np.array([np.vbytes(b"x"), b"y"])
        assert mixed.dtype == dtype
        assert mixed.tolist() == [b"x", b"y"]

    def test_setitem(self, dtype):
        arr = np.empty(2, dtype=dtype)
        arr[0] = np.vbytes(b"q\x00")
        arr[1:] = [np.vbytes(b"a\x00b")]
        assert arr.tolist() == [b"q\x00", b"a\x00b"]
        arr.fill(np.vbytes(b"z\x00"))
        assert arr.tolist() == [b"z\x00"] * 2

    def test_scalar_expressions_keep_bytes_semantics(self):
        x = np.vbytes(b"a")
        assert x + b"b" == b"ab"
        assert type(x + b"b") is bytes
        assert x.decode() == "a"

    def test_pickle_and_copy(self):
        x = np.vbytes(b"x\x00\xff")
        for y in (pickle.loads(pickle.dumps(x)), copy.copy(x), copy.deepcopy(x)):
            assert type(y) is np.vbytes
            assert y == x

    def test_null_returns_na_object(self):
        dt = ByteStringDType(na_object=None)
        arr = np.array([b"x", None], dtype=dt)
        assert type(arr[0]) is np.vbytes
        assert arr[1] is None
        assert list(arr) == [b"x", None]

    def test_no_implicit_text_conversion(self):
        with pytest.raises(TypeError):
            np.array([np.vbytes(b"x")], dtype=StringDType())

    def test_subclass(self, dtype):
        class MyV(np.vbytes):
            pass

        x = MyV(b"x\x00")
        assert isinstance(x, np.vbytes)
        arr = np.array([x])
        assert arr.dtype == dtype
        assert arr[0] == b"x\x00"
        assert type(arr[0]) is np.vbytes


class TestScalarCAPI:
    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    def test_cast_to_bytestring_keeps_output_arena(self, value):
        from numpy._core._multiarray_tests import cast_scalar_to_ctype

        result = cast_scalar_to_ctype(np.vbytes(value), ByteStringDType(), False)
        assert result.item() == value
        assert result.copy().item() == value
        # The direct API accepts a type number, not an owning descriptor.
        with pytest.raises(TypeError, match="requires an owning descriptor"):
            cast_scalar_to_ctype(np.vbytes(value), ByteStringDType(), True)

    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    def test_from_scalar_keeps_owned_descriptor(self, value):
        from numpy._core._multiarray_tests import array_from_scalar

        dtype = ByteStringDType()
        result = array_from_scalar(np.vbytes(value), dtype)
        assert result.dtype == dtype
        assert result.dtype is not dtype
        assert result.item() == value
        assert result.copy().item() == value

    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    @pytest.mark.parametrize("hashed", [False, True])
    @pytest.mark.parametrize("direct", [False, True])
    def test_cast_to_bool(self, value, hashed, direct):
        from numpy._core._multiarray_tests import cast_scalar_to_ctype

        scalar = np.vbytes(value)
        if hashed:
            hash(scalar)
        result = cast_scalar_to_ctype(scalar, np.dtype(bool), direct)
        assert result.item() is bool(value)

    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    def test_cast_to_fixed_bytes(self, value):
        from numpy._core._multiarray_tests import cast_scalar_to_ctype

        # The sized descriptor is only accepted by CastScalarToCtype.
        dtype = np.dtype(f"S{max(len(value), 1)}")
        result = cast_scalar_to_ctype(np.vbytes(value), dtype, False)
        assert result.tobytes() == value.ljust(dtype.itemsize, b"\0")

    @pytest.mark.parametrize("direct", [False, True])
    def test_unsupported_cast(self, direct):
        from numpy._core._multiarray_tests import cast_scalar_to_ctype

        with pytest.raises(TypeError):
            cast_scalar_to_ctype(np.vbytes(b"1"), np.dtype("i8"), direct)

    @pytest.mark.parametrize("value", NUL_AND_HIGH_BYTE_VALUES)
    def test_raw_storage_rejected(self, value):
        from numpy._core._multiarray_tests import scalar_as_ctype

        scalar = np.vbytes(value)
        with pytest.raises(TypeError, match="do not expose packed"):
            scalar_as_ctype(scalar)
        with pytest.raises(TypeError, match="do not expose packed"):
            scalar.__array_interface__
        with pytest.raises(TypeError, match="do not expose packed"):
            scalar.byteswap()
        # The ordinary Python bytes buffer remains valid.
        assert memoryview(scalar).tobytes() == value

    @pytest.mark.parametrize("direct", [False, True])
    @pytest.mark.parametrize("scalar", [np.int64(0), np.float64(1)])
    def test_legacy_scalar_cast_unchanged(self, direct, scalar):
        from numpy._core._multiarray_tests import cast_scalar_to_ctype, scalar_as_ctype

        result = cast_scalar_to_ctype(scalar, np.dtype(bool), direct)
        assert result.item() is bool(scalar)
        scalar_as_ctype(scalar)
