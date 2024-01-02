import pytest

import operator
import numpy as np

from numpy.testing import assert_array_equal, assert_equal, assert_raises


COMPARISONS = [
    (operator.eq, np.equal, "=="),
    (operator.ne, np.not_equal, "!="),
    (operator.lt, np.less, "<"),
    (operator.le, np.less_equal, "<="),
    (operator.gt, np.greater, ">"),
    (operator.ge, np.greater_equal, ">="),
]

BIG_INT = np.iinfo(np.int64).max


@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
def test_mixed_string_comparison_ufuncs_fail(op, ufunc, sym):
    arr_string = np.array(["a", "b"], dtype="S")
    arr_unicode = np.array(["a", "c"], dtype="U")

    with pytest.raises(TypeError, match="did not contain a loop"):
        ufunc(arr_string, arr_unicode)

    with pytest.raises(TypeError, match="did not contain a loop"):
        ufunc(arr_unicode, arr_string)

@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
def test_mixed_string_comparisons_ufuncs_with_cast(op, ufunc, sym):
    arr_string = np.array(["a", "b"], dtype="S")
    arr_unicode = np.array(["a", "c"], dtype="U")

    # While there is no loop, manual casting is acceptable:
    res1 = ufunc(arr_string, arr_unicode, signature="UU->?", casting="unsafe")
    res2 = ufunc(arr_string, arr_unicode, signature="SS->?", casting="unsafe")

    expected = op(arr_string.astype('U'), arr_unicode)
    assert_array_equal(res1, expected)
    assert_array_equal(res2, expected)


@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
@pytest.mark.parametrize("dtypes", [
        ("S2", "S2"), ("S2", "S10"),
        ("<U1", "<U1"), ("<U1", ">U1"), (">U1", ">U1"),
        ("<U1", "<U10"), ("<U1", ">U10")])
@pytest.mark.parametrize("aligned", [True, False])
def test_string_comparisons(op, ufunc, sym, dtypes, aligned):
    # ensure native byte-order for the first view to stay within unicode range
    native_dt = np.dtype(dtypes[0]).newbyteorder("=")
    arr = np.arange(2**15).view(native_dt).astype(dtypes[0])
    if not aligned:
        # Make `arr` unaligned:
        new = np.zeros(arr.nbytes + 1, dtype=np.uint8)[1:].view(dtypes[0])
        new[...] = arr
        arr = new

    arr2 = arr.astype(dtypes[1], copy=True)
    np.random.shuffle(arr2)
    arr[0] = arr2[0]  # make sure one matches

    expected = [op(d1, d2) for d1, d2 in zip(arr.tolist(), arr2.tolist())]
    assert_array_equal(op(arr, arr2), expected)
    assert_array_equal(ufunc(arr, arr2), expected)
    assert_array_equal(
        np.char.compare_chararrays(arr, arr2, sym, False), expected
    )

    expected = [op(d2, d1) for d1, d2 in zip(arr.tolist(), arr2.tolist())]
    assert_array_equal(op(arr2, arr), expected)
    assert_array_equal(ufunc(arr2, arr), expected)
    assert_array_equal(
        np.char.compare_chararrays(arr2, arr, sym, False), expected
    )


@pytest.mark.parametrize(["op", "ufunc", "sym"], COMPARISONS)
@pytest.mark.parametrize("dtypes", [
        ("S2", "S2"), ("S2", "S10"), ("<U1", "<U1"), ("<U1", ">U10")])
def test_string_comparisons_empty(op, ufunc, sym, dtypes):
    arr = np.empty((1, 0, 1, 5), dtype=dtypes[0])
    arr2 = np.empty((100, 1, 0, 1), dtype=dtypes[1])

    expected = np.empty(np.broadcast_shapes(arr.shape, arr2.shape), dtype=bool)
    assert_array_equal(op(arr, arr2), expected)
    assert_array_equal(ufunc(arr, arr2), expected)
    assert_array_equal(
        np.char.compare_chararrays(arr, arr2, sym, False), expected
    )


@pytest.mark.parametrize("str_dt", ["S", "U"])
@pytest.mark.parametrize("float_dt", np.typecodes["AllFloat"])
def test_float_to_string_cast(str_dt, float_dt):
    float_dt = np.dtype(float_dt)
    fi = np.finfo(float_dt)
    arr = np.array([np.nan, np.inf, -np.inf, fi.max, fi.min], dtype=float_dt)
    expected = ["nan", "inf", "-inf", str(fi.max), str(fi.min)]
    if float_dt.kind == 'c':
        expected = [f"({r}+0j)" for r in expected]

    res = arr.astype(str_dt)
    assert_array_equal(res, np.array(expected, dtype=str_dt))


@pytest.mark.parametrize("dt", ["S", "U"])
class TestMethods:

    def test_add(self, dt):
        in1 = np.array(
            ["", "abc", "12345", "MixedCase", "12345 \0 ", "UPPER"],
            dtype=dt
        )
        in2 = in1.copy()
        out = np.array(
            ["", "abcabc", "1234512345", "MixedCaseMixedCase",
             "12345 \0 12345 \0 ", "UPPERUPPER"],
            dtype=dt
        )
        assert_array_equal(np.strings.add(in1, in2), out)

    def test_isalpha(self, dt):
        in_ = np.array(["", "a", "A", "\n", "abc", "aBc123", "abc\n"], dtype=dt)
        out = [False, True, True, False, True, False, False]
        assert_array_equal(np.strings.isalpha(in_), out)

    def test_isdigit(self, dt):
        in_ = np.array(["", "a", "0", "012345", "012345a"], dtype=dt)
        out = [False, False, True, True, False]
        assert_array_equal(np.strings.isdigit(in_), out)

    def test_isspace(self, dt):
        in_ = np.array(["", "a", "1", " ", "\t", "\r", "\n", " \t\r \n", " \t\r\na"],
                       dtype=dt)
        out = [False, False, False, True, True, True, True, True, False]
        assert_array_equal(np.strings.isspace(in_), out)

    def test_isdecimal(self, dt):
        if dt == "U":
            in_ = np.array(["", "a", "0",
                            "\u2460",  # CIRCLED DIGIT 1
                            "\xbc",  # VULGAR FRACTION ONE QUARTER
                            "\u0660", # ARABIC_INDIC DIGIT ZERO
                            "012345", "012345a"])
            out = [False, False, True, False, False, True, True, False]
            assert_array_equal(np.strings.isdecimal(in_), out)
        else:
            with assert_raises(TypeError):
                in_ = np.array(b"1")
                np.strings.isdecimal(in_)

    def test_isnumeric(self, dt):
        if dt == "U":
            in_ = np.array(["", "a", "0",
                            "\u2460",  # CIRCLED DIGIT 1
                            "\xbc",  # VULGAR FRACTION ONE QUARTER
                            "\u0660", # ARABIC_INDIC DIGIT ZERO
                            "012345", "012345a"])
            out = [False, False, True, True, True, True, True, False]
            assert_array_equal(np.strings.isnumeric(in_), out)
        else:
            with assert_raises(TypeError):
                in_ = np.array(b"1")
                np.strings.isnumeric(in_)

    def test_str_len(self, dt):
        in_ = np.array(["", "abc", "12345", "MixedCase", "12345 \0 ", "UPPER"], dtype=dt)
        assert_array_equal(np.strings.str_len(in_), [0, 3, 5, 9, 8, 5])

    def test_find(self, dt):
        in1 = np.array(
            ["abcdefghiabc", "abcdefghiabc", "abcdefghiabc", "abc", "abc",
             "abc", "rrarrrrrrrrra", "rrarrrrrrrrra", "rrarrrrrrrrra", "",
             "", "", "", "", ""],
            dtype=dt,
        )
        in2 = np.array(
            ["abc", "abc", "def", "", "", "", "a", "a", "a", "", "", "",
             "xx", "xx", "xx"],
            dtype=dt,
        )
        in3 = np.array(
            [0, 1, 4, 0, 3, 4, 0, 4, 4, 0, 1, BIG_INT, 0, 1, BIG_INT]
        )
        in4 = np.array(
            [BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, 6,
             BIG_INT, 1, 0, BIG_INT, 1, 0],
        )
        out = np.array(
            [0, 9, -1, 0, 3, -1, 2, 12, -1, 0, -1, -1, -1, -1, -1],
        )
        assert_array_equal(np.strings.find(in1, in2, in3, in4), out)

    def test_rfind(self, dt):
        in1 = np.array(
            ["abcdefghiabc", "abcdefghiabc", "abcdefghiabc", "abcdefghiabc", "abc",
             "abc", "abc", "rrarrrrrrrrra", "rrarrrrrrrrra", "rrarrrrrrrrra"],
            dtype=dt,
        )
        in2 = np.array(
            ["abc", "", "abcd", "abcz", "", "", "", "a", "a", "a"],
            dtype=dt,
        )
        in3 = np.array([0, 0, 0, 0, 0, 3, 4, 0, 4, 4])
        in4 = np.array(
            [BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT,
             BIG_INT, BIG_INT, 6]
        )
        out = np.array(
            [9, 12, 0, -1, 3, 3, -1, 12, 12, -1],
        )
        assert_array_equal(np.strings.rfind(in1, in2, in3, in4), out)

    def test_count(self, dt):
        in1 = np.array(
            ["aaa", "aaa", "aaa", "aaa", "aaa", "aaa", "aaa", "aaa", "aaa", "aaa",
             "aaa", "aaa", "aaa", "aaa", "aaa", "", "", "", "", "", ""],
            dtype=dt,
        )
        in2 = np.array(
            ["a", "b", "a", "a", "a", "a", "a", "a", "a", "a", "", "", "", "", "",
             "", "", "", "xx", "xx", "xx"],
            dtype=dt,
        )
        in3 = np.array([0, 0, 1, 10, -1, -10, 0, 0, 0, 0, 1, 3, 10, -1, -10, 0, 1,
                        BIG_INT, 0, 1, BIG_INT])
        in4 = np.array(
            [BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, 1, 10, -1, -10,
             BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, 1, 0, BIG_INT,
             1, 0]
        )
        out = np.array(
            [3, 0, 2, 0, 1, 3, 1, 3, 2, 0, 3, 1, 0, 2, 4, 1, 0, 0, 0, 0, 0],
        )
        assert_array_equal(np.strings.count(in1, in2, in3, in4), out)

    def test_startswith(self, dt):
        in1 = np.array(
            ["hello", "hello", "hello", "hello", "hello", "hello", "hello", "hello",
             "hello", "hello", "helloworld", "helloworld", "helloworld", "", "", "",
             "hello", "hello", "hello", "hello", "hello", "hello", "hello", "hello",
             "hello", "hello"],
            dtype=dt,
        )
        in2 = np.array(
            ["he", "hello", "hello world", "", "ello", "ello", "o", "o", "", "lo",
             "lowo", "lowo", "lowo", "", "", "", "he", "he", "hello", "hello world",
             "ello", "ello", "o", "o", "", "lo"],
            dtype=dt,
        )
        in3 = np.array([0, 0, 0, 0, 0, 1, 4, 5, 5, 6, 3, 3, 3, 0, 0, 1, 0, -53, 0,
                        -1, -5, -4, -2, -1, -3, -9])
        in4 = np.array([BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT,
                        BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, 7, 6, 1, 0, 0,
                        -1, -1, -1, -10, BIG_INT, BIG_INT, BIG_INT, BIG_INT, -3,
                        BIG_INT])
        out = np.array([True, True, False, True, False, True, True, False, True, False,
                        True, True, False, True, True, False, True, True,
                        False, False, False, True, False, True, True, False])
        assert_array_equal(np.strings.startswith(in1, in2, in3, in4), out)

    def test_endswith(self, dt):
        in1 = np.array(
            ["hello", "hello", "hello", "hello", "helloworld", "helloworld", "helloworld",
             "helloworld", "helloworld", "helloworld", "helloworld", "helloworld", "ab",
             "ab", "", "", "", "hello", "hello", "hello", "hello", "helloworld",
             "helloworld", "helloworld", "helloworld", "helloworld", "helloworld", "helloworld",
             "helloworld", "helloworld"],
            dtype=dt,
        )
        in2 = np.array(
            ["lo", "he", "", "hello world", "worl", "worl", "world", "lowo", "lowo", "lowo",
             "lowo", "lowo", "ab", "ab", "", "", "", "lo", "he", "", "hello world", "worl",
             "worl", "worl", "world", "lowo", "lowo", "lowo", "lowo", "lowo"],
            dtype=dt,
        )
        in3 = np.array([0, 0, 0, 0, 0, 3, 3, 1, 2, 3, 4, 3, 0, 0, 0, 0, 1, -2, -2,
                        -3, -10, -6, -5, -5, -7, -99, -8, -7, 3, -8])
        in4 = np.array([BIG_INT, BIG_INT, BIG_INT, BIG_INT, BIG_INT, 9, 12, 7, 7, 7,
                        7, 8,1, 0, 1, 0, 0, BIG_INT, BIG_INT, -3, -2, BIG_INT, -1, 9,
                        12, -3, -3, -3, -4, -2])
        out = np.array([True, False, True, False, False, True, True, True, True, True,
                        False, False, False, False, True, True, False, True, False, True,
                        False, False, True, True, True, True, True, True, False, False])
        assert_array_equal(np.strings.endswith(in1, in2, in3, in4), out)

    def test_lstrip(self, dt):
        in1 = np.array(["   hello   ", "hello", " \t\n\r\f\vabc \t\n\r\f\v"], dtype=dt)
        out = np.array(["hello   ", "hello", "abc \t\n\r\f\v"], dtype=dt)
        assert_array_equal(np.strings.lstrip(in1), out)

        in1 = np.array(["xyzzyhelloxyzzy", "hello", "xyxz", "xyxzx"], dtype=dt)
        in2 = np.array(["xyz", "xyz", "xyxz", "x"], dtype=dt)
        out = np.array(["helloxyzzy", "hello", "", "yxzx"], dtype=dt)
        assert_array_equal(np.strings.lstrip(in1, in2), out)

    def test_rstrip(self, dt):
        in1 = np.array(["   hello   ", "hello", " \t\n\r\f\vabc \t\n\r\f\v"], dtype=dt)
        out = np.array(["   hello", "hello", " \t\n\r\f\vabc"], dtype=dt)
        assert_array_equal(np.strings.rstrip(in1), out)

        in1 = np.array(["xyzzyhelloxyzzy", "hello", "xyxz", "xyxzx"], dtype=dt)
        in2 = np.array(["xyz", "xyz", "xyxz", "x"], dtype=dt)
        out = np.array(["xyzzyhello", "hello", "", "xyxz"], dtype=dt)
        assert_array_equal(np.strings.rstrip(in1, in2), out)

    def test_strip(self, dt):
        in1 = np.array(["   hello   ", "hello", " \t\n\r\f\vabc \t\n\r\f\v"], dtype=dt)
        out = np.array(["hello", "hello", "abc"], dtype=dt)
        assert_array_equal(np.strings.strip(in1), out)

        in1 = np.array(["xyzzyhelloxyzzy", "hello", "xyxz", "xyxz"], dtype=dt)
        in2 = np.array(["xyz", "xyz", "xyxz", "x"], dtype=dt)
        out = np.array(["hello", "hello", "", "yxz"], dtype=dt)
        assert_array_equal(np.strings.strip(in1, in2), out)
