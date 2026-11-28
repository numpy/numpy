"""
Test the scalar constructors, which also do type-coercion
"""
import pytest

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal


class TestFromString:
    def test_floating(self):
        # Ticket #640, floats from string
        fsingle = np.single('1.234')
        fdouble = np.double('1.234')
        flongdouble = np.longdouble('1.234')
        assert_almost_equal(fsingle, 1.234)
        assert_almost_equal(fdouble, 1.234)
        assert_almost_equal(flongdouble, 1.234)

    def test_floating_overflow(self):
        """ Strings containing an unrepresentable float overflow """
        fhalf = np.half('1e10000')
        assert_equal(fhalf, np.inf)
        fsingle = np.single('1e10000')
        assert_equal(fsingle, np.inf)
        fdouble = np.double('1e10000')
        assert_equal(fdouble, np.inf)
        flongdouble = pytest.warns(RuntimeWarning, np.longdouble, '1e10000')
        assert_equal(flongdouble, np.inf)

        fhalf = np.half('-1e10000')
        assert_equal(fhalf, -np.inf)
        fsingle = np.single('-1e10000')
        assert_equal(fsingle, -np.inf)
        fdouble = np.double('-1e10000')
        assert_equal(fdouble, -np.inf)
        flongdouble = pytest.warns(RuntimeWarning, np.longdouble, '-1e10000')
        assert_equal(flongdouble, -np.inf)


class TestExtraArgs:
    def test_superclass(self):
        # try both positional and keyword arguments
        s = np.str_(b'\\x61', encoding='unicode-escape')
        assert s == 'a'
        s = np.str_(b'\\x61', 'unicode-escape')
        assert s == 'a'

        # previously this would return '\\xx'
        with pytest.raises(UnicodeDecodeError):
            np.str_(b'\\xx', encoding='unicode-escape')
        with pytest.raises(UnicodeDecodeError):
            np.str_(b'\\xx', 'unicode-escape')

        # superclass fails, but numpy succeeds
        assert np.bytes_(-2) == b'-2'

    def test_datetime(self):
        dt = np.datetime64('2000-01', ('M', 2))
        assert np.datetime_data(dt) == ('M', 2)

        with pytest.raises(TypeError):
            np.datetime64('2000', garbage=True)

    def test_bool(self):
        with pytest.raises(TypeError):
            np.bool(False, garbage=True)

    def test_void(self):
        with pytest.raises(TypeError):
            np.void(b'test', garbage=True)


class TestFromInt:
    def test_intp(self):
        # Ticket #99
        assert_equal(1024, np.intp(1024))

    def test_uint64_from_negative(self):
        with pytest.raises(OverflowError):
            np.uint64(-2)


def _float32_pyint_cases():
    # (int, correctly rounded float32 value as an int).  Each int is within
    # one unit of a float32 midpoint, so rounding it to float64 first lands on
    # the midpoint and the float64 -> float32 tie then picks the wrong side.
    cases = []
    for exp in [54, 62, 64, 65, 100, 127]:
        lo = 2**exp - 2**(exp - 24)   # float32 neighbour below 2**exp
        mid = 2**exp - 2**(exp - 25)  # midpoint; ties to even goes to 2**exp
        cases += [(mid - 1, lo), (mid, 2**exp), (mid + 1, 2**exp)]
        lo = 2**exp + 2**(exp - 23)   # odd significand
        mid = lo + 2**(exp - 24)      # midpoint; ties to even goes up
        cases += [(mid - 1, lo), (mid, lo + 2**(exp - 23)),
                  (mid + 1, lo + 2**(exp - 23))]
    return cases


class TestFloat32FromPyInt:
    # gh-20687: Python ints used to be rounded via float64, so they could
    # convert to the wrong float32 neighbour.

    @pytest.mark.parametrize("value, expected", _float32_pyint_cases())
    @pytest.mark.parametrize("sign", [1, -1])
    def test_correctly_rounded(self, value, expected, sign):
        value, expected = sign * value, sign * expected
        assert int(np.float32(value)) == expected
        assert int(np.array(value, dtype=np.float32)) == expected
        assert int(np.array([value], dtype=np.float32)[0]) == expected
        arr = np.zeros(1, dtype=np.float32)
        arr[0] = value
        assert int(arr[0]) == expected
        obj = np.array([value], dtype=object)
        assert int(obj.astype(np.float32)[0]) == expected
        c = np.complex64(value)
        assert int(c.real) == expected and c.imag == 0
        assert int(np.array([value], dtype=np.complex64)[0].real) == expected

    @pytest.mark.parametrize("value", [2**54 - 2**29 - 1, 2**63 - 2**38 - 1,
                                       2**64 - 2**39 - 1, -(2**63)])
    def test_matches_integer_array_cast(self, value):
        dtype = np.uint64 if value >= 2**63 else np.int64
        from_array = np.array([value], dtype=dtype).astype(np.float32)[0]
        assert np.float32(value) == from_array

    def test_overflow(self):
        # Rounds to infinity, as before, with a warning
        with pytest.warns(RuntimeWarning, match="overflow"):
            assert np.float32(2**128) == np.inf
        with pytest.warns(RuntimeWarning, match="overflow"):
            assert np.float32(-(2**200)) == -np.inf
        # Too large for a float64, like float(value)
        with pytest.raises(OverflowError):
            np.float32(2**1024)

    def test_float32_max_boundary(self):
        fmax = int(np.finfo(np.float32).max)
        half_ulp = 2**(127 - 24)
        assert int(np.float32(fmax + half_ulp - 1)) == fmax
        with pytest.warns(RuntimeWarning, match="overflow"):
            assert np.float32(fmax + half_ulp) == np.inf


int_types = [np.byte, np.short, np.intc, np.long, np.longlong]
uint_types = [np.ubyte, np.ushort, np.uintc, np.ulong, np.ulonglong]
float_types = [np.half, np.single, np.double, np.longdouble]
cfloat_types = [np.csingle, np.cdouble, np.clongdouble]


class TestArrayFromScalar:
    """ gh-15467 and gh-19125 """

    def _do_test(self, t1, t2, arg=2):
        if arg is None:
            x = t1()
        elif isinstance(arg, tuple):
            x = t1(*arg)
        else:
            x = t1(arg)
        arr = np.array(x, dtype=t2)
        # type should be preserved exactly
        if t2 is None:
            assert arr.dtype.type is t1
        else:
            assert arr.dtype.type is t2

    @pytest.mark.parametrize('t1', int_types + uint_types)
    @pytest.mark.parametrize('t2', int_types + uint_types + [None])
    def test_integers(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize('t1', float_types)
    @pytest.mark.parametrize('t2', float_types + [None])
    def test_reals(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize('t1', cfloat_types)
    @pytest.mark.parametrize('t2', cfloat_types + [None])
    @pytest.mark.parametrize('arg', [2, 1 + 3j, (1, 2), None])
    def test_complex(self, t1, t2, arg):
        self._do_test(t1, t2, arg)

    @pytest.mark.parametrize('t', cfloat_types)
    def test_complex_errors(self, t):
        with pytest.raises(TypeError):
            t(1j, 1j)
        with pytest.raises(TypeError):
            t(1, None)
        with pytest.raises(TypeError):
            t(None, 1)
        with pytest.raises(TypeError):
            t(np.array([1, 2]), 1)

    @pytest.mark.parametrize('t', [np.csingle, np.cdouble])
    def test_complex_two_arg_numpy_complex(self, t):
        assert t(np.complex64(1 + 2j), 1) == 1 + 3j

    @pytest.mark.parametrize('value', [
        np.array(1 + 2j), np.complex64(1 + 2j),
        np.timedelta64(2, "D"), np.datetime64("2020-01-01"),
        np.array([1, 2]),
    ])
    def test_clongdouble_two_arg_requires_real_components(self, value):
        with pytest.raises(TypeError):
            np.clongdouble(value, 1)
        with pytest.raises(TypeError):
            np.clongdouble(1, value)

    @pytest.mark.parametrize('value', [
        np.array(True), np.array(1, dtype=np.int64),
        np.array(np.longdouble(1)),
    ])
    def test_clongdouble_two_arg_0d_real_array(self, value):
        result = np.clongdouble(value, value)
        assert result.real == value[()]
        assert result.imag == value[()]

    def test_clongdouble_two_arg_python_bool(self):
        assert np.clongdouble(True, 2) == 1 + 2j
        assert np.clongdouble(2, False) == 2

    def test_clongdouble_constructor_preserves_precision(self):
        real = np.nextafter(np.longdouble(1), np.longdouble(2))
        imag = np.nextafter(np.longdouble(2), np.longdouble(3))

        value = np.clongdouble(real)
        assert value.real == real
        assert value.imag == 0

        value = np.clongdouble(real, imag)

        assert value.real == real
        assert value.imag == imag
        if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
            assert value.real != np.longdouble(float(real))
            assert value.imag != np.longdouble(float(imag))

    @pytest.mark.skipif(np.finfo(np.longdouble).nmant < 63,
                        reason="needs a >= 64-bit longdouble mantissa")
    def test_clongdouble_constructor_int_precision(self):
        value = np.clongdouble(2**63 + 1, 2**63 + 3)
        assert int(value.real) == 2**63 + 1
        assert int(value.imag) == 2**63 + 3

@pytest.mark.parametrize("length",
        [5, np.int8(5), np.array(5, dtype=np.uint16)])
def test_void_via_length(length):
    res = np.void(length)
    assert type(res) is np.void
    assert res.item() == b"\0" * 5
    assert res.dtype == "V5"

@pytest.mark.parametrize("bytes_",
        [b"spam", np.array(567.)])
def test_void_from_byteslike(bytes_):
    res = np.void(bytes_)
    expected = bytes(bytes_)
    assert type(res) is np.void
    assert res.item() == expected

    # Passing dtype can extend it (this is how filling works)
    res = np.void(bytes_, dtype="V100")
    assert type(res) is np.void
    assert res.item()[:len(expected)] == expected
    assert res.item()[len(expected):] == b"\0" * (res.nbytes - len(expected))
    # As well as shorten:
    res = np.void(bytes_, dtype="V4")
    assert type(res) is np.void
    assert res.item() == expected[:4]

def test_void_arraylike_trumps_byteslike():
    # The memoryview is converted as an array-like of shape (18,)
    # rather than a single bytes-like of that length.
    m = memoryview(b"just one mintleaf?")
    res = np.void(m)
    assert type(res) is np.ndarray
    assert res.dtype == "V1"
    assert res.shape == (18,)

def test_void_dtype_arg():
    # Basic test for the dtype argument (positional and keyword)
    res = np.void((1, 2), dtype="i,i")
    assert res.item() == (1, 2)
    res = np.void((2, 3), "i,i")
    assert res.item() == (2, 3)

@pytest.mark.parametrize("data",
        [5, np.int8(5), np.array(5, dtype=np.uint16)])
def test_void_from_integer_with_dtype(data):
    # The "length" meaning is ignored, rather data is used:
    res = np.void(data, dtype="i,i")
    assert type(res) is np.void
    assert res.dtype == "i,i"
    assert res["f0"] == 5 and res["f1"] == 5

def test_void_from_structure():
    dtype = np.dtype([('s', [('f', 'f8'), ('u', 'U1')]), ('i', 'i2')])
    data = np.array(((1., 'a'), 2), dtype=dtype)
    res = np.void(data[()], dtype=dtype)
    assert type(res) is np.void
    assert res.dtype == dtype
    assert res == data[()]

def test_void_bad_dtype():
    with pytest.raises(TypeError,
            match="void: descr must be a `void.*int64"):
        np.void(4, dtype="i8")

    # Subarray dtype (with shape `(4,)` is rejected):
    with pytest.raises(TypeError,
            match=r"void: descr must be a `void.*\(4,\)"):
        np.void(4, dtype="4i")
