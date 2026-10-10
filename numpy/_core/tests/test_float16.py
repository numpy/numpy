import math
import os
import subprocess
import sys
import sysconfig

import pytest

import numpy as np
from numpy.testing import IS_EDITABLE, IS_WASM

if IS_EDITABLE or IS_WASM:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )

h = None

@pytest.fixture(scope="module", autouse=True)
def _build_float16_extension(tmpdir_factory):
    # Based in part on tests/test_limited_api
    # Double chekc that meson exists
    try:
        subprocess.check_output(["meson", "--version"], stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("No usable 'meson' found")

    # numpy/_core/tests/examples/float16
    srcdir = os.path.join(os.path.dirname(__file__), 'examples',  "float16")
    if not os.path.isdir(srcdir):
        pytest.skip(f"float16 Meson example directory not found: {srcdir}")

    # Temporary build directory for this test run
    build_dir = tmpdir_factory.mktemp("float16") / "build"
    os.makedirs(build_dir, exist_ok=True)

    # Ensure we use the correct Python interpreter even when `meson` is
    # installed in a different Python environment (see gh-24956)
    native_file = str(build_dir / 'interpreter-native-file.ini')
    with open(native_file, 'w') as f:
        f.write("[binaries]\n")
        f.write(f"python = '{sys.executable}'\n")
        f.write(f"python3 = '{sys.executable}'")

    # Configure and build
    if sysconfig.get_platform() == "win-arm64":
        pytest.skip("Meson unable to find MSVC linker on win-arm64")
    if sys.platform == "win32":
        subprocess.check_call(["meson", "setup",
                               "--werror",
                               "--buildtype=release",
                               "--vsenv", "--native-file", native_file,
                               str(srcdir)],
                              cwd=build_dir,
                              )
    else:
        subprocess.check_call(["meson", "setup", "--werror",
                               "--native-file", native_file, str(srcdir)],
                              cwd=build_dir
                              )
    try:
        subprocess.check_call(
            ["meson", "compile", "-vv"], cwd=build_dir)
    except subprocess.CalledProcessError as p:
        print(f"{p.stdout=}")
        print(f"{p.stderr=}")
        raise

    sys.path.insert(0, str(build_dir))
    import _float16_tests

    global h
    h = _float16_tests


#
# Helpers
#

# +0
FLOAT16_PZERO = np.uint16(0x0000)

# -0
FLOAT16_NZERO = np.uint16(0x8000)

# 1.0
FLOAT16_ONE = np.uint16(0x3C00)

# -1.0
FLOAT16_NEGONE = np.uint16(0xBC00)

# +inf
# 0 11111 0000000000
FLOAT16_PINF = np.uint16(0x7C00)

# -inf
# 1 11111 0000000000
FLOAT16_NINF = np.uint16(0xFC00)

# not a number
FLOAT16_NAN = np.uint16(0x7E00)

# largest finite float16
FLOAT16_MAX = np.uint16(0x7BFF)

# Smallest Positive Subnormal
# sign bit = 0; exponent bits = 0; fraction bits = 1
# 0 00000 0000000001
# i.e smallest number above 0
FLOAT16_SMALLSUB = np.uint16(0x0001)

# Smallest Positive Subnormal
# sign bit = 0; exponent bits = 0; fraction bits = 11111111111
# 0 00000 1111111111
# i.e biggest number before next subnormal
FLOAT16_LARGESUB = np.uint16(0x03FF)

# Smallest positive normal number
# 0 00001 0000000000
FLOAT16_SMALLNORM = np.uint16(0x0400)

def f16_from_bits(bits: np.uint16) -> np.float16:
    return np.array(bits, dtype=np.uint16).view(np.float16)

def bits_from_f16(x: np.float16) -> np.uint16:
    return np.array(x, dtype=np.float16).view(np.uint16)

def numpy_float16_from_bits(bits: np.uint16) -> np.float32:
    return f16_from_bits(bits).astype(np.float32)

def numpy_float16_to_bits_from_f32(x: np.float32) -> np.uint16:
    return x.astype(np.float16).view(np.uint16)

def numpy_float16_to_bits_from_f64(x: np.float64) -> np.uint16:
    return x.astype(np.float16).view(np.uint16)


def numpy_float16_nextafter(x_bits: np.uint16, y_bits: np.uint16) -> np.uint16:
    x = f16_from_bits(x_bits)
    y = f16_from_bits(y_bits)
    # Overflow is expected when stepping from max finite toward +inf.
    with np.errstate(over="ignore", invalid="ignore"):
        res = np.nextafter(x, y, dtype=np.float16)
    return bits_from_f16(res)


def assert_isnan(x):
    assert isinstance(x, (float, np.floating))
    assert np.isnan(x)

#
# Conversions
#

# Tests npy_float16_to_float(npy_half h),
# npy_float16bits_to_floatbits(h)
@pytest.mark.parametrize("bits", [
    FLOAT16_PZERO,
    FLOAT16_NZERO,
    FLOAT16_ONE,
    FLOAT16_NEGONE,
    FLOAT16_PINF,
    FLOAT16_NINF,
    FLOAT16_NAN,
    FLOAT16_MAX,
    FLOAT16_SMALLSUB,
    FLOAT16_LARGESUB,
    FLOAT16_SMALLNORM,
])
def test_float16_to_float(bits):
    got = h.float16_to_float(bits)
    expected = numpy_float16_from_bits(bits)
    if np.isnan(expected):
        assert_isnan(got)
    else:
        assert got == np.float32(expected)

# Tests npy_float16_to_double(npy_half h),
# npy_float16bits_to_doublebits(npy_uint16 h)
@pytest.mark.parametrize("bits", [
    FLOAT16_PZERO,
    FLOAT16_NZERO,
    FLOAT16_ONE,
    FLOAT16_NEGONE,
    FLOAT16_PINF,
    FLOAT16_NINF,
    FLOAT16_NAN,
    FLOAT16_MAX,
    FLOAT16_SMALLSUB,
    FLOAT16_LARGESUB,
    FLOAT16_SMALLNORM,
])
def test_float16_to_double(bits):
    got = h.float16_to_double(bits)
    expected = numpy_float16_from_bits(bits).astype(np.float64)
    if np.isnan(expected):
        assert_isnan(got)
    else:
        assert got == expected

# Tests npy_double_to_float16(double d),
# npy_doublebits_to_float16bits(npy_uint64 d),
# _npy_float16_header_overflow()
@pytest.mark.parametrize("doubles", [
    np.float64(0.0),
    np.float64(-0.0),
    np.float64(1.0),
    np.float64(-1.0),
    np.float64(0.5),
    np.float64(65504.0),
    np.float64(1e-7),
    np.float64(-1e-7),
    np.float64(np.inf),
    np.float64(-np.inf),
    np.float64(np.nan),
])
def test_double_to_float16(doubles):
    got = np.uint16(h.double_to_float16(doubles))
    expected = numpy_float16_to_bits_from_f64(doubles)

    if np.isnan(doubles):
        assert (got & FLOAT16_PINF) == FLOAT16_PINF
        assert (got & FLOAT16_LARGESUB) != FLOAT16_PZERO
    else:
        assert got == expected

# Tests npy_float_to_float16(float f),
# npy_floatbits_to_float16bits(npy_uint32 f),
# _npy_float16_header_overflow(),
# _npy_float16_header_underflow()
@pytest.mark.parametrize("floats", [
    np.float32(0.0),
    np.float32(-0.0),
    np.float32(1.0),
    np.float32(-1.0),
    np.float32(0.5),
    np.float32(65504.0),
    np.float32(1e-7),
    np.float32(-1e-7),
    np.float32(np.inf),
    np.float32(-np.inf),
    np.float32(np.nan),
])
def test_float_to_float16(floats):
    got = np.uint16(h.float_to_float16(floats))
    expected = numpy_float16_to_bits_from_f32(floats)

    if np.isnan(floats):
        assert (got & FLOAT16_PINF) == FLOAT16_PINF
        assert (got & FLOAT16_LARGESUB) != FLOAT16_PZERO
    else:
        assert got == expected

#
# Comparisons
#

# Tests npy_float16_eq(npy_half h1, npy_half h2),
# npy_float16_ne(npy_half h1, npy_half h2)
@pytest.mark.parametrize("a_bits, b_bits", [
    (FLOAT16_ONE, FLOAT16_ONE),
    (FLOAT16_PZERO, FLOAT16_NZERO),
    (FLOAT16_NZERO, FLOAT16_PZERO),
    (FLOAT16_ONE, FLOAT16_NEGONE),
    (FLOAT16_NEGONE, FLOAT16_ONE),
    (FLOAT16_ONE, FLOAT16_PZERO),
    (FLOAT16_PZERO, FLOAT16_ONE),
])
def test_float16_eq_and_ne(a_bits, b_bits):
    a = numpy_float16_from_bits(a_bits)
    b = numpy_float16_from_bits(b_bits)

    expected_eq = (a == b) or (a == 0.0 and b == 0.0 and
                               math.copysign(1.0, a) != math.copysign(1.0, b))
    expected_eq_numpy = np.array(a, dtype=np.float16) == np.array(b, dtype=np.float16)

    got_eq = bool(h.float16_eq(int(a_bits), int(b_bits)))
    got_ne = bool(h.float16_ne(int(a_bits), int(b_bits)))

    assert got_eq == bool(expected_eq_numpy)
    assert got_ne == (not got_eq)

# Tests npy_float16_eq(npy_half h1, npy_half h2),
# npy_float16_ne(npy_half h1, npy_half h2)
@pytest.mark.parametrize("a_bits, b_bits", [
    (FLOAT16_NAN, FLOAT16_ONE),
    (FLOAT16_ONE, FLOAT16_NAN),
    (FLOAT16_NAN, FLOAT16_NAN),
])
def test_float16_eq_and_ne_with_nan(a_bits, b_bits):
    assert not h.float16_eq(int(a_bits), int(b_bits))
    assert h.float16_ne(int(a_bits), int(b_bits))

# Tests npy_float16_le(npy_half h1, npy_half h2)
# npy_float16_lt(npy_half h1, npy_half h2),
# npy_float16_ge(npy_half h1, npy_half h2),
# npy_float16_gt(npy_half h1, npy_half h2)
@pytest.mark.parametrize("a_bits, b_bits", [
    (FLOAT16_NEGONE, FLOAT16_PZERO),
    (FLOAT16_PZERO, FLOAT16_ONE),
    (FLOAT16_NEGONE, FLOAT16_ONE),
    (FLOAT16_NINF, FLOAT16_MAX),
])
def test_float16_lt_le_gt_ge(a_bits, b_bits):
    a = numpy_float16_from_bits(a_bits)
    b = numpy_float16_from_bits(b_bits)

    expected_lt = a < b
    expected_le = a <= b
    expected_gt = a > b
    expected_ge = a >= b

    got_lt = bool(h.float16_lt(int(a_bits), int(b_bits)))
    got_le = bool(h.float16_le(int(a_bits), int(b_bits)))
    got_gt = bool(h.float16_gt(int(a_bits), int(b_bits)))
    got_ge = bool(h.float16_ge(int(a_bits), int(b_bits)))

    assert got_lt == expected_lt
    assert got_le == expected_le
    assert got_gt == expected_gt
    assert got_ge == expected_ge

#
# No nan Comparison Variants
#

# Tests _npy_float16_lt_nonan(npy_half h1, npy_half h2),
# _npy_float16_le_nonan(npy_half h1, npy_half h2)
@pytest.mark.parametrize("a_bits, b_bits", [
    (FLOAT16_PZERO, FLOAT16_NZERO),
    (FLOAT16_NZERO, FLOAT16_PZERO),
])
def test_float16_eq_nonan_zeros(a_bits, b_bits):
    assert not h.float16_lt_nonan(int(a_bits), int(b_bits))
    assert h.float16_le_nonan(int(a_bits), int(b_bits))

#
# Misc functions
#

# Tests npy_float16_iszero(npy_half h),
# npy_float16_isnan(npy_half h),
# npy_float16_isinf(npy_half h),
# npy_float16_isfinite(npy_half h),
# npy_float16_signbit(npy_half h)
@pytest.mark.parametrize("bits, iszero, isnan, isinf, isfinite, signbit", [
    (FLOAT16_PZERO, True, False, False, True, False),
    (FLOAT16_NZERO, True, False, False, True, True),
    (FLOAT16_ONE, False, False, False, True, False),
    (FLOAT16_NEGONE, False, False, False, True, True),
    (FLOAT16_PINF, False, False, True, False, False),
    (FLOAT16_NINF, False, False, True, False, True),
    (FLOAT16_NAN, False, True, False, False, False),
])
def test_predicates(bits, iszero, isnan, isinf, isfinite, signbit):
    assert bool(h.float16_iszero(int(bits))) == iszero
    assert bool(h.float16_isnan(int(bits))) == isnan
    assert bool(h.float16_isinf(int(bits))) == isinf
    assert bool(h.float16_isfinite(int(bits))) == isfinite
    assert bool(h.float16_signbit(int(bits))) == signbit

#
# Copysign
#

# Tests npy_float16_copysign(npy_half x, npy_half y)
@pytest.mark.parametrize("x_bits, y_bits", [
    (FLOAT16_ONE, FLOAT16_PZERO),
    (FLOAT16_ONE, FLOAT16_NZERO),
    (FLOAT16_NEGONE, FLOAT16_PZERO),
    (FLOAT16_NEGONE, FLOAT16_NZERO),
    (FLOAT16_ONE, FLOAT16_PINF),
    (FLOAT16_ONE, FLOAT16_NINF),
])
def test_float16_copysign(x_bits, y_bits):
    x = f16_from_bits(x_bits)
    y = f16_from_bits(y_bits)

    got_bits = np.uint16(h.float16_copysign(int(x_bits), int(y_bits)))
    got = f16_from_bits(got_bits)

    expected = np.copysign(x.astype(np.float32),
                           y.astype(np.float32)).astype(np.float16)
    assert got == expected

#
# Nextafter, and Divmod
#


# Tests npy_float16_nextafter(npy_half x, npy_half y)
@pytest.mark.parametrize("x_bits, y_bits", [
    (FLOAT16_PZERO, FLOAT16_ONE),
    (FLOAT16_PZERO, FLOAT16_NEGONE),
    (FLOAT16_ONE, FLOAT16_PINF),
    (FLOAT16_MAX, FLOAT16_PINF),
    (FLOAT16_MAX, FLOAT16_PZERO),
    (FLOAT16_NZERO, FLOAT16_ONE),
    (FLOAT16_NINF, FLOAT16_ONE),
])
def test_float16_nextafter_basic(x_bits, y_bits):
    got_bits = np.uint16(h.float16_nextafter(int(x_bits), int(y_bits)))
    expected_bits = numpy_float16_nextafter(x_bits, y_bits)

    x = f16_from_bits(x_bits)
    y = f16_from_bits(y_bits)
    if np.isnan(x) or np.isnan(y):
        assert np.isnan(f16_from_bits(got_bits))
    else:
        assert got_bits == expected_bits

# Tests npy_float16_nextafter(npy_half x, npy_half y)
@pytest.mark.parametrize("x_bits, y_bits", [
    (FLOAT16_NAN, FLOAT16_ONE),
    (FLOAT16_ONE, FLOAT16_NAN),
    (FLOAT16_NAN, FLOAT16_NAN),
])
def test_float16_nextafter_nan(x_bits, y_bits):
    got_bits = np.uint16(h.float16_nextafter(int(x_bits), int(y_bits)))
