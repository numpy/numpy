"""Tests for numpy.lib._scimath_impl (np.emath)."""
import numpy as np
from numpy import emath
from numpy.testing import assert_allclose


class TestSqrt:
    """Tests for emath.sqrt."""

    def test_positive_real_output(self):
        result = emath.sqrt(4.0)
        assert np.isreal(result)
        assert_allclose(result, 2.0)

    def test_negative_complex_output(self):
        result = emath.sqrt(-4.0)
        assert_allclose(result, 2j)

    def test_dtype_float32_to_complex64(self):
        arr = np.array([-4.0], dtype=np.float32)
        result = emath.sqrt(arr)
        assert result.dtype == np.complex64

    def test_dtype_float64_to_complex128(self):
        arr = np.array([-4.0], dtype=np.float64)
        result = emath.sqrt(arr)
        assert result.dtype == np.complex128

    def test_dtype_int16_to_complex64(self):
        arr = np.array([-4], dtype=np.int16)
        result = emath.sqrt(arr)
        assert result.dtype == np.complex64

    def test_dtype_int64_to_complex128(self):
        arr = np.array([-4], dtype=np.int64)
        result = emath.sqrt(arr)
        assert result.dtype == np.complex128

    def test_scalar_returns_scalar(self):
        result = emath.sqrt(-4)
        assert np.isscalar(result)

    def test_array_returns_array(self):
        result = emath.sqrt(np.array([-4, -9]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


class TestLog:
    """Tests for emath.log."""

    def test_positive_real_output(self):
        result = emath.log(np.e)
        assert np.isreal(result)
        assert_allclose(result, 1.0)

    def test_negative_complex_output(self):
        result = emath.log(-1.0)
        assert_allclose(result, np.pi * 1j)

    def test_dtype_float32_to_complex64(self):
        arr = np.array([-1.0], dtype=np.float32)
        result = emath.log(arr)
        assert result.dtype == np.complex64

    def test_dtype_float64_to_complex128(self):
        arr = np.array([-1.0], dtype=np.float64)
        result = emath.log(arr)
        assert result.dtype == np.complex128

    def test_scalar_returns_scalar(self):
        result = emath.log(-1)
        assert np.isscalar(result)

    def test_array_returns_array(self):
        result = emath.log(np.array([-1, -np.e]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


class TestLog2:
    """Tests for emath.log2."""

    def test_positive_real_output(self):
        result = emath.log2(8)
        assert np.isreal(result)
        assert_allclose(result, 3.0)

    def test_negative_complex_output(self):
        result = emath.log2(-8)
        assert np.iscomplexobj(result)
        assert_allclose(result.real, 3.0)

    def test_dtype_float32_to_complex64(self):
        arr = np.array([-8.0], dtype=np.float32)
        result = emath.log2(arr)
        assert result.dtype == np.complex64

    def test_dtype_float64_to_complex128(self):
        arr = np.array([-8.0], dtype=np.float64)
        result = emath.log2(arr)
        assert result.dtype == np.complex128

    def test_scalar_returns_scalar(self):
        result = emath.log2(-8)
        assert np.isscalar(result)

    def test_array_returns_array(self):
        result = emath.log2(np.array([-4, -8]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


class TestLog10:
    """Tests for emath.log10."""

    def test_positive_real_output(self):
        result = emath.log10(100)
        assert np.isreal(result)
        assert_allclose(result, 2.0)

    def test_negative_complex_output(self):
        result = emath.log10(-100)
        assert np.iscomplexobj(result)
        assert_allclose(result.real, 2.0)

    def test_dtype_float32_to_complex64(self):
        arr = np.array([-10.0], dtype=np.float32)
        result = emath.log10(arr)
        assert result.dtype == np.complex64

    def test_dtype_float64_to_complex128(self):
        arr = np.array([-10.0], dtype=np.float64)
        result = emath.log10(arr)
        assert result.dtype == np.complex128

    def test_scalar_returns_scalar(self):
        result = emath.log10(-10)
        assert np.isscalar(result)

    def test_array_returns_array(self):
        result = emath.log10(np.array([-10, -100]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


class TestLogn:
    """Tests for emath.logn."""

    def test_positive_real_output(self):
        result = emath.logn(2, 8)
        assert np.isreal(result)
        assert_allclose(result, 3.0)

    def test_negative_x_complex_output(self):
        result = emath.logn(2, -8)
        assert np.iscomplexobj(result)
        assert_allclose(result.real, 3.0)

    def test_negative_base_complex_output(self):
        result = emath.logn(-2, 8)
        assert np.iscomplexobj(result)

    def test_dtype_float32_to_complex128(self):
        # logn uses division which promotes to complex128
        x = np.array([-8.0], dtype=np.float32)
        result = emath.logn(2, x)
        assert result.dtype == np.complex128

    def test_dtype_float64_to_complex128(self):
        x = np.array([-8.0], dtype=np.float64)
        result = emath.logn(2, x)
        assert result.dtype == np.complex128

    def test_scalar_returns_scalar(self):
        result = emath.logn(2, -8)
        assert np.isscalar(result)

    def test_array_returns_array(self):
        result = emath.logn(2, np.array([-4, -8]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


class TestPower:
    """Tests for emath.power."""

    def test_positive_base_real_output(self):
        result = emath.power(2, 3)
        assert np.isreal(result)
        assert result == 8

    def test_negative_base_complex_output(self):
        result = emath.power(-2, 2)
        assert np.iscomplexobj(result)

    def test_negative_base_fractional_exponent(self):
        result = emath.power(-2, 0.5)
        assert np.iscomplexobj(result)
        assert result.imag != 0

    def test_negative_exponent_float_conversion(self):
        result = emath.power(2, -1)
        assert_allclose(result, 0.5)

    def test_dtype_float32_to_complex64(self):
        # Both args must be small types to preserve complex64
        x = np.array([-2.0], dtype=np.float32)
        p = np.array([2], dtype=np.int16)
        result = emath.power(x, p)
        assert result.dtype == np.complex64

    def test_dtype_float64_to_complex128(self):
        x = np.array([-2.0], dtype=np.float64)
        result = emath.power(x, 2)
        assert result.dtype == np.complex128

    def test_scalar_returns_scalar(self):
        result = emath.power(-2, 2)
        assert np.isscalar(result)

    def test_array_returns_array(self):
        result = emath.power(np.array([-2, -3]), 2)
        assert isinstance(result, np.ndarray)


class TestArccos:
    """Tests for emath.arccos."""

    def test_within_domain_real_output(self):
        result = emath.arccos(0.5)
        assert np.isreal(result)

    def test_outside_domain_complex_output(self):
        result = emath.arccos(2.0)
        assert np.iscomplexobj(result)

    def test_dtype_float32_to_complex64(self):
        arr = np.array([2.0], dtype=np.float32)
        result = emath.arccos(arr)
        assert result.dtype == np.complex64

    def test_dtype_float64_to_complex128(self):
        arr = np.array([2.0], dtype=np.float64)
        result = emath.arccos(arr)
        assert result.dtype == np.complex128

    def test_scalar_returns_scalar(self):
        result = emath.arccos(2)
        assert np.isscalar(result)

    def test_array_returns_array(self):
        result = emath.arccos(np.array([2, 3]))
        assert isinstance(result, np.ndarray)


class TestArcsin:
    """Tests for emath.arcsin."""

    def test_within_domain_real_output(self):
        result = emath.arcsin(0.5)
        assert np.isreal(result)

    def test_outside_domain_complex_output(self):
        result = emath.arcsin(2.0)
        assert np.iscomplexobj(result)

    def test_dtype_float32_to_complex64(self):
        arr = np.array([2.0], dtype=np.float32)
        result = emath.arcsin(arr)
        assert result.dtype == np.complex64

    def test_dtype_float64_to_complex128(self):
        arr = np.array([2.0], dtype=np.float64)
        result = emath.arcsin(arr)
        assert result.dtype == np.complex128

    def test_scalar_returns_scalar(self):
        result = emath.arcsin(2)
        assert np.isscalar(result)

    def test_array_returns_array(self):
        result = emath.arcsin(np.array([2, 3]))
        assert isinstance(result, np.ndarray)


class TestArctanh:
    """Tests for emath.arctanh."""

    def test_within_domain_real_output(self):
        result = emath.arctanh(0.5)
        assert np.isreal(result)

    def test_outside_domain_complex_output(self):
        result = emath.arctanh(2.0)
        assert np.iscomplexobj(result)

    def test_dtype_float32_to_complex64(self):
        arr = np.array([2.0], dtype=np.float32)
        result = emath.arctanh(arr)
        assert result.dtype == np.complex64

    def test_dtype_float64_to_complex128(self):
        arr = np.array([2.0], dtype=np.float64)
        result = emath.arctanh(arr)
        assert result.dtype == np.complex128

    def test_scalar_returns_scalar(self):
        result = emath.arctanh(2)
        assert np.isscalar(result)

    def test_array_returns_array(self):
        result = emath.arctanh(np.array([2, 3]))
        assert isinstance(result, np.ndarray)
