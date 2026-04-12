from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

###

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

_f64_nd: npt.NDArray[np.float64]
_c128_nd: npt.NDArray[np.complex128]
_py_float_1d: list[float]
_py_complex_1d: list[complex]

_i64: np.int64
_f32: np.float16
_f80: np.longdouble
_c64: np.complex64
_c160: np.clongdouble

_i64_2d: _Array2D[np.int64]
_f16_2d: _Array2D[np.float16]
_f32_2d: _Array2D[np.float32]
_f80_2d: _Array2D[np.longdouble]
_c64_2d: _Array2D[np.complex64]
_c160_2d: _Array2D[np.clongdouble]

_i64_nd: npt.NDArray[np.int64]
_f32_nd: npt.NDArray[np.float32]
_f80_nd: npt.NDArray[np.longdouble]
_c64_nd: npt.NDArray[np.complex64]
_c160_nd: npt.NDArray[np.clongdouble]

###

# fftshift

assert_type(np.fft.fftshift(_py_float_1d, axes=0), npt.NDArray[Any])
assert_type(np.fft.fftshift(_f32_2d), _Array2D[np.float32])
assert_type(np.fft.fftshift(_f64_nd), npt.NDArray[np.float64])

# ifftshift

assert_type(np.fft.ifftshift(_f64_nd), npt.NDArray[np.float64])
assert_type(np.fft.ifftshift(_f32_2d), _Array2D[np.float32])
assert_type(np.fft.ifftshift(_py_float_1d, axes=0), npt.NDArray[Any])

# fftfreq

assert_type(np.fft.fftfreq(5), _Array1D[np.float64])
assert_type(np.fft.fftfreq(5, True), _Array1D[np.float64])
assert_type(np.fft.fftfreq(5, 1), _Array1D[np.float64])
assert_type(np.fft.fftfreq(5, 1.0), _Array1D[np.float64])
assert_type(np.fft.fftfreq(5, 1j), _Array1D[np.complex128 | Any])

assert_type(np.fft.fftfreq(5, _i64), _Array1D[np.float64])
assert_type(np.fft.fftfreq(5, _f32), _Array1D[np.float64])
assert_type(np.fft.fftfreq(5, _f80), _Array1D[np.longdouble])
assert_type(np.fft.fftfreq(5, _c64), _Array1D[np.complex128])
assert_type(np.fft.fftfreq(5, _c160), _Array1D[np.clongdouble])

assert_type(np.fft.fftfreq(5, _i64_2d), _Array2D[np.float64])
assert_type(np.fft.fftfreq(5, _f32_2d), _Array2D[np.float64])
assert_type(np.fft.fftfreq(5, _f80_2d), _Array2D[np.longdouble])
assert_type(np.fft.fftfreq(5, _c64_2d), _Array2D[np.complex128])
assert_type(np.fft.fftfreq(5, _c160_2d), _Array2D[np.clongdouble])

assert_type(np.fft.fftfreq(5, _i64_nd), npt.NDArray[np.float64])
assert_type(np.fft.fftfreq(5, _f32_nd), npt.NDArray[np.float64])
assert_type(np.fft.fftfreq(5, _f80_nd), npt.NDArray[np.longdouble])
assert_type(np.fft.fftfreq(5, _c64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.fftfreq(5, _c160_nd), npt.NDArray[np.clongdouble])

# rfftfreq  (same as fftfreq)

assert_type(np.fft.rfftfreq(5), _Array1D[np.float64])
assert_type(np.fft.rfftfreq(5, True), _Array1D[np.float64])
assert_type(np.fft.rfftfreq(5, 1), _Array1D[np.float64])
assert_type(np.fft.rfftfreq(5, 1.0), _Array1D[np.float64])
assert_type(np.fft.rfftfreq(5, 1j), _Array1D[np.complex128 | Any])

assert_type(np.fft.rfftfreq(5, _i64), _Array1D[np.float64])
assert_type(np.fft.rfftfreq(5, _f32), _Array1D[np.float64])
assert_type(np.fft.rfftfreq(5, _f80), _Array1D[np.longdouble])
assert_type(np.fft.rfftfreq(5, _c64), _Array1D[np.complex128])
assert_type(np.fft.rfftfreq(5, _c160), _Array1D[np.clongdouble])

assert_type(np.fft.rfftfreq(5, _i64_2d), _Array2D[np.float64])
assert_type(np.fft.rfftfreq(5, _f32_2d), _Array2D[np.float64])
assert_type(np.fft.rfftfreq(5, _f80_2d), _Array2D[np.longdouble])
assert_type(np.fft.rfftfreq(5, _c64_2d), _Array2D[np.complex128])
assert_type(np.fft.rfftfreq(5, _c160_2d), _Array2D[np.clongdouble])

assert_type(np.fft.rfftfreq(5, _i64_nd), npt.NDArray[np.float64])
assert_type(np.fft.rfftfreq(5, _f32_nd), npt.NDArray[np.float64])
assert_type(np.fft.rfftfreq(5, _f80_nd), npt.NDArray[np.longdouble])
assert_type(np.fft.rfftfreq(5, _c64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.rfftfreq(5, _c160_nd), npt.NDArray[np.clongdouble])

# *fft

assert_type(np.fft.fft(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.fft(_i64_2d), _Array2D[np.complex128])
assert_type(np.fft.fft(_f32_2d), _Array2D[np.complex64])
assert_type(np.fft.fft(_c64_2d), _Array2D[np.complex64])
assert_type(np.fft.fft(_py_float_1d), _Array1D[np.complex128])

assert_type(np.fft.ifft(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.ifft(_i64_2d), _Array2D[np.complex128])
assert_type(np.fft.ifft(_f32_2d), _Array2D[np.complex64])
assert_type(np.fft.ifft(_c64_2d), _Array2D[np.complex64])
assert_type(np.fft.ifft(_py_float_1d), _Array1D[np.complex128])

assert_type(np.fft.rfft(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.rfft(_i64_2d), _Array2D[np.complex128])
assert_type(np.fft.rfft(_f32_2d), _Array2D[np.complex64])
assert_type(np.fft.rfft(_py_float_1d), _Array1D[np.complex128])

assert_type(np.fft.irfft(_f64_nd), npt.NDArray[np.float64])
assert_type(np.fft.irfft(_i64_2d), _Array2D[np.float64])
assert_type(np.fft.irfft(_f16_2d), _Array2D[np.float16])
assert_type(np.fft.irfft(_f32_2d), _Array2D[np.float32])
assert_type(np.fft.irfft(_c64_2d), _Array2D[np.float32])
assert_type(np.fft.irfft(_py_complex_1d), _Array1D[np.float64])

assert_type(np.fft.hfft(_f64_nd), npt.NDArray[np.float64])
assert_type(np.fft.hfft(_i64_2d), _Array2D[np.float64])
assert_type(np.fft.hfft(_f16_2d), _Array2D[np.float16])
assert_type(np.fft.hfft(_f32_2d), _Array2D[np.float32])
assert_type(np.fft.hfft(_c64_2d), _Array2D[np.float32])
assert_type(np.fft.hfft(_py_complex_1d), _Array1D[np.float64])

assert_type(np.fft.ihfft(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.ihfft(_i64_2d), _Array2D[np.complex128])
assert_type(np.fft.ihfft(_f32_2d), _Array2D[np.complex64])
assert_type(np.fft.ihfft(_py_float_1d), _Array1D[np.complex128])

# *fftn

assert_type(np.fft.fftn(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.fftn(_i64_2d), _Array2D[np.complex128])
assert_type(np.fft.fftn(_f32_2d), _Array2D[np.complex64])
assert_type(np.fft.fftn(_c64_2d), _Array2D[np.complex64])
assert_type(np.fft.fftn(_py_float_1d), _Array1D[np.complex128])

assert_type(np.fft.ifftn(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.ifftn(_i64_2d), _Array2D[np.complex128])
assert_type(np.fft.ifftn(_f32_2d), _Array2D[np.complex64])
assert_type(np.fft.ifftn(_c64_2d), _Array2D[np.complex64])
assert_type(np.fft.ifftn(_py_float_1d), _Array1D[np.complex128])

assert_type(np.fft.rfftn(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.rfftn(_i64_2d), _Array2D[np.complex128])
assert_type(np.fft.rfftn(_f32_2d), _Array2D[np.complex64])
assert_type(np.fft.rfftn(_py_float_1d), _Array1D[np.complex128])

assert_type(np.fft.irfftn(_f64_nd), npt.NDArray[np.float64])
assert_type(np.fft.irfftn(_i64_2d), _Array2D[np.float64])
assert_type(np.fft.irfftn(_f16_2d), _Array2D[np.float16])
assert_type(np.fft.irfftn(_f32_2d), _Array2D[np.float32])
assert_type(np.fft.irfftn(_c64_2d), _Array2D[np.float32])
assert_type(np.fft.irfftn(_py_complex_1d), _Array1D[np.float64])

# *fft2

assert_type(np.fft.fft2(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.fft2(_i64_2d), _Array2D[np.complex128])
assert_type(np.fft.fft2(_f32_2d), _Array2D[np.complex64])
assert_type(np.fft.fft2(_c64_2d), _Array2D[np.complex64])
assert_type(np.fft.fft2(_py_float_1d), _Array1D[np.complex128])

assert_type(np.fft.ifft2(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.ifft2(_i64_2d), _Array2D[np.complex128])
assert_type(np.fft.ifft2(_f32_2d), _Array2D[np.complex64])
assert_type(np.fft.ifft2(_c64_2d), _Array2D[np.complex64])
assert_type(np.fft.ifft2(_py_float_1d), _Array1D[np.complex128])

assert_type(np.fft.rfft2(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.rfft2(_i64_2d), _Array2D[np.complex128])
assert_type(np.fft.rfft2(_f32_2d), _Array2D[np.complex64])
assert_type(np.fft.rfft2(_py_float_1d), _Array1D[np.complex128])

assert_type(np.fft.irfft2(_f64_nd), npt.NDArray[np.float64])
assert_type(np.fft.irfft2(_i64_2d), _Array2D[np.float64])
assert_type(np.fft.irfft2(_f16_2d), _Array2D[np.float16])
assert_type(np.fft.irfft2(_f32_2d), _Array2D[np.float32])
assert_type(np.fft.irfft2(_c64_2d), _Array2D[np.float32])
assert_type(np.fft.irfft2(_py_complex_1d), _Array1D[np.float64])
