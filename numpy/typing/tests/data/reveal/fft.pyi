from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

###

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

_f64_nd: npt.NDArray[np.float64]
_c128_nd: npt.NDArray[np.complex128]
_py_float_1d: list[float]

_i64: np.int64
_f32: np.float16
_f80: np.longdouble
_c64: np.complex64
_c160: np.clongdouble

_i64_2d: _Array2D[np.int64]
_f32_2d: _Array2D[np.float16]
_f80_2d: _Array2D[np.longdouble]
_c64_2d: _Array2D[np.complex64]
_c160_2d: _Array2D[np.clongdouble]

_i64_nd: npt.NDArray[np.int64]
_f32_nd: npt.NDArray[np.float16]
_f80_nd: npt.NDArray[np.longdouble]
_c64_nd: npt.NDArray[np.complex64]
_c160_nd: npt.NDArray[np.clongdouble]

###

# fftshift

assert_type(np.fft.fftshift(_f64_nd), npt.NDArray[np.float64])
assert_type(np.fft.fftshift(_py_float_1d, axes=0), npt.NDArray[Any])

# ifftshift

assert_type(np.fft.ifftshift(_f64_nd), npt.NDArray[np.float64])
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
...

# the other fft functions

assert_type(np.fft.fft(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.ifft(_f64_nd, axis=1), npt.NDArray[np.complex128])
assert_type(np.fft.rfft(_f64_nd, n=None), npt.NDArray[np.complex128])
assert_type(np.fft.irfft(_f64_nd, norm="ortho"), npt.NDArray[np.float64])
assert_type(np.fft.hfft(_f64_nd, n=2), npt.NDArray[np.float64])
assert_type(np.fft.ihfft(_f64_nd), npt.NDArray[np.complex128])

assert_type(np.fft.fftn(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.ifftn(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.rfftn(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.irfftn(_f64_nd), npt.NDArray[np.float64])

assert_type(np.fft.rfft2(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.ifft2(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.fft2(_f64_nd), npt.NDArray[np.complex128])
assert_type(np.fft.irfft2(_f64_nd), npt.NDArray[np.float64])
