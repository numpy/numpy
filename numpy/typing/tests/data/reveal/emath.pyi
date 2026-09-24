from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

###

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

_i64: np.int64
_i64_2d: _Array2D[np.int64]
_f32: np.float32
_f32_2d: _Array2D[np.float32]
_f64: np.float64
_f64_2d: _Array2D[np.float64]
_f64_nd: npt.NDArray[np.float64]
_c128: np.complex128
_c128_2d: _Array2D[np.complex128]
_c128_nd: npt.NDArray[np.complex128]

###

# sqrt
assert_type(np.emath.sqrt(_c128_2d), _Array2D[np.complex128])
assert_type(np.emath.sqrt(_f64_2d), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.sqrt(_f32_2d), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.sqrt(_c128), np.complex128)
assert_type(np.emath.sqrt(_f64), np.complex128 | np.float64)
assert_type(np.emath.sqrt(_f32), np.complex64 | np.float32)
assert_type(np.emath.sqrt(1j), np.complex128 | Any)
assert_type(np.emath.sqrt([_c128]), _Array1D[np.complex128])
assert_type(np.emath.sqrt([_f64]), _Array1D[np.complex128 | np.float64])
assert_type(np.emath.sqrt([_f32]), _Array1D[np.complex64 | np.float32])
assert_type(np.emath.sqrt([1j]), _Array1D[np.complex128])
assert_type(np.emath.sqrt([[_c128]]), _Array2D[np.complex128])
assert_type(np.emath.sqrt([[_f64]]), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.sqrt([[_f32]]), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.sqrt([[1j]]), _Array2D[np.complex128])
assert_type(np.emath.sqrt([_f64_nd]), npt.NDArray[Any] | Any)

# log
assert_type(np.emath.log(_c128_2d), _Array2D[np.complex128])
assert_type(np.emath.log(_f64_2d), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.log(_f32_2d), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.log(_c128), np.complex128)
assert_type(np.emath.log(_f64), np.complex128 | np.float64)
assert_type(np.emath.log(_f32), np.complex64 | np.float32)
assert_type(np.emath.log(1j), np.complex128 | Any)
assert_type(np.emath.log([_c128]), _Array1D[np.complex128])
assert_type(np.emath.log([_f64]), _Array1D[np.complex128 | np.float64])
assert_type(np.emath.log([_f32]), _Array1D[np.complex64 | np.float32])
assert_type(np.emath.log([1j]), _Array1D[np.complex128])
assert_type(np.emath.log([[_c128]]), _Array2D[np.complex128])
assert_type(np.emath.log([[_f64]]), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.log([[_f32]]), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.log([[1j]]), _Array2D[np.complex128])
assert_type(np.emath.log([_f64_nd]), npt.NDArray[Any] | Any)

# log10
assert_type(np.emath.log10(_c128_2d), _Array2D[np.complex128])
assert_type(np.emath.log10(_f64_2d), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.log10(_f32_2d), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.log10(_c128), np.complex128)
assert_type(np.emath.log10(_f64), np.complex128 | np.float64)
assert_type(np.emath.log10(_f32), np.complex64 | np.float32)
assert_type(np.emath.log10(1j), np.complex128 | Any)
assert_type(np.emath.log10([_c128]), _Array1D[np.complex128])
assert_type(np.emath.log10([_f64]), _Array1D[np.complex128 | np.float64])
assert_type(np.emath.log10([_f32]), _Array1D[np.complex64 | np.float32])
assert_type(np.emath.log10([1j]), _Array1D[np.complex128])
assert_type(np.emath.log10([[_c128]]), _Array2D[np.complex128])
assert_type(np.emath.log10([[_f64]]), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.log10([[_f32]]), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.log10([[1j]]), _Array2D[np.complex128])
assert_type(np.emath.log10([_f64_nd]), npt.NDArray[Any] | Any)

# log2
assert_type(np.emath.log2(_c128_2d), _Array2D[np.complex128])
assert_type(np.emath.log2(_f64_2d), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.log2(_f32_2d), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.log2(_c128), np.complex128)
assert_type(np.emath.log2(_f64), np.complex128 | np.float64)
assert_type(np.emath.log2(_f32), np.complex64 | np.float32)
assert_type(np.emath.log2(1j), np.complex128 | Any)
assert_type(np.emath.log2([_c128]), _Array1D[np.complex128])
assert_type(np.emath.log2([_f64]), _Array1D[np.complex128 | np.float64])
assert_type(np.emath.log2([_f32]), _Array1D[np.complex64 | np.float32])
assert_type(np.emath.log2([1j]), _Array1D[np.complex128])
assert_type(np.emath.log2([[_c128]]), _Array2D[np.complex128])
assert_type(np.emath.log2([[_f64]]), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.log2([[_f32]]), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.log2([[1j]]), _Array2D[np.complex128])
assert_type(np.emath.log2([_f64_nd]), npt.NDArray[Any] | Any)

# logn
assert_type(np.emath.logn(2, _c128_2d), _Array2D[np.complex128])
assert_type(np.emath.logn(2, _f64_2d), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.logn(_f32, _f32_2d), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.logn(1j, _f64_2d), _Array2D[np.complex128 | Any])
assert_type(np.emath.logn(2, _c128), np.complex128)
assert_type(np.emath.logn(2, _f64), np.complex128 | np.float64)
assert_type(np.emath.logn(_f32, _f32), np.complex64 | np.float32)
assert_type(np.emath.logn(2, 1j), np.complex128 | Any)
assert_type(np.emath.logn(2, [_c128]), _Array1D[np.complex128])
assert_type(np.emath.logn(2, [_f64]), _Array1D[np.complex128 | np.float64])
assert_type(np.emath.logn(_f32, [_f32]), _Array1D[np.complex64 | np.float32])
assert_type(np.emath.logn(2, [1j]), _Array1D[np.complex128 | Any])
assert_type(np.emath.logn(2, [[_c128]]), _Array2D[np.complex128])
assert_type(np.emath.logn(2, [[_f64]]), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.logn(_f32, [[_f32]]), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.logn(2, [[1j]]), _Array2D[np.complex128 | Any])
assert_type(np.emath.logn([2, 3], _f64_2d), npt.NDArray[Any] | Any)

# power
assert_type(np.emath.power(_c128_2d, 2), _Array2D[np.complex128])
assert_type(np.emath.power(_f64_2d, 2), _Array2D[np.complex128 | np.float64 | Any])
assert_type(np.emath.power(_f32_2d, _f32), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.power(_i64_2d, 2), _Array2D[np.complex128 | np.float64 | np.int64])
assert_type(np.emath.power(_c128, 2), np.complex128)
assert_type(np.emath.power(_f64, 2), np.complex128 | np.float64 | Any)
assert_type(np.emath.power(_f32, _f32), np.complex64 | np.float32)
assert_type(np.emath.power(_i64, 2), np.complex128 | np.float64 | np.int64)
assert_type(np.emath.power(_f64, 2j), np.complex128 | Any)
assert_type(np.emath.power([_c128], 2), _Array1D[np.complex128])
assert_type(np.emath.power([_f64], 2), _Array1D[np.complex128 | np.float64 | Any])
assert_type(np.emath.power([_f32], _f32), _Array1D[np.complex64 | np.float32])
assert_type(np.emath.power([_i64], [2, 4]), _Array1D[np.complex128 | np.float64 | np.int64])
assert_type(np.emath.power([_f64], 2j), _Array1D[np.complex128 | Any])
assert_type(np.emath.power([[_c128]], 2), _Array2D[np.complex128])
assert_type(np.emath.power([[_f64]], [2]), _Array2D[np.complex128 | np.float64 | Any])
assert_type(np.emath.power([[_f32]], _f32), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.power([[_i64]], 2), _Array2D[np.complex128 | np.float64 | np.int64])
assert_type(np.emath.power([[1j]], 2), _Array2D[np.complex128 | Any])
assert_type(np.emath.power(_f64_2d, [2]), npt.NDArray[Any] | Any)

# arccos
assert_type(np.emath.arccos(_c128_2d), _Array2D[np.complex128])
assert_type(np.emath.arccos(_f64_2d), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.arccos(_f32_2d), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.arccos(_c128), np.complex128)
assert_type(np.emath.arccos(_f64), np.complex128 | np.float64)
assert_type(np.emath.arccos(_f32), np.complex64 | np.float32)
assert_type(np.emath.arccos(1j), np.complex128 | Any)
assert_type(np.emath.arccos([_c128]), _Array1D[np.complex128])
assert_type(np.emath.arccos([_f64]), _Array1D[np.complex128 | np.float64])
assert_type(np.emath.arccos([_f32]), _Array1D[np.complex64 | np.float32])
assert_type(np.emath.arccos([1j]), _Array1D[np.complex128])
assert_type(np.emath.arccos([[_c128]]), _Array2D[np.complex128])
assert_type(np.emath.arccos([[_f64]]), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.arccos([[_f32]]), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.arccos([[1j]]), _Array2D[np.complex128])
assert_type(np.emath.arccos([_f64_nd]), npt.NDArray[Any] | Any)

# arcsin
assert_type(np.emath.arcsin(_c128_2d), _Array2D[np.complex128])
assert_type(np.emath.arcsin(_f64_2d), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.arcsin(_f32_2d), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.arcsin(_c128), np.complex128)
assert_type(np.emath.arcsin(_f64), np.complex128 | np.float64)
assert_type(np.emath.arcsin(_f32), np.complex64 | np.float32)
assert_type(np.emath.arcsin(1j), np.complex128 | Any)
assert_type(np.emath.arcsin([_c128]), _Array1D[np.complex128])
assert_type(np.emath.arcsin([_f64]), _Array1D[np.complex128 | np.float64])
assert_type(np.emath.arcsin([_f32]), _Array1D[np.complex64 | np.float32])
assert_type(np.emath.arcsin([1j]), _Array1D[np.complex128])
assert_type(np.emath.arcsin([[_c128]]), _Array2D[np.complex128])
assert_type(np.emath.arcsin([[_f64]]), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.arcsin([[_f32]]), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.arcsin([[1j]]), _Array2D[np.complex128])
assert_type(np.emath.arcsin([_f64_nd]), npt.NDArray[Any] | Any)

# arctanh
assert_type(np.emath.arctanh(_c128_2d), _Array2D[np.complex128])
assert_type(np.emath.arctanh(_f64_2d), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.arctanh(_f32_2d), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.arctanh(_c128), np.complex128)
assert_type(np.emath.arctanh(_f64), np.complex128 | np.float64)
assert_type(np.emath.arctanh(_f32), np.complex64 | np.float32)
assert_type(np.emath.arctanh(1j), np.complex128 | Any)
assert_type(np.emath.arctanh([_c128]), _Array1D[np.complex128])
assert_type(np.emath.arctanh([_f64]), _Array1D[np.complex128 | np.float64])
assert_type(np.emath.arctanh([_f32]), _Array1D[np.complex64 | np.float32])
assert_type(np.emath.arctanh([1j]), _Array1D[np.complex128])
assert_type(np.emath.arctanh([[_c128]]), _Array2D[np.complex128])
assert_type(np.emath.arctanh([[_f64]]), _Array2D[np.complex128 | np.float64])
assert_type(np.emath.arctanh([[_f32]]), _Array2D[np.complex64 | np.float32])
assert_type(np.emath.arctanh([[1j]]), _Array2D[np.complex128])
assert_type(np.emath.arctanh([_f64_nd]), npt.NDArray[Any] | Any)
