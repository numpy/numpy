from typing import Any, Literal as L, assert_type

import numpy as np
import numpy.typing as npt

###

type _FalseType = L[False]
type _TrueType = L[True]

_py_b0: _FalseType
_py_b1: _TrueType
_py_b: bool
_py_i: int

_b: np.bool[bool]
_b0: np.bool[_FalseType]
_b1: np.bool[_TrueType]

_u8: np.uint8
_i32: np.int32
_i64: np.int64
_u32: np.uint32
_u64: np.uint64

_i32_nd: npt.NDArray[np.int32]

###

assert_type(_i64 << _i64, np.int64)
assert_type(_i64 >> _i64, np.int64)
assert_type(_i64 | _i64, np.int64)
assert_type(_i64 ^ _i64, np.int64)
assert_type(_i64 & _i64, np.int64)

assert_type(_i32 << _i32, np.int32)
assert_type(_i32 >> _i32, np.int32)
assert_type(_i32 | _i32, np.int32)
assert_type(_i32 ^ _i32, np.int32)
assert_type(_i32 & _i32, np.int32)

assert_type(_i64 << _i32, np.signedinteger)
assert_type(_i64 >> _i32, np.signedinteger)
assert_type(_i64 | _i32, np.signedinteger)
assert_type(_i64 ^ _i32, np.signedinteger)
assert_type(_i64 & _i32, np.signedinteger)

assert_type(_i64 << _b, np.int64)
assert_type(_i64 >> _b, np.int64)
assert_type(_i64 | _b, np.int64)
assert_type(_i64 ^ _b, np.int64)
assert_type(_i64 & _b, np.int64)

assert_type(_i64 << _py_b, np.int64)
assert_type(_i64 >> _py_b, np.int64)
assert_type(_i64 | _py_b, np.int64)
assert_type(_i64 ^ _py_b, np.int64)
assert_type(_i64 & _py_b, np.int64)

assert_type(_u64 << _u64, np.uint64)
assert_type(_u64 >> _u64, np.uint64)
assert_type(_u64 | _u64, np.uint64)
assert_type(_u64 ^ _u64, np.uint64)
assert_type(_u64 & _u64, np.uint64)

assert_type(_u32 << _u32, np.uint32)
assert_type(_u32 >> _u32, np.uint32)
assert_type(_u32 | _u32, np.uint32)
assert_type(_u32 ^ _u32, np.uint32)
assert_type(_u32 & _u32, np.uint32)

assert_type(_u32 << _i32, np.signedinteger)
assert_type(_u32 >> _i32, np.signedinteger)
assert_type(_u32 | _i32, np.signedinteger)
assert_type(_u32 ^ _i32, np.signedinteger)
assert_type(_u32 & _i32, np.signedinteger)

assert_type(_u32 << _py_i, np.uint32)
assert_type(_u32 >> _py_i, np.uint32)
assert_type(_u32 | _py_i, np.uint32)
assert_type(_u32 ^ _py_i, np.uint32)
assert_type(_u32 & _py_i, np.uint32)

assert_type(_u64 << _b, np.uint64)
assert_type(_u64 >> _b, np.uint64)
assert_type(_u64 | _b, np.uint64)
assert_type(_u64 ^ _b, np.uint64)
assert_type(_u64 & _b, np.uint64)

assert_type(_u64 << _py_b, np.uint64)
assert_type(_u64 >> _py_b, np.uint64)
assert_type(_u64 | _py_b, np.uint64)
assert_type(_u64 ^ _py_b, np.uint64)
assert_type(_u64 & _py_b, np.uint64)

assert_type(_b << _b, np.int8)
assert_type(_b >> _b, np.int8)
assert_type(_b | _b, np.bool)
assert_type(_b ^ _b, np.bool)
assert_type(_b & _b, np.bool)

assert_type(_b << _py_b, np.int8)
assert_type(_b >> _py_b, np.int8)
assert_type(_b | _py_b, np.bool)
assert_type(_b ^ _py_b, np.bool)
assert_type(_b & _py_b, np.bool)

assert_type(_b << _py_i, np.int_)
assert_type(_b >> _py_i, np.int_)
assert_type(_b | _py_i, np.bool | np.int_)
assert_type(_b ^ _py_i, np.bool | np.int_)
assert_type(_b & _py_i, np.bool | np.int_)

assert_type(~_i64, np.int64)
assert_type(~_i32, np.int32)
assert_type(~_u64, np.uint64)
assert_type(~_u32, np.uint32)
assert_type(~_b, np.bool)
assert_type(~_b0, np.bool[_TrueType])
assert_type(~_b1, np.bool[_FalseType])
assert_type(~_i32_nd, npt.NDArray[np.int32])

assert_type(_b | _b0, np.bool)
assert_type(_b0 | _b, np.bool)
assert_type(_b | _b1, np.bool[_TrueType])
assert_type(_b1 | _b, np.bool[_TrueType])

assert_type(_b ^ _b0, np.bool)
assert_type(_b0 ^ _b, np.bool)
assert_type(_b ^ _b1, np.bool)
assert_type(_b1 ^ _b, np.bool)

assert_type(_b & _b0, np.bool[_FalseType])
assert_type(_b0 & _b, np.bool[_FalseType])
assert_type(_b & _b1, np.bool)
assert_type(_b1 & _b, np.bool)

assert_type(_b0 | _b0, np.bool[_FalseType])
assert_type(_b0 | _b1, np.bool[_TrueType])
assert_type(_b1 | _b0, np.bool[_TrueType])
assert_type(_b1 | _b1, np.bool[_TrueType])

assert_type(_b0 ^ _b0, np.bool[_FalseType])
assert_type(_b0 ^ _b1, np.bool[_TrueType])
assert_type(_b1 ^ _b0, np.bool[_TrueType])
assert_type(_b1 ^ _b1, np.bool[_FalseType])

assert_type(_b0 & _b0, np.bool[_FalseType])
assert_type(_b0 & _b1, np.bool[_FalseType])
assert_type(_b1 & _b0, np.bool[_FalseType])
assert_type(_b1 & _b1, np.bool[_TrueType])

###

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

_py_b_1d: list[bool]
_py_i_1d: list[int]

_b_1d: _Array1D[np.bool]
_b_2d: _Array2D[np.bool]
_u8_1d: _Array1D[np.uint8]
_u8_2d: _Array2D[np.uint8]
_i64_1d: _Array1D[np.int64]
_i64_2d: _Array2D[np.int64]

assert_type(_b_1d & _py_b, _Array1D[np.bool])
assert_type(_b_1d & _py_b_1d, npt.NDArray[np.bool])
assert_type(_b_1d & _py_i, npt.NDArray[Any])
assert_type(_b_1d & _py_i_1d, npt.NDArray[np.int64])
assert_type(_b_1d & _b, _Array1D[np.bool])
assert_type(_b_1d & _b_1d, _Array1D[np.bool])
assert_type(_b_1d & _b_2d, npt.NDArray[np.bool])
assert_type(_b_1d & _u8, _Array1D[np.uint8])
assert_type(_b_1d & _u8_1d, _Array1D[np.uint8])
assert_type(_b_1d & _u8_2d, npt.NDArray[Any])
assert_type(_b_1d & _i64, _Array1D[np.int64])
assert_type(_b_1d & _i64_1d, _Array1D[np.int64])
assert_type(_b_1d & _i64_2d, npt.NDArray[np.int64])
assert_type(_u8_1d & _py_b, _Array1D[np.uint8])
assert_type(_u8_1d & _py_b_1d, npt.NDArray[np.uint8])
assert_type(_u8_1d & _py_i, _Array1D[np.uint8])
assert_type(_u8_1d & _py_i_1d, npt.NDArray[np.int64])
assert_type(_u8_1d & _b, _Array1D[np.uint8])
assert_type(_u8_1d & _b_1d, _Array1D[np.uint8])
assert_type(_u8_1d & _b_2d, npt.NDArray[np.uint8])
assert_type(_u8_1d & _u8, _Array1D[np.uint8])
assert_type(_u8_1d & _u8_1d, _Array1D[np.uint8])
assert_type(_u8_1d & _u8_2d, npt.NDArray[np.uint8])
assert_type(_u8_1d & _i64, npt.NDArray[np.int64])
assert_type(_u8_1d & _i64_1d, npt.NDArray[np.int64])
assert_type(_u8_1d & _i64_2d, npt.NDArray[np.int64])
assert_type(_i64_1d & _py_b, _Array1D[np.int64])
assert_type(_i64_1d & _py_b_1d, npt.NDArray[np.int64])
assert_type(_i64_1d & _py_i, _Array1D[np.int64])
assert_type(_i64_1d & _py_i_1d, npt.NDArray[np.int64])
assert_type(_i64_1d & _b, _Array1D[np.int64])
assert_type(_i64_1d & _b_1d, _Array1D[np.int64])
assert_type(_i64_1d & _b_2d, npt.NDArray[np.int64])
assert_type(_i64_1d & _u8, _Array1D[np.int64])
assert_type(_i64_1d & _u8_1d, _Array1D[np.int64])
assert_type(_i64_1d & _u8_2d, npt.NDArray[np.int64])
assert_type(_i64_1d & _i64, _Array1D[np.int64])
assert_type(_i64_1d & _i64_1d, _Array1D[np.int64])
assert_type(_i64_1d & _i64_2d, npt.NDArray[np.int64])

assert_type(_py_b & _b_1d, _Array1D[np.bool])
assert_type(_py_b_1d & _b_1d, npt.NDArray[np.bool])
assert_type(_py_i & _b_1d, npt.NDArray[Any])
assert_type(_py_i_1d & _b_1d, npt.NDArray[np.int64])
assert_type(_b & _b_1d, _Array1D[np.bool])
assert_type(_b_2d & _b_1d, npt.NDArray[np.bool])
assert_type(_u8 & _b_1d, _Array1D[np.uint8])
assert_type(_u8_1d & _b_1d, _Array1D[np.uint8])
assert_type(_u8_2d & _b_1d, npt.NDArray[np.uint8])
assert_type(_i64 & _b_1d, _Array1D[np.int64])
assert_type(_i64_1d & _b_1d, _Array1D[np.int64])
assert_type(_i64_2d & _b_1d, npt.NDArray[np.int64])
assert_type(_py_b & _u8_1d, _Array1D[np.uint8])
assert_type(_py_b_1d & _u8_1d, npt.NDArray[np.uint8])
assert_type(_py_i & _u8_1d, _Array1D[np.uint8])
assert_type(_py_i_1d & _u8_1d, npt.NDArray[np.int64])
assert_type(_b & _u8_1d, _Array1D[np.uint8])
assert_type(_b_1d & _u8_1d, _Array1D[np.uint8])
assert_type(_b_2d & _u8_1d, npt.NDArray[Any])
assert_type(_u8 & _u8_1d, _Array1D[np.uint8])
assert_type(_u8_2d & _u8_1d, npt.NDArray[np.uint8])
assert_type(_i64 & _u8_1d, npt.NDArray[np.int64])
assert_type(_i64_1d & _u8_1d, _Array1D[np.int64])
assert_type(_i64_2d & _u8_1d, npt.NDArray[np.int64])
assert_type(_py_b & _i64_1d, _Array1D[np.int64])
assert_type(_py_b_1d & _i64_1d, npt.NDArray[np.int64])
assert_type(_py_i & _i64_1d, _Array1D[np.int64])
assert_type(_py_i_1d & _i64_1d, npt.NDArray[np.int64])
assert_type(_b & _i64_1d, _Array1D[np.int64])
assert_type(_b_1d & _i64_1d, _Array1D[np.int64])
assert_type(_b_2d & _i64_1d, npt.NDArray[np.int64])
assert_type(_u8 & _i64_1d, _Array1D[np.int64])
assert_type(_u8_1d & _i64_1d, npt.NDArray[np.int64])
assert_type(_u8_2d & _i64_1d, npt.NDArray[np.int64])
assert_type(_i64 & _i64_1d, _Array1D[np.int64])
assert_type(_i64_2d & _i64_1d, npt.NDArray[np.int64])

# `ndarray.__xor__` and `ndarray.__xor__` are identical to `ndarray.__and__`

assert_type(_b_1d << _py_b, _Array1D[np.int8])
assert_type(_b_1d << _py_b_1d, npt.NDArray[np.int8])
assert_type(_b_1d << _py_i, npt.NDArray[Any])
assert_type(_b_1d << _py_i_1d, npt.NDArray[np.int64])
assert_type(_b_1d << _b, _Array1D[np.int8])
assert_type(_b_1d << _b_1d, _Array1D[np.int8])
assert_type(_b_1d << _b_2d, npt.NDArray[np.int8])
assert_type(_b_1d << _u8, _Array1D[np.uint8])
assert_type(_b_1d << _u8_1d, _Array1D[np.uint8])
assert_type(_b_1d << _u8_2d, npt.NDArray[Any])
assert_type(_b_1d << _i64, _Array1D[np.int64])
assert_type(_b_1d << _i64_1d, _Array1D[np.int64])
assert_type(_b_1d << _i64_2d, npt.NDArray[np.int64])
assert_type(_u8_1d << _py_b, _Array1D[np.uint8])
assert_type(_u8_1d << _py_b_1d, npt.NDArray[np.uint8])
assert_type(_u8_1d << _py_i, _Array1D[np.uint8])
assert_type(_u8_1d << _py_i_1d, npt.NDArray[np.int64])
assert_type(_u8_1d << _b, _Array1D[np.uint8])
assert_type(_u8_1d << _b_1d, _Array1D[np.uint8])
assert_type(_u8_1d << _b_2d, npt.NDArray[np.uint8])
assert_type(_u8_1d << _u8, _Array1D[np.uint8])
assert_type(_u8_1d << _u8_1d, _Array1D[np.uint8])
assert_type(_u8_1d << _u8_2d, npt.NDArray[np.uint8])
assert_type(_u8_1d << _i64, npt.NDArray[np.int64])
assert_type(_u8_1d << _i64_1d, npt.NDArray[np.int64])
assert_type(_u8_1d << _i64_2d, npt.NDArray[np.int64])
assert_type(_i64_1d << _py_b, _Array1D[np.int64])
assert_type(_i64_1d << _py_b_1d, npt.NDArray[np.int64])
assert_type(_i64_1d << _py_i, _Array1D[np.int64])
assert_type(_i64_1d << _py_i_1d, npt.NDArray[np.int64])
assert_type(_i64_1d << _b, _Array1D[np.int64])
assert_type(_i64_1d << _b_1d, _Array1D[np.int64])
assert_type(_i64_1d << _b_2d, npt.NDArray[np.int64])
assert_type(_i64_1d << _u8, _Array1D[np.int64])
assert_type(_i64_1d << _u8_1d, _Array1D[np.int64])
assert_type(_i64_1d << _u8_2d, npt.NDArray[np.int64])
assert_type(_i64_1d << _i64, _Array1D[np.int64])
assert_type(_i64_1d << _i64_1d, _Array1D[np.int64])
assert_type(_i64_1d << _i64_2d, npt.NDArray[np.int64])

assert_type(_py_b << _b_1d, _Array1D[np.int8])
assert_type(_py_b_1d << _b_1d, npt.NDArray[np.int8])
assert_type(_py_i << _b_1d, npt.NDArray[Any])
assert_type(_py_i_1d << _b_1d, npt.NDArray[np.int64])
assert_type(_b << _b_1d, _Array1D[np.int8])
assert_type(_b_2d << _b_1d, npt.NDArray[np.int8])
assert_type(_u8 << _b_1d, _Array1D[np.uint8])
assert_type(_u8_1d << _b_1d, _Array1D[np.uint8])
assert_type(_u8_2d << _b_1d, npt.NDArray[np.uint8])
assert_type(_i64 << _b_1d, _Array1D[np.int64])
assert_type(_i64_1d << _b_1d, _Array1D[np.int64])
assert_type(_i64_2d << _b_1d, npt.NDArray[np.int64])
assert_type(_py_b << _u8_1d, _Array1D[np.uint8])
assert_type(_py_b_1d << _u8_1d, npt.NDArray[np.uint8])
assert_type(_py_i << _u8_1d, _Array1D[np.uint8])
assert_type(_py_i_1d << _u8_1d, npt.NDArray[np.int64])
assert_type(_b << _u8_1d, _Array1D[np.uint8])
assert_type(_b_1d << _u8_1d, _Array1D[np.uint8])
assert_type(_b_2d << _u8_1d, npt.NDArray[Any])
assert_type(_u8 << _u8_1d, _Array1D[np.uint8])
assert_type(_u8_2d << _u8_1d, npt.NDArray[np.uint8])
assert_type(_i64 << _u8_1d, npt.NDArray[np.int64])
assert_type(_i64_1d << _u8_1d, _Array1D[np.int64])
assert_type(_i64_2d << _u8_1d, npt.NDArray[np.int64])
assert_type(_py_b << _i64_1d, _Array1D[np.int64])
assert_type(_py_b_1d << _i64_1d, npt.NDArray[np.int64])
assert_type(_py_i << _i64_1d, _Array1D[np.int64])
assert_type(_py_i_1d << _i64_1d, npt.NDArray[np.int64])
assert_type(_b << _i64_1d, _Array1D[np.int64])
assert_type(_b_1d << _i64_1d, _Array1D[np.int64])
assert_type(_b_2d << _i64_1d, npt.NDArray[np.int64])
assert_type(_u8 << _i64_1d, _Array1D[np.int64])
assert_type(_u8_1d << _i64_1d, npt.NDArray[np.int64])
assert_type(_u8_2d << _i64_1d, npt.NDArray[np.int64])
assert_type(_i64 << _i64_1d, _Array1D[np.int64])
assert_type(_i64_2d << _i64_1d, npt.NDArray[np.int64])
