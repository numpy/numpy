import datetime as dt
from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
from numpy._typing import _64Bit, _128Bit

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

b: bool
c: complex
f: float
i: int

c16: np.complex128
c8: np.complex64

# Can't directly import `np.float128` as it is not available on all platforms
f16: np.floating[_128Bit]
f8: np.float64
f4: np.float32

i8: np.int64
i4: np.int32

u8: np.uint64
u4: np.uint32

b_: np.bool

M8: np.datetime64
M8_none: np.datetime64[None]
M8_int: np.datetime64[int]
date: dt.date
time: dt.datetime

m8: np.timedelta64
m8_none: np.timedelta64[None]
m8_int: np.timedelta64[int]
m8_delta: np.timedelta64[dt.timedelta]
delta: dt.timedelta

AR_b: npt.NDArray[np.bool]
AR_u: npt.NDArray[np.uint32]
AR_i: npt.NDArray[np.int64]
AR_f: npt.NDArray[np.float64]
AR_c: npt.NDArray[np.complex128]
AR_m: npt.NDArray[np.timedelta64]
AR_M: npt.NDArray[np.datetime64]
AR_O: npt.NDArray[np.object_]
AR_S: npt.NDArray[np.bytes_]

AR_b_2d: _Array2D[np.bool]
AR_u32_2d: _Array2D[np.uint32]
AR_u64_2d: _Array2D[np.uint64]
AR_i32_2d: _Array2D[np.int32]
AR_i64_2d: _Array2D[np.int64]
AR_f32_2d: _Array2D[np.float32]
AR_f64_2d: _Array2D[np.float64]
AR_f64_1d: _Array1D[np.float64]
AR_c128_2d: _Array2D[np.complex128]
AR_m8_2d: _Array2D[np.timedelta64[int]]
AR_M8_2d: _Array2D[np.datetime64[int]]
AR_T_2d: np.ndarray[tuple[int, int], np.dtypes.StringDType]
AR_Any_2d: _Array2D[Any]

AR_U: npt.NDArray[np.str_]
AR_T: np.ndarray[tuple[Any, ...], np.dtypes.StringDType]
AR_floating: npt.NDArray[np.floating]

AR_LIKE_b: list[bool]
AR_LIKE_u: list[np.uint32]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_c: list[complex]
AR_LIKE_m: list[np.timedelta64[int]]
AR_LIKE_M: list[np.datetime64[int]]
AR_LIKE_O: list[np.object_[int]]

###

# unary ops

assert_type(-f16, np.floating[_128Bit])
assert_type(-c16, np.complex128)
assert_type(-c8, np.complex64)
assert_type(-f8, np.float64)
assert_type(-f4, np.float32)
assert_type(-i8, np.int64)
assert_type(-i4, np.int32)
assert_type(-u8, np.uint64)
assert_type(-u4, np.uint32)
assert_type(-m8, np.timedelta64)
assert_type(-m8_none, np.timedelta64[None])
assert_type(-m8_int, np.timedelta64[int])
assert_type(-m8_delta, np.timedelta64[dt.timedelta])
assert_type(-AR_f, npt.NDArray[np.float64])

assert_type(+f16, np.floating[_128Bit])
assert_type(+c16, np.complex128)
assert_type(+c8, np.complex64)
assert_type(+f8, np.float64)
assert_type(+f4, np.float32)
assert_type(+i8, np.int64)
assert_type(+i4, np.int32)
assert_type(+u8, np.uint64)
assert_type(+u4, np.uint32)
assert_type(+m8_none, np.timedelta64[None])
assert_type(+m8_int, np.timedelta64[int])
assert_type(+m8_delta, np.timedelta64[dt.timedelta])
assert_type(+AR_f, npt.NDArray[np.float64])

assert_type(abs(f16), np.floating[_128Bit])
assert_type(abs(c16), np.float64)
assert_type(abs(c8), np.float32)
assert_type(abs(f8), np.float64)
assert_type(abs(f4), np.float32)
assert_type(abs(i8), np.int64)
assert_type(abs(i4), np.int32)
assert_type(abs(u8), np.uint64)
assert_type(abs(u4), np.uint32)
assert_type(abs(m8), np.timedelta64)
assert_type(abs(m8_none), np.timedelta64[None])
assert_type(abs(m8_int), np.timedelta64[int])
assert_type(abs(m8_delta), np.timedelta64[dt.timedelta])
assert_type(abs(b_), np.bool)
assert_type(abs(AR_O), npt.NDArray[np.object_])

# Time structures

assert_type(M8 + m8, np.datetime64)
assert_type(M8 + i, np.datetime64)
assert_type(M8 + i8, np.datetime64)
assert_type(M8 - M8, np.timedelta64)
assert_type(M8 - i, np.datetime64)
assert_type(M8 - i8, np.datetime64)

assert_type(M8_none + i, np.datetime64[None])
assert_type(M8_none - i, np.datetime64[None])

assert_type(M8_none + i8, np.datetime64[None])
assert_type(M8_none - i8, np.datetime64[None])

# NOTE: Mypy incorrectly infers `timedelta64[Any]`, but pyright behaves correctly.
assert_type(M8_none + m8, np.datetime64[None])  # type: ignore[assert-type]
assert_type(M8_none - M8, np.timedelta64[None])  # type: ignore[assert-type]
# NOTE: Mypy incorrectly infers `datetime64[Any]`, but pyright behaves correctly.
assert_type(M8_none - m8, np.datetime64[None])  # type: ignore[assert-type]

assert_type(m8 + m8, np.timedelta64)
assert_type(m8 + i, np.timedelta64)  # type: ignore[deprecated]
assert_type(m8 + i8, np.timedelta64)  # type: ignore[deprecated]
assert_type(m8 - m8, np.timedelta64)
assert_type(m8 - i, np.timedelta64)  # type: ignore[deprecated]
assert_type(m8 - i8, np.timedelta64)  # type: ignore[deprecated]
assert_type(m8 * f, np.timedelta64)
assert_type(m8 * f4, np.timedelta64)
assert_type(m8 * np.True_, np.timedelta64)
assert_type(m8 / f, np.timedelta64)
assert_type(m8 / f4, np.timedelta64)
assert_type(m8 / m8, np.float64)
assert_type(m8 // m8, np.int64)
assert_type(m8 % m8, np.timedelta64)
# NOTE: Mypy incorrectly infers `tuple[Any, ...]`, but pyright behaves correctly.
assert_type(divmod(m8, m8), tuple[np.int64, np.timedelta64])  # type: ignore[assert-type]

assert_type(m8_none + m8, np.timedelta64[None])
assert_type(m8_none + i, np.timedelta64[None])  # type: ignore[deprecated]
assert_type(m8_none + i8, np.timedelta64[None])  # type: ignore[deprecated]
assert_type(m8_none - i, np.timedelta64[None])  # type: ignore[deprecated]
assert_type(m8_none - i8, np.timedelta64[None])  # type: ignore[deprecated]

assert_type(m8_int + i, np.timedelta64[int])  # type: ignore[deprecated]
assert_type(m8_int + m8_delta, np.timedelta64[int])
assert_type(m8_int + m8, np.timedelta64)
assert_type(m8_int - i, np.timedelta64[int])  # type: ignore[deprecated]
assert_type(m8_int - m8_delta, np.timedelta64[int])
assert_type(m8_int - m8_int, np.timedelta64[int])
assert_type(m8_int - m8_none, np.timedelta64[None])
assert_type(m8_int - m8, np.timedelta64)

assert_type(m8_delta + date, dt.date)
assert_type(m8_delta + time, dt.datetime)
assert_type(m8_delta + delta, dt.timedelta)
assert_type(m8_delta - delta, dt.timedelta)
assert_type(m8_delta / delta, float)
assert_type(m8_delta // delta, int)
assert_type(m8_delta % delta, dt.timedelta)
assert_type(divmod(m8_delta, delta), tuple[int, dt.timedelta])

# boolean

assert_type(b_ / b, np.float64)
assert_type(b_ / b_, np.float64)
assert_type(b_ / i, np.float64)
assert_type(b_ / i8, np.float64)
assert_type(b_ / i4, np.float64)
assert_type(b_ / u8, np.float64)
assert_type(b_ / u4, np.float64)
assert_type(b_ / f, np.float64)
assert_type(b_ / f16, np.floating[_128Bit])
assert_type(b_ / f8, np.float64)
assert_type(b_ / f4, np.float32)
assert_type(b_ / c, np.complex128)
assert_type(b_ / c16, np.complex128)
assert_type(b_ / c8, np.complex64)

assert_type(b / b_, np.float64)
assert_type(b_ / b_, np.float64)
assert_type(i / b_, np.float64)
assert_type(i8 / b_, np.float64)
assert_type(i4 / b_, np.float64)
assert_type(u8 / b_, np.float64)
assert_type(u4 / b_, np.float64)
assert_type(f / b_, np.float64)
assert_type(f16 / b_, np.floating[_128Bit])
assert_type(f8 / b_, np.float64)
assert_type(f4 / b_, np.float32)
assert_type(c / b_, np.complex128)
assert_type(c16 / b_, np.complex128)
assert_type(c8 / b_, np.complex64)

# Complex

assert_type(c16 + f16, np.complexfloating)
assert_type(c16 + c16, np.complex128)
assert_type(c16 + f8, np.complex128)
assert_type(c16 + i8, np.complex128)
assert_type(c16 + c8, np.complex128)
assert_type(c16 + f4, np.complex128)
assert_type(c16 + i4, np.complex128)
assert_type(c16 + b_, np.complex128)
assert_type(c16 + b, np.complex128)
assert_type(c16 + c, np.complex128)
assert_type(c16 + f, np.complex128)
assert_type(c16 + AR_f, npt.NDArray[np.complex128])

assert_type(f16 + c16, np.complexfloating)
assert_type(c16 + c16, np.complex128)
assert_type(f8 + c16, np.complex128)
assert_type(i8 + c16, np.complex128)
assert_type(c8 + c16, np.complex128)
assert_type(f4 + c16, np.complexfloating)
assert_type(i4 + c16, np.complex128)
assert_type(b_ + c16, np.complex128)
assert_type(b + c16, np.complex128)
assert_type(c + c16, np.complex128)
assert_type(f + c16, np.complex128)
assert_type(AR_f + c16, npt.NDArray[np.complex128])

assert_type(c8 + f16, np.complex64 | np.complexfloating[_128Bit, _128Bit])
assert_type(c8 + c16, np.complex128)
assert_type(c8 + f8, np.complex64 | np.complex128)
assert_type(c8 + i8, np.complex64 | np.complexfloating[_64Bit, _64Bit])
assert_type(c8 + c8, np.complex64)
assert_type(c8 + f4, np.complex64)
assert_type(c8 + i4, np.complex64)
assert_type(c8 + b_, np.complex64)
assert_type(c8 + b, np.complex64)
assert_type(c8 + c, np.complex64 | np.complex128)
assert_type(c8 + f, np.complex64 | np.complex128)
assert_type(c8 + AR_f, npt.NDArray[Any])

assert_type(f16 + c8, np.complexfloating[_128Bit, _128Bit] | np.complex64)
assert_type(c16 + c8, np.complex128)
assert_type(f8 + c8, np.complexfloating[_64Bit, _64Bit])
assert_type(i8 + c8, np.complexfloating[_64Bit, _64Bit] | np.complex64)
assert_type(c8 + c8, np.complex64)
assert_type(f4 + c8, np.complex64)
assert_type(i4 + c8, np.complex64)
assert_type(b_ + c8, np.complex64)
assert_type(b + c8, np.complex64)
assert_type(c + c8, np.complex64 | np.complex128)
assert_type(f + c8, np.complex64 | np.complex128)
assert_type(AR_f + c8, Any)

# Float

assert_type(f8 + f16, np.floating)
assert_type(f8 + f8, np.float64)
assert_type(f8 + i8, np.float64)
assert_type(f8 + f4, np.float64)
assert_type(f8 + i4, np.float64)
assert_type(f8 + b_, np.float64)
assert_type(f8 + b, np.float64)
assert_type(f8 + c, np.float64 | np.complex128)
assert_type(f8 + f, np.float64)
assert_type(f8 + AR_f, npt.NDArray[np.float64])

assert_type(f16 + f8, np.floating)
assert_type(f8 + f8, np.float64)
assert_type(i8 + f8, np.float64)
assert_type(f4 + f8, np.float64)
assert_type(i4 + f8, np.float64)
assert_type(b_ + f8, np.float64)
assert_type(b + f8, np.float64)
assert_type(c + f8, complex)
assert_type(f + f8, np.float64)
assert_type(AR_f + f8, npt.NDArray[np.float64])

assert_type(f4 + f16, np.floating)
assert_type(f4 + f8, np.float64)
assert_type(f4 + i8, np.floating)
assert_type(f4 + f4, np.float32)
assert_type(f4 + i4, np.floating)
assert_type(f4 + b_, np.float32)
assert_type(f4 + b, np.float32)
assert_type(f4 + c, np.complexfloating)
assert_type(f4 + f, np.float32)
assert_type(f4 + AR_f, npt.NDArray[np.float64])

assert_type(f16 + f4, np.floating)
assert_type(f8 + f4, np.float64)
assert_type(i8 + f4, np.floating)
assert_type(f4 + f4, np.float32)
assert_type(i4 + f4, np.floating)
assert_type(b_ + f4, np.float32)
assert_type(b + f4, np.float32)
assert_type(c + f4, np.complexfloating)
assert_type(f + f4, np.float32)
assert_type(AR_f + f4, npt.NDArray[np.float64])

# Int

assert_type(i8 + i8, np.int64)
assert_type(i8 + u8, Any)
assert_type(i8 + i4, np.signedinteger)
assert_type(i8 + u4, Any)
assert_type(i8 + b_, np.int64)
assert_type(i8 + b, np.int64)
assert_type(i8 + c, np.complex128)
assert_type(i8 + f, np.float64)
assert_type(i8 + AR_f, npt.NDArray[np.float64])

assert_type(u8 + u8, np.uint64)
assert_type(u8 + i4, Any)
assert_type(u8 + u4, np.unsignedinteger)
assert_type(u8 + b_, np.uint64)
assert_type(u8 + b, np.uint64)
assert_type(u8 + c, np.complex128)
assert_type(u8 + f, np.float64)
assert_type(u8 + AR_f, npt.NDArray[np.float64])

assert_type(i8 + i8, np.int64)
assert_type(u8 + i8, Any)
assert_type(i4 + i8, np.signedinteger)
assert_type(u4 + i8, Any)
assert_type(b_ + i8, np.int64)
assert_type(b + i8, np.int64)
assert_type(c + i8, np.complex128)
assert_type(f + i8, np.float64)
assert_type(AR_f + i8, npt.NDArray[np.float64])

assert_type(u8 + u8, np.uint64)
assert_type(i4 + u8, Any)
assert_type(u4 + u8, np.unsignedinteger)
assert_type(b_ + u8, np.uint64)
assert_type(b + u8, np.uint64)
assert_type(c + u8, np.complex128)
assert_type(f + u8, np.float64)
assert_type(AR_f + u8, npt.NDArray[np.float64])

assert_type(i4 + i8, np.signedinteger)
assert_type(i4 + i4, np.int32)
assert_type(i4 + b_, np.int32)
assert_type(i4 + b, np.int32)
assert_type(i4 + AR_f, npt.NDArray[np.float64])

assert_type(u4 + i8, Any)
assert_type(u4 + i4, Any)
assert_type(u4 + u8, np.unsignedinteger)
assert_type(u4 + u4, np.uint32)
assert_type(u4 + b_, np.uint32)
assert_type(u4 + b, np.uint32)
assert_type(u4 + AR_f, npt.NDArray[np.float64])

assert_type(i8 + i4, np.signedinteger)
assert_type(i4 + i4, np.int32)
assert_type(b_ + i4, np.int32)
assert_type(b + i4, np.int32)
assert_type(AR_f + i4, npt.NDArray[np.float64])

assert_type(i8 + u4, Any)
assert_type(i4 + u4, Any)
assert_type(u8 + u4, np.unsignedinteger)
assert_type(u4 + u4, np.uint32)
assert_type(b_ + u4, np.uint32)
assert_type(b + u4, np.uint32)
assert_type(AR_f + u4, npt.NDArray[np.float64])

# regression tests for https://github.com/numpy/numpy/issues/28805

assert_type(AR_floating + f, npt.NDArray[np.floating])
assert_type(AR_floating - f, npt.NDArray[np.floating])
assert_type(AR_floating * f, npt.NDArray[np.floating])
assert_type(AR_floating ** f, npt.NDArray[np.floating])
assert_type(AR_floating / f, npt.NDArray[np.floating])
assert_type(AR_floating // f, npt.NDArray[np.floating])
assert_type(AR_floating % f, npt.NDArray[np.floating])
assert_type(divmod(AR_floating, f), tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]])

assert_type(f + AR_floating, npt.NDArray[np.floating])
assert_type(f - AR_floating, npt.NDArray[np.floating])
assert_type(f * AR_floating, npt.NDArray[np.floating])
assert_type(f ** AR_floating, npt.NDArray[np.floating])
assert_type(f / AR_floating, npt.NDArray[np.floating])
assert_type(f // AR_floating, npt.NDArray[np.floating])
assert_type(f % AR_floating, npt.NDArray[np.floating])
assert_type(divmod(f, AR_floating), tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]])

# character-like

assert_type(AR_S + b"", npt.NDArray[np.bytes_])
assert_type(AR_S + [b""], npt.NDArray[np.bytes_])
assert_type([b""] + AR_S, npt.NDArray[np.bytes_])
assert_type(AR_S + AR_S, npt.NDArray[np.bytes_])

assert_type(AR_U + "", npt.NDArray[np.str_])
assert_type(AR_U + [""], npt.NDArray[np.str_])
assert_type("" + AR_U, npt.NDArray[np.str_])
assert_type([""] + AR_U, npt.NDArray[np.str_])
assert_type(AR_U + AR_U, npt.NDArray[np.str_])

assert_type(AR_T + "", np.ndarray[tuple[Any, ...], np.dtypes.StringDType])
assert_type(AR_T + [""], np.ndarray[tuple[Any, ...], np.dtypes.StringDType])
assert_type("" + AR_T, np.ndarray[tuple[Any, ...], np.dtypes.StringDType])
assert_type([""] + AR_T, np.ndarray[tuple[Any, ...], np.dtypes.StringDType])
assert_type(AR_T + AR_T, np.ndarray[tuple[Any, ...], np.dtypes.StringDType])  # type: ignore[assert-type]
assert_type(AR_T + AR_U, np.ndarray[tuple[Any, ...], np.dtypes.StringDType])
assert_type(AR_U + AR_T, np.ndarray[tuple[Any, ...], np.dtypes.StringDType])  # type: ignore[assert-type]

assert_type(AR_T * i, np.ndarray[tuple[Any, ...], np.dtypes.StringDType])
assert_type(AR_T * AR_LIKE_i, np.ndarray[tuple[Any, ...], np.dtypes.StringDType])
assert_type(AR_T * AR_i, np.ndarray[tuple[Any, ...], np.dtypes.StringDType])
assert_type(i * AR_T, np.ndarray[tuple[Any, ...], np.dtypes.StringDType])
# mypy incorrectly infers `AR_LIKE_i * AR_T` as `list[int]`
assert_type(AR_i * AR_T, np.ndarray[tuple[Any, ...], np.dtypes.StringDType])  # type: ignore[assert-type]

# __add__

assert_type(AR_Any_2d + 1, _Array2D[Any])
assert_type(AR_b_2d + 1, _Array2D[np.int64])
assert_type(AR_b_2d + AR_f32_2d, _Array2D[np.float32])
assert_type(AR_u32_2d + AR_u64_2d, _Array2D[np.uint64])
assert_type(AR_u64_2d + AR_u32_2d, _Array2D[np.uint64])
assert_type(AR_i32_2d + AR_i64_2d, _Array2D[np.int64])
assert_type(AR_i64_2d + 1, _Array2D[np.int64])
assert_type(AR_i64_2d + 1.0, _Array2D[np.float64])
assert_type(AR_i64_2d + m8_int, _Array2D[np.timedelta64])
assert_type(AR_i64_2d + AR_u64_2d, Any)
assert_type(AR_i64_2d + AR_i32_2d, _Array2D[np.int64])
assert_type(AR_i64_2d + AR_f, npt.NDArray[np.float64])
assert_type(AR_f32_2d + 1.0, _Array2D[np.float32])
assert_type(AR_f32_2d + 1j, _Array2D[np.complex64])
assert_type(AR_f32_2d + AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d + 1j, _Array2D[np.complex128])
assert_type(AR_f64_2d + AR_f32_2d, _Array2D[np.float64])
assert_type(AR_f64_2d + AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d + AR_c128_2d, _Array2D[np.complex128])
assert_type(AR_f64_2d + AR_f, npt.NDArray[np.float64])
assert_type(AR_c128_2d + 1j, _Array2D[np.complex128])
assert_type(AR_c128_2d + AR_f64_2d, _Array2D[np.complex128])
assert_type(AR_M8_2d + m8_int, _Array2D[np.datetime64])

assert_type(AR_b + AR_LIKE_b, npt.NDArray[np.bool])
assert_type(AR_b + AR_LIKE_u, npt.NDArray[np.uint32])
assert_type(AR_u + AR_LIKE_i, npt.NDArray[np.int64])
assert_type(AR_i + AR_LIKE_i, npt.NDArray[np.int64])
assert_type(AR_i + AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f + AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f + AR_LIKE_c, npt.NDArray[np.complex128])
assert_type(AR_f + AR_LIKE_O, npt.NDArray[np.object_])
assert_type(AR_c + AR_LIKE_c, npt.NDArray[np.complex128])
assert_type(AR_m + AR_LIKE_m, npt.NDArray[np.timedelta64])
assert_type(AR_m + AR_LIKE_M, npt.NDArray[np.datetime64])
assert_type(AR_M + AR_LIKE_m, npt.NDArray[np.datetime64])
assert_type(AR_O + 1, npt.NDArray[np.object_])

# __radd__

assert_type(1 + AR_Any_2d, _Array2D[Any])
assert_type(1 + AR_b_2d, _Array2D[np.int64])
assert_type(1 + AR_i64_2d, _Array2D[np.int64])
assert_type(1.0 + AR_i64_2d, _Array2D[np.float64])
assert_type(1.0 + AR_f32_2d, _Array2D[np.float32])
assert_type(1j + AR_f32_2d, _Array2D[np.complex64])
assert_type(True + AR_f64_2d, _Array2D[np.float64])
assert_type(1j + AR_f64_2d, _Array2D[np.complex128])
assert_type(1j + AR_c128_2d, _Array2D[np.complex128])

assert_type(AR_LIKE_b + AR_b, npt.NDArray[np.bool])
assert_type(AR_LIKE_u + AR_b, npt.NDArray[np.uint32])
assert_type(AR_LIKE_i + AR_u, npt.NDArray[np.int64])
assert_type(AR_LIKE_i + AR_i, npt.NDArray[np.int64])
assert_type(AR_LIKE_f + AR_i, npt.NDArray[np.float64])
assert_type(AR_LIKE_f + AR_f, npt.NDArray[np.float64])
assert_type(AR_LIKE_c + AR_f, npt.NDArray[np.complex128])
assert_type(AR_LIKE_O + AR_f, npt.NDArray[np.object_])
assert_type(AR_LIKE_c + AR_c, npt.NDArray[np.complex128])
assert_type(AR_LIKE_m + AR_m, npt.NDArray[np.timedelta64])
assert_type(AR_LIKE_M + AR_m, npt.NDArray[np.datetime64])
assert_type(AR_LIKE_m + AR_M, npt.NDArray[np.datetime64])
assert_type(1 + AR_O, npt.NDArray[np.object_])

# __sub__

assert_type(AR_Any_2d - 1, _Array2D[Any])
assert_type(AR_b_2d - 1, _Array2D[np.int64])
assert_type(AR_b_2d - AR_f32_2d, _Array2D[np.float32])
assert_type(AR_u32_2d - AR_u64_2d, _Array2D[np.uint64])
assert_type(AR_u64_2d - AR_u32_2d, _Array2D[np.uint64])
assert_type(AR_i32_2d - AR_i64_2d, _Array2D[np.int64])
assert_type(AR_i64_2d - 1, _Array2D[np.int64])
assert_type(AR_i64_2d - 1.0, _Array2D[np.float64])
assert_type(AR_i64_2d - m8_int, _Array2D[np.timedelta64])
assert_type(AR_i64_2d - AR_u64_2d, Any)
assert_type(AR_i64_2d - AR_i32_2d, _Array2D[np.int64])
assert_type(AR_i64_2d - AR_f, npt.NDArray[np.float64])
assert_type(AR_f32_2d - 1.0, _Array2D[np.float32])
assert_type(AR_f32_2d - 1j, _Array2D[np.complex64])
assert_type(AR_f32_2d - AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d - 1j, _Array2D[np.complex128])
assert_type(AR_f64_2d - AR_f32_2d, _Array2D[np.float64])
assert_type(AR_f64_2d - AR_f64_1d, npt.NDArray[np.float64])
assert_type(AR_f64_2d - AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d - AR_c128_2d, _Array2D[np.complex128])
assert_type(AR_f64_2d - AR_f, npt.NDArray[np.float64])
assert_type(AR_c128_2d - 1j, _Array2D[np.complex128])
assert_type(AR_c128_2d - AR_f64_2d, _Array2D[np.complex128])
assert_type(AR_M8_2d - m8_int, _Array2D[np.datetime64])
assert_type(AR_M8_2d - AR_M8_2d, _Array2D[np.timedelta64])
assert_type(AR_O - 1, npt.NDArray[np.object_])

# __rsub__

assert_type(1 - AR_Any_2d, _Array2D[Any])
assert_type(1 - AR_b_2d, _Array2D[np.int64])
assert_type(1 - AR_i64_2d, _Array2D[np.int64])
assert_type(1.0 - AR_i64_2d, _Array2D[np.float64])
assert_type(m8_int - AR_i64_2d, _Array2D[np.timedelta64])
assert_type(M8_int - AR_i64_2d, _Array2D[np.datetime64])
assert_type(1.0 - AR_f32_2d, _Array2D[np.float32])
assert_type(1j - AR_f32_2d, _Array2D[np.complex64])
assert_type(True - AR_f64_2d, _Array2D[np.float64])
assert_type(1j - AR_f64_2d, _Array2D[np.complex128])
assert_type(1j - AR_c128_2d, _Array2D[np.complex128])
assert_type(M8_int - AR_M8_2d, _Array2D[np.timedelta64])

assert_type(AR_LIKE_u - AR_b, npt.NDArray[np.uint32])
assert_type(AR_LIKE_i - AR_u, npt.NDArray[np.int64])
assert_type(AR_LIKE_i - AR_i, npt.NDArray[np.int64])
assert_type(AR_LIKE_f - AR_i, npt.NDArray[np.float64])
assert_type(AR_LIKE_f - AR_f, npt.NDArray[np.float64])
assert_type(AR_LIKE_c - AR_f, npt.NDArray[np.complex128])
assert_type(AR_LIKE_O - AR_f, npt.NDArray[np.object_])
assert_type(AR_LIKE_c - AR_c, npt.NDArray[np.complex128])
assert_type(AR_LIKE_m - AR_m, npt.NDArray[np.timedelta64])
assert_type(AR_LIKE_M - AR_m, npt.NDArray[np.datetime64])
assert_type(AR_LIKE_M - AR_M, npt.NDArray[np.timedelta64])
assert_type(1 - AR_O, npt.NDArray[np.object_])

# __mul__

assert_type(AR_Any_2d * 1, _Array2D[Any])
assert_type(AR_b_2d * 1, _Array2D[np.int64])
assert_type(AR_b_2d * AR_f32_2d, _Array2D[np.float32])
assert_type(AR_u32_2d * AR_u64_2d, _Array2D[np.uint64])
assert_type(AR_u64_2d * AR_u32_2d, _Array2D[np.uint64])
assert_type(AR_i32_2d * AR_i64_2d, _Array2D[np.int64])
assert_type(AR_i64_2d * 1, _Array2D[np.int64])
assert_type(AR_i64_2d * 1.0, _Array2D[np.float64])
assert_type(AR_i64_2d * AR_u64_2d, Any)
assert_type(AR_i64_2d * AR_i32_2d, _Array2D[np.int64])
assert_type(AR_i64_2d * AR_m8_2d, _Array2D[np.timedelta64])
assert_type(AR_i64_2d * AR_T_2d, np.ndarray[tuple[int, int], np.dtypes.StringDType])
assert_type(AR_i64_2d * AR_f, npt.NDArray[np.float64])
assert_type(AR_f32_2d * 1.0, _Array2D[np.float32])
assert_type(AR_f32_2d * 1j, _Array2D[np.complex64])
assert_type(AR_f32_2d * AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d * 1j, _Array2D[np.complex128])
assert_type(AR_f64_2d * AR_f32_2d, _Array2D[np.float64])
assert_type(AR_f64_2d * AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d * AR_c128_2d, _Array2D[np.complex128])
assert_type(AR_f64_2d * AR_f, npt.NDArray[np.float64])
assert_type(AR_c128_2d * 1j, _Array2D[np.complex128])
assert_type(AR_c128_2d * AR_f64_2d, _Array2D[np.complex128])
assert_type(AR_m8_2d * AR_f64_2d, _Array2D[np.timedelta64])
assert_type(AR_T_2d * AR_i64_2d, np.ndarray[tuple[int, int], np.dtypes.StringDType])

assert_type(AR_b * AR_LIKE_b, npt.NDArray[np.bool])
assert_type(AR_b * AR_LIKE_u, npt.NDArray[np.uint32])
assert_type(AR_u * AR_LIKE_i, npt.NDArray[np.int64])
assert_type(AR_i * AR_LIKE_b, npt.NDArray[np.int64])
assert_type(AR_i * AR_LIKE_i, npt.NDArray[np.int64])
assert_type(AR_i * AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f * AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f * AR_LIKE_c, npt.NDArray[np.complex128])
assert_type(AR_f * AR_LIKE_m, npt.NDArray[np.timedelta64])
assert_type(AR_f * AR_LIKE_O, npt.NDArray[np.object_])
assert_type(AR_c * AR_LIKE_c, npt.NDArray[np.complex128])
assert_type(AR_m * AR_LIKE_f, npt.NDArray[np.timedelta64])
assert_type(AR_O * 1, npt.NDArray[np.object_])

# __rmul__

assert_type(1 * AR_Any_2d, _Array2D[Any])
assert_type(1 * AR_b_2d, _Array2D[np.int64])
assert_type(i8 * AR_b_2d, _Array2D[np.int64])
assert_type(1 * AR_i64_2d, _Array2D[np.int64])
assert_type(1.0 * AR_i64_2d, _Array2D[np.float64])
assert_type(m8_int * AR_i64_2d, _Array2D[np.timedelta64])
assert_type(1.0 * AR_f32_2d, _Array2D[np.float32])
assert_type(1j * AR_f32_2d, _Array2D[np.complex64])
assert_type(True * AR_f64_2d, _Array2D[np.float64])
assert_type(1j * AR_f64_2d, _Array2D[np.complex128])
assert_type(1j * AR_c128_2d, _Array2D[np.complex128])
assert_type(1.5 * AR_m8_2d, _Array2D[np.timedelta64])
assert_type(2 * AR_T_2d, np.ndarray[tuple[int, int], np.dtypes.StringDType])

# mypy incorrectly infers `list * ndarray` as `list`
assert_type(AR_LIKE_b * AR_b, npt.NDArray[np.bool])  # type: ignore[assert-type]
assert_type(AR_LIKE_u * AR_b, npt.NDArray[np.uint32])  # type: ignore[assert-type]
assert_type(AR_LIKE_i * AR_b, npt.NDArray[np.int64])  # type: ignore[assert-type]
assert_type(AR_LIKE_f * AR_b, npt.NDArray[np.float64])  # type: ignore[assert-type]
assert_type(AR_LIKE_b * AR_f, npt.NDArray[np.float64])  # type: ignore[assert-type]
assert_type(AR_LIKE_f * AR_f, npt.NDArray[np.float64])  # type: ignore[assert-type]
assert_type(AR_LIKE_c * AR_f, npt.NDArray[np.complex128])  # type: ignore[assert-type]
assert_type(AR_LIKE_m * AR_f, npt.NDArray[np.timedelta64])  # type: ignore[assert-type]
assert_type(AR_LIKE_O * AR_f, npt.NDArray[np.object_])  # type: ignore[assert-type]
assert_type(AR_LIKE_c * AR_c, npt.NDArray[np.complex128])  # type: ignore[assert-type]
assert_type(AR_LIKE_f * AR_m, npt.NDArray[np.timedelta64])  # type: ignore[assert-type]
assert_type(1 * AR_O, npt.NDArray[np.object_])

# __pow__

assert_type(AR_Any_2d ** 1, _Array2D[Any])
assert_type(AR_b_2d ** 1, _Array2D[np.int64 | Any])
assert_type(AR_b_2d ** AR_b_2d, _Array2D[np.int8])
assert_type(AR_b_2d ** AR_f32_2d, _Array2D[np.float32])
assert_type(AR_u32_2d ** AR_u64_2d, _Array2D[np.uint64])
assert_type(AR_u64_2d ** AR_u32_2d, _Array2D[np.uint64])
assert_type(AR_i32_2d ** AR_i64_2d, _Array2D[np.int64])
assert_type(AR_i64_2d ** 1, _Array2D[np.int64])
assert_type(AR_i64_2d ** 1.0, _Array2D[np.float64])
assert_type(AR_i64_2d ** AR_u64_2d, Any)
assert_type(AR_i64_2d ** AR_i32_2d, _Array2D[np.int64])
assert_type(AR_i64_2d ** AR_f, npt.NDArray[np.float64])
assert_type(AR_f32_2d ** 1.0, _Array2D[np.float32])
assert_type(AR_f32_2d ** 1j, _Array2D[np.complex64])
assert_type(AR_f32_2d ** AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d ** 1j, _Array2D[np.complex128])
assert_type(AR_f64_2d ** AR_f32_2d, _Array2D[np.float64])
assert_type(AR_f64_2d ** AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d ** AR_c128_2d, _Array2D[np.complex128])
assert_type(AR_f64_2d ** AR_f, npt.NDArray[np.float64])
assert_type(AR_c128_2d ** 1j, _Array2D[np.complex128])
assert_type(AR_c128_2d ** AR_f64_2d, _Array2D[np.complex128])

assert_type(AR_b ** AR_LIKE_b, npt.NDArray[np.int8])
assert_type(AR_b ** AR_LIKE_u, npt.NDArray[np.uint32])
assert_type(AR_u ** AR_LIKE_i, npt.NDArray[np.int64])
assert_type(AR_i ** AR_LIKE_b, npt.NDArray[np.int64])
assert_type(AR_i ** AR_LIKE_i, npt.NDArray[np.int64])
assert_type(AR_i ** AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f ** AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f ** AR_LIKE_c, npt.NDArray[np.complex128])
assert_type(AR_f ** AR_LIKE_O, npt.NDArray[np.object_])
assert_type(AR_c ** AR_LIKE_c, npt.NDArray[np.complex128])
assert_type(AR_O ** 1, npt.NDArray[np.object_])

# __rpow__

assert_type(1 ** AR_Any_2d, _Array2D[Any])
assert_type(1 ** AR_b_2d, _Array2D[np.int64 | Any])
assert_type(True ** AR_b_2d, _Array2D[np.int8])
assert_type(i8 ** AR_b_2d, _Array2D[np.int64])
assert_type(1 ** AR_i64_2d, _Array2D[np.int64])
assert_type(1.0 ** AR_i64_2d, _Array2D[np.float64])
assert_type(1.0 ** AR_f32_2d, _Array2D[np.float32])
assert_type(1j ** AR_f32_2d, _Array2D[np.complex64])
assert_type(True ** AR_f64_2d, _Array2D[np.float64])
assert_type(1j ** AR_f64_2d, _Array2D[np.complex128])
assert_type(1j ** AR_c128_2d, _Array2D[np.complex128])

assert_type(AR_LIKE_b ** AR_b, npt.NDArray[np.int8])
assert_type(AR_LIKE_u ** AR_b, npt.NDArray[np.uint32])
assert_type(AR_LIKE_i ** AR_b, npt.NDArray[np.int64])
assert_type(AR_LIKE_f ** AR_b, npt.NDArray[np.float64])
assert_type(AR_LIKE_b ** AR_f, npt.NDArray[np.float64])
assert_type(AR_LIKE_f ** AR_f, npt.NDArray[np.float64])
assert_type(AR_LIKE_c ** AR_f, npt.NDArray[np.complex128])
assert_type(AR_LIKE_O ** AR_f, npt.NDArray[np.object_])
assert_type(AR_LIKE_c ** AR_c, npt.NDArray[np.complex128])
assert_type(1 ** AR_O, npt.NDArray[np.object_])

# __truediv__

assert_type(AR_Any_2d / 1, _Array2D[Any])
assert_type(AR_b_2d / 1, _Array2D[np.float64])
assert_type(AR_b_2d / AR_f32_2d, _Array2D[np.float32])
assert_type(AR_b_2d / AR_f64_1d, npt.NDArray[np.float64])
assert_type(AR_i64_2d / 1j, _Array2D[np.complex128 | Any])
assert_type(AR_i64_2d / AR_i32_2d, _Array2D[np.float64])
assert_type(AR_i64_2d / AR_f, npt.NDArray[np.float64])
assert_type(AR_f32_2d / 2, _Array2D[np.float32])
assert_type(AR_f32_2d / 1.0, _Array2D[np.float32])
assert_type(AR_f32_2d / 1j, _Array2D[np.complex64])
assert_type(AR_f32_2d / AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f32_2d / AR_f64_1d, npt.NDArray[np.float64])
assert_type(AR_f64_2d / 1j, _Array2D[np.complex128])
assert_type(AR_f64_2d / AR_f32_2d, _Array2D[np.float64])
assert_type(AR_f64_2d / AR_f64_1d, npt.NDArray[np.float64])
assert_type(AR_f64_2d / AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d / AR_c128_2d, _Array2D[np.complex128])
assert_type(AR_f64_2d / AR_f, npt.NDArray[np.float64])
assert_type(AR_c128_2d / 1j, _Array2D[np.complex128])
assert_type(AR_c128_2d / AR_f64_2d, _Array2D[np.complex128])
assert_type(AR_m8_2d / AR_m8_2d, _Array2D[np.float64])
assert_type(AR_m8_2d / AR_f64_2d, _Array2D[np.timedelta64])

assert_type(AR_i / AR_LIKE_i, npt.NDArray[np.float64])
assert_type(AR_f / AR_LIKE_b, npt.NDArray[np.float64])
assert_type(AR_f / AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f / AR_LIKE_c, npt.NDArray[np.complex128])
assert_type(AR_f / AR_LIKE_O, npt.NDArray[np.object_])
assert_type(AR_c / AR_LIKE_c, npt.NDArray[np.complex128])
assert_type(AR_m / AR_LIKE_f, npt.NDArray[np.timedelta64])
assert_type(AR_m / AR_LIKE_m, npt.NDArray[np.float64])
assert_type(AR_O / 1, npt.NDArray[np.object_])

# __rtruediv__

assert_type(1 / AR_Any_2d, _Array2D[Any])
assert_type(1 / AR_b_2d, _Array2D[np.float64])
assert_type(f4 / AR_b_2d, _Array2D[np.float32])
assert_type(1j / AR_i64_2d, _Array2D[np.complex128 | Any])
assert_type(m8_int / AR_i64_2d, _Array2D[np.timedelta64])
assert_type(2 / AR_f32_2d, _Array2D[np.float32])
assert_type(1.0 / AR_f32_2d, _Array2D[np.float32])
assert_type(1j / AR_f32_2d, _Array2D[np.complex64])
assert_type(True / AR_f64_2d, _Array2D[np.float64])
assert_type(1j / AR_f64_2d, _Array2D[np.complex128])
assert_type(1j / AR_c128_2d, _Array2D[np.complex128])
assert_type(m8_int / AR_m8_2d, _Array2D[np.float64])

assert_type(AR_LIKE_b / AR_f, npt.NDArray[np.float64])
assert_type(AR_LIKE_i / AR_i, npt.NDArray[np.float64])
assert_type(AR_LIKE_f / AR_f, npt.NDArray[np.float64])
assert_type(AR_LIKE_c / AR_f, npt.NDArray[np.complex128])
assert_type(AR_LIKE_O / AR_f, npt.NDArray[np.object_])
assert_type(AR_LIKE_c / AR_c, npt.NDArray[np.complex128])
assert_type(AR_LIKE_m / AR_f, npt.NDArray[np.timedelta64])
assert_type(AR_LIKE_m / AR_m, npt.NDArray[np.float64])
assert_type(1 / AR_O, npt.NDArray[np.object_])

# __floordiv__

assert_type(AR_Any_2d // 1, _Array2D[Any])
assert_type(AR_b_2d // 1, _Array2D[np.int64 | Any])
assert_type(AR_b_2d // AR_b_2d, _Array2D[np.int8])
assert_type(AR_b_2d // AR_f32_2d, _Array2D[np.float32])
assert_type(AR_u32_2d // AR_u64_2d, _Array2D[np.uint64])
assert_type(AR_u64_2d // AR_u32_2d, _Array2D[np.uint64])
assert_type(AR_i32_2d // AR_i64_2d, _Array2D[np.int64])
assert_type(AR_i64_2d // 1, _Array2D[np.int64])
assert_type(AR_i64_2d // 1.0, _Array2D[np.float64])
assert_type(AR_i64_2d // AR_u64_2d, Any)
assert_type(AR_i64_2d // AR_i32_2d, _Array2D[np.int64])
assert_type(AR_i64_2d // AR_f, npt.NDArray[np.float64])
assert_type(AR_f32_2d // 1.0, _Array2D[np.float32])
assert_type(AR_f32_2d // AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d // AR_f32_2d, _Array2D[np.float64])
assert_type(AR_f64_2d // AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d // AR_f, npt.NDArray[np.float64])
assert_type(AR_m8_2d // AR_m8_2d, _Array2D[np.int64])
assert_type(AR_m8_2d // AR_f64_2d, _Array2D[np.timedelta64])

assert_type(AR_b // AR_LIKE_b, npt.NDArray[np.int8])
assert_type(AR_b // AR_LIKE_u, npt.NDArray[np.uint32])
assert_type(AR_u // AR_LIKE_i, npt.NDArray[np.int64])
assert_type(AR_i // AR_LIKE_b, npt.NDArray[np.int64])
assert_type(AR_i // AR_LIKE_i, npt.NDArray[np.int64])
assert_type(AR_i // AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f // AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f // AR_LIKE_O, npt.NDArray[np.object_])
assert_type(AR_m // AR_LIKE_f, npt.NDArray[np.timedelta64])
assert_type(AR_m // AR_LIKE_m, npt.NDArray[np.int64])
assert_type(AR_O // 1, npt.NDArray[np.object_])

# __rfloordiv__

assert_type(1 // AR_Any_2d, _Array2D[Any])
assert_type(1 // AR_b_2d, _Array2D[np.int64 | Any])
assert_type(True // AR_b_2d, _Array2D[np.int8])
assert_type(i8 // AR_b_2d, _Array2D[np.int64])
assert_type(1 // AR_i64_2d, _Array2D[np.int64])
assert_type(1.0 // AR_i64_2d, _Array2D[np.float64])
assert_type(m8_int // AR_i64_2d, _Array2D[np.timedelta64])
assert_type(1.0 // AR_f32_2d, _Array2D[np.float32])
assert_type(True // AR_f64_2d, _Array2D[np.float64])
assert_type(m8_int // AR_m8_2d, _Array2D[np.int64])

assert_type(AR_LIKE_b // AR_b, npt.NDArray[np.int8])
assert_type(AR_LIKE_u // AR_b, npt.NDArray[np.uint32])
assert_type(AR_LIKE_i // AR_b, npt.NDArray[np.int64])
assert_type(AR_LIKE_f // AR_b, npt.NDArray[np.float64])
assert_type(AR_LIKE_b // AR_f, npt.NDArray[np.float64])
assert_type(AR_LIKE_f // AR_f, npt.NDArray[np.float64])
assert_type(AR_LIKE_O // AR_f, npt.NDArray[np.object_])
assert_type(AR_LIKE_m // AR_f, npt.NDArray[np.timedelta64])
assert_type(AR_LIKE_m // AR_m, npt.NDArray[np.int64])
assert_type(1 // AR_O, npt.NDArray[np.object_])

# __mod__

assert_type(AR_Any_2d % 1, _Array2D[Any])
assert_type(AR_b_2d % 1, _Array2D[np.int64 | Any])
assert_type(AR_b_2d % AR_b_2d, _Array2D[np.int8])
assert_type(AR_b_2d % AR_f32_2d, _Array2D[np.float32])
assert_type(AR_u32_2d % AR_u64_2d, _Array2D[np.uint64])
assert_type(AR_u64_2d % AR_u32_2d, _Array2D[np.uint64])
assert_type(AR_i32_2d % AR_i64_2d, _Array2D[np.int64])
assert_type(AR_i64_2d % 1, _Array2D[np.int64])
assert_type(AR_i64_2d % 1.0, _Array2D[np.float64])
assert_type(AR_i64_2d % AR_u64_2d, Any)
assert_type(AR_i64_2d % AR_i32_2d, _Array2D[np.int64])
assert_type(AR_i64_2d % AR_f, npt.NDArray[np.float64])
assert_type(AR_f32_2d % 1.0, _Array2D[np.float32])
assert_type(AR_f32_2d % AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d % AR_f32_2d, _Array2D[np.float64])
assert_type(AR_f64_2d % AR_f64_2d, _Array2D[np.float64])
assert_type(AR_f64_2d % AR_f, npt.NDArray[np.float64])
assert_type(AR_m8_2d % AR_m8_2d, _Array2D[np.timedelta64])

assert_type(AR_b % AR_LIKE_b, npt.NDArray[np.int8])
assert_type(AR_b % AR_LIKE_u, npt.NDArray[np.uint32])
assert_type(AR_u % AR_LIKE_i, npt.NDArray[np.int64])
assert_type(AR_i % AR_LIKE_b, npt.NDArray[np.int64])
assert_type(AR_i % AR_LIKE_i, npt.NDArray[np.int64])
assert_type(AR_i % AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f % AR_LIKE_f, npt.NDArray[np.float64])
assert_type(AR_f % AR_LIKE_O, npt.NDArray[np.object_])
assert_type(AR_m % AR_LIKE_m, npt.NDArray[np.timedelta64])
assert_type(AR_O % 1, npt.NDArray[np.object_])

# __rmod__

assert_type(1 % AR_Any_2d, _Array2D[Any])
assert_type(1 % AR_b_2d, _Array2D[np.int64 | Any])
assert_type(True % AR_b_2d, _Array2D[np.int8])
assert_type(i8 % AR_b_2d, _Array2D[np.int64])
assert_type(1 % AR_i64_2d, _Array2D[np.int64])
assert_type(1.0 % AR_i64_2d, _Array2D[np.float64])
assert_type(1.0 % AR_f32_2d, _Array2D[np.float32])
assert_type(True % AR_f64_2d, _Array2D[np.float64])
assert_type(m8_int % AR_m8_2d, _Array2D[np.timedelta64])

assert_type(AR_LIKE_b % AR_b, npt.NDArray[np.int8])
assert_type(AR_LIKE_u % AR_b, npt.NDArray[np.uint32])
assert_type(AR_LIKE_i % AR_b, npt.NDArray[np.int64])
assert_type(AR_LIKE_f % AR_b, npt.NDArray[np.float64])
assert_type(AR_LIKE_b % AR_f, npt.NDArray[np.float64])
assert_type(AR_LIKE_f % AR_f, npt.NDArray[np.float64])
assert_type(AR_LIKE_O % AR_f, npt.NDArray[np.object_])
assert_type(AR_LIKE_m % AR_m, npt.NDArray[np.timedelta64])
assert_type(1 % AR_O, npt.NDArray[np.object_])

# __divmod__

# NOTE: the __divmod__ calls are workarounds for https://github.com/microsoft/pyright/issues/9663
assert_type(divmod(AR_Any_2d, 1), tuple[_Array2D[Any], _Array2D[Any]])
assert_type(AR_b_2d.__divmod__(1), tuple[_Array2D[np.int64 | Any], _Array2D[np.int64 | Any]])
assert_type(divmod(AR_b_2d, AR_b_2d), tuple[_Array2D[np.int8], _Array2D[np.int8]])
assert_type(divmod(AR_b_2d, AR_f32_2d), tuple[_Array2D[np.float32], _Array2D[np.float32]])
assert_type(divmod(AR_u32_2d, AR_u64_2d), tuple[_Array2D[np.uint64], _Array2D[np.uint64]])
assert_type(divmod(AR_u64_2d, AR_u32_2d), tuple[_Array2D[np.uint64], _Array2D[np.uint64]])
assert_type(divmod(AR_i32_2d, AR_i64_2d), tuple[_Array2D[np.int64], _Array2D[np.int64]])
assert_type(AR_i64_2d.__divmod__(1), tuple[_Array2D[np.int64], _Array2D[np.int64]])
assert_type(AR_i64_2d.__divmod__(1.0), tuple[_Array2D[np.float64], _Array2D[np.float64]])
assert_type(divmod(AR_i64_2d, AR_u64_2d), tuple[npt.NDArray[Any], npt.NDArray[Any]])
assert_type(divmod(AR_i64_2d, AR_i32_2d), tuple[_Array2D[np.int64], _Array2D[np.int64]])
assert_type(divmod(AR_i64_2d, AR_f), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(AR_f32_2d.__divmod__(1.0), tuple[_Array2D[np.float32], _Array2D[np.float32]])
assert_type(divmod(AR_f32_2d, AR_f64_2d), tuple[_Array2D[np.float64], _Array2D[np.float64]])
assert_type(divmod(AR_f64_2d, AR_f32_2d), tuple[_Array2D[np.float64], _Array2D[np.float64]])
assert_type(divmod(AR_f64_2d, AR_f64_2d), tuple[_Array2D[np.float64], _Array2D[np.float64]])
assert_type(divmod(AR_f64_2d, AR_f), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(divmod(AR_m8_2d, AR_m8_2d), tuple[_Array2D[np.int64], _Array2D[np.timedelta64]])

assert_type(AR_b.__divmod__(AR_LIKE_b), tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]])
assert_type(AR_b.__divmod__(AR_LIKE_u), tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]])
assert_type(AR_u.__divmod__(AR_LIKE_i), tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]])
assert_type(AR_i.__divmod__(AR_LIKE_b), tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]])
assert_type(AR_i.__divmod__(AR_LIKE_i), tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]])
assert_type(AR_i.__divmod__(AR_LIKE_f), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(AR_f.__divmod__(AR_LIKE_f), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(AR_m.__divmod__(AR_LIKE_m), tuple[npt.NDArray[np.int64], npt.NDArray[np.timedelta64]])

# __rdivmod__

assert_type(divmod(1, AR_Any_2d), tuple[_Array2D[Any], _Array2D[Any]])
assert_type(divmod(1, AR_b_2d), tuple[_Array2D[np.int64 | Any], _Array2D[np.int64 | Any]])
assert_type(divmod(True, AR_b_2d), tuple[_Array2D[np.int8], _Array2D[np.int8]])
assert_type(divmod(i8, AR_b_2d), tuple[_Array2D[np.int64], _Array2D[np.int64]])
assert_type(divmod(1, AR_i64_2d), tuple[_Array2D[np.int64], _Array2D[np.int64]])
assert_type(divmod(1.0, AR_i64_2d), tuple[_Array2D[np.float64], _Array2D[np.float64]])
assert_type(divmod(1.0, AR_f32_2d), tuple[_Array2D[np.float32], _Array2D[np.float32]])
assert_type(divmod(True, AR_f64_2d), tuple[_Array2D[np.float64], _Array2D[np.float64]])
assert_type(divmod(m8_int, AR_m8_2d), tuple[_Array2D[np.int64], _Array2D[np.timedelta64]])

assert_type(divmod(AR_LIKE_b, AR_b), tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]])
assert_type(divmod(AR_LIKE_u, AR_b), tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]])
assert_type(divmod(AR_LIKE_i, AR_b), tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]])
assert_type(divmod(AR_LIKE_f, AR_b), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(divmod(AR_LIKE_b, AR_f), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(divmod(AR_LIKE_f, AR_f), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(divmod(AR_LIKE_m, AR_m), tuple[npt.NDArray[np.int64], npt.NDArray[np.timedelta64]])
