from typing import Any, Literal, assert_type

import numpy as np
import numpy.typing as npt

i4: np.int32
f8: np.float64
m8_ns: np.timedelta64[int]
M8_ns: np.datetime64[int]

AR_i8: npt.NDArray[np.int64]
AR_i4: npt.NDArray[np.int32]
AR_f2: npt.NDArray[np.float16]
AR_f8: npt.NDArray[np.float64]
AR_f16: npt.NDArray[np.longdouble]
AR_c8: npt.NDArray[np.complex64]
AR_c16: npt.NDArray[np.complex128]

AR_f8_1d: np.ndarray[tuple[int], np.dtype[np.float64]]
AR_f8_2d: np.ndarray[tuple[int, int], np.dtype[np.float64]]
AR_c16_1d: np.ndarray[tuple[int], np.dtype[np.complex128]]
AR_c16_2d: np.ndarray[tuple[int, int], np.dtype[np.complex128]]

AR_LIKE_b: list[bool]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_c: list[complex]

class ComplexObj:
    real: slice
    imag: slice

assert_type(np.mintypecode(["f8"], typeset="qfQF"), str)

assert_type(np.real(ComplexObj()), slice)
assert_type(np.real(AR_f8), npt.NDArray[np.float64])
assert_type(np.real(AR_c16), npt.NDArray[np.float64])
assert_type(np.real(AR_LIKE_f), npt.NDArray[Any])

assert_type(np.imag(ComplexObj()), slice)
assert_type(np.imag(AR_f8), npt.NDArray[np.float64])
assert_type(np.imag(AR_c16), npt.NDArray[np.float64])
assert_type(np.imag(AR_LIKE_f), npt.NDArray[Any])

assert_type(np.iscomplex(f8), np.bool)
assert_type(np.iscomplex(AR_f8), npt.NDArray[np.bool])
assert_type(np.iscomplex(AR_LIKE_f), npt.NDArray[np.bool])

assert_type(np.isreal(f8), np.bool)
assert_type(np.isreal(AR_f8), npt.NDArray[np.bool])
assert_type(np.isreal(AR_LIKE_f), npt.NDArray[np.bool])

assert_type(np.iscomplexobj(f8), bool)
assert_type(np.isrealobj(f8), bool)

assert_type(np.nan_to_num(True), np.bool)
assert_type(np.nan_to_num(0), np.int_ | Any)
assert_type(np.nan_to_num(0.0), np.float64 | Any)
assert_type(np.nan_to_num(0j), np.complex128 | Any)
assert_type(np.nan_to_num(i4), np.int32)
assert_type(np.nan_to_num(f8), np.float64)
assert_type(np.nan_to_num(m8_ns), np.timedelta64[int])
assert_type(np.nan_to_num(M8_ns), np.datetime64[int])
assert_type(np.nan_to_num(AR_LIKE_b), npt.NDArray[np.bool])
assert_type(np.nan_to_num(AR_LIKE_i), npt.NDArray[np.int_])
assert_type(np.nan_to_num(AR_LIKE_f), npt.NDArray[np.float64])
assert_type(np.nan_to_num(AR_LIKE_c), npt.NDArray[np.complex128])
assert_type(np.nan_to_num(AR_f8), npt.NDArray[np.float64])
assert_type(np.nan_to_num(AR_c16), npt.NDArray[np.complex128])
assert_type(np.nan_to_num(AR_f8_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.nan_to_num(AR_f8_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.nan_to_num(AR_c16_1d), np.ndarray[tuple[int], np.dtype[np.complex128]])
assert_type(np.nan_to_num(AR_c16_2d), np.ndarray[tuple[int, int], np.dtype[np.complex128]])

assert_type(np.real_if_close(AR_LIKE_f), npt.NDArray[Any])
assert_type(np.real_if_close(AR_f8), npt.NDArray[np.float64])
assert_type(np.real_if_close(AR_c8), npt.NDArray[np.float32 | np.complex64])
assert_type(np.real_if_close(AR_c16), npt.NDArray[np.float64 | np.complex128])
assert_type(np.real_if_close(AR_f8_1d), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.real_if_close(AR_f8_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.real_if_close(AR_c16_1d), np.ndarray[tuple[int], np.dtype[np.float64 | np.complex128]])
assert_type(np.real_if_close(AR_c16_2d), np.ndarray[tuple[int, int], np.dtype[np.float64 | np.complex128]])

assert_type(np.typename("h"), Literal["short"])  # type: ignore[deprecated]
assert_type(np.typename("B"), Literal["unsigned char"])  # type: ignore[deprecated]
assert_type(np.typename("V"), Literal["void"])  # type: ignore[deprecated]
assert_type(np.typename("S1"), Literal["character"])  # type: ignore[deprecated]

assert_type(np.common_type(AR_i4), type[np.float64])
assert_type(np.common_type(AR_f2), type[np.float16])
assert_type(np.common_type(AR_f2, AR_i4), type[np.float64])
assert_type(np.common_type(AR_f16, AR_i4), type[np.longdouble])
assert_type(np.common_type(AR_c8, AR_f2), type[np.complex64])
assert_type(np.common_type(AR_f2, AR_c8, AR_i4), type[np.complexfloating])
