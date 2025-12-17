from typing import Any, Literal, assert_type

import numpy as np
import numpy.typing as npt
from numpy.linalg._linalg import (
    EighResult,
    EigResult,
    QRResult,
    SlogdetResult,
    SVDResult,
)

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]

bool_list_1d: list[bool]
bool_list_2d: list[list[bool]]
int_list_1d: list[int]
int_list_2d: list[list[int]]
float_list_1d: list[float]
float_list_2d: list[list[float]]
float_list_3d: list[list[list[float]]]
float_list_4d: list[list[list[list[float]]]]
complex_list_1d: list[complex]
complex_list_2d: list[list[complex]]
complex_list_3d: list[list[list[complex]]]
bytes_list_2d: list[list[bytes]]
str_list_2d: list[list[str]]

AR_any: np.ndarray
AR_f_: npt.NDArray[np.floating]
AR_c_: npt.NDArray[np.complexfloating]
AR_i8: npt.NDArray[np.int64]
AR_f2: npt.NDArray[np.float16]
AR_f4: npt.NDArray[np.float32]
AR_f8: npt.NDArray[np.float64]
AR_f10: npt.NDArray[np.longdouble]
AR_c8: npt.NDArray[np.complex64]
AR_c16: npt.NDArray[np.complex128]
AR_c20: npt.NDArray[np.clongdouble]
AR_O: npt.NDArray[np.object_]
AR_M: npt.NDArray[np.datetime64]
AR_m: npt.NDArray[np.timedelta64]
AR_S: npt.NDArray[np.bytes_]
AR_U: npt.NDArray[np.str_]
AR_b: npt.NDArray[np.bool]

SC_f8: np.float64
AR_f8_0d: np.ndarray[tuple[()], np.dtype[np.float64]]
AR_f8_1d: _Array1D[np.float64]
AR_f8_2d: _Array2D[np.float64]
AR_f8_3d: _Array3D[np.float64]
AR_f8_4d: np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]

AR_f2_2d: _Array2D[np.float16]
AR_f4_1d: _Array1D[np.float32]
AR_f4_2d: _Array2D[np.float32]
AR_f4_3d: _Array3D[np.float32]
AR_f10_2d: _Array2D[np.longdouble]
AR_f10_3d: _Array3D[np.longdouble]

###

assert_type(np.linalg.tensorsolve(AR_i8, AR_i8), npt.NDArray[np.float64])
assert_type(np.linalg.tensorsolve(AR_i8, AR_f8), npt.NDArray[np.float64])
assert_type(np.linalg.tensorsolve(AR_f4, AR_f4), npt.NDArray[np.float32])
assert_type(np.linalg.tensorsolve(AR_c16, AR_f8), npt.NDArray[np.complex128])
assert_type(np.linalg.tensorsolve(AR_c8, AR_f4), npt.NDArray[np.complex64])
assert_type(np.linalg.tensorsolve(AR_f4, AR_c8), npt.NDArray[np.complex64])

assert_type(np.linalg.solve(AR_i8, AR_i8), npt.NDArray[np.float64])
assert_type(np.linalg.solve(AR_i8, AR_f8), npt.NDArray[np.float64])
assert_type(np.linalg.solve(AR_f4, AR_f4), npt.NDArray[np.float32])
assert_type(np.linalg.solve(AR_c16, AR_f8), npt.NDArray[np.complex128])
assert_type(np.linalg.solve(AR_c8, AR_f4), npt.NDArray[np.complex64])
assert_type(np.linalg.solve(AR_f4, AR_c8), npt.NDArray[np.complex64])

assert_type(np.linalg.tensorinv(AR_i8), npt.NDArray[np.float64])
assert_type(np.linalg.tensorinv(AR_f8), npt.NDArray[np.float64])
assert_type(np.linalg.tensorinv(AR_c16), npt.NDArray[np.complex128])

assert_type(np.linalg.inv(AR_i8), npt.NDArray[np.float64])
assert_type(np.linalg.inv(AR_f8), npt.NDArray[np.float64])
assert_type(np.linalg.inv(AR_c16), npt.NDArray[np.complex128])

assert_type(np.linalg.pinv(AR_i8), npt.NDArray[np.float64])
assert_type(np.linalg.pinv(AR_f8), npt.NDArray[np.float64])
assert_type(np.linalg.pinv(AR_c16), npt.NDArray[np.complex128])

assert_type(np.linalg.matrix_power(AR_i8, -1), npt.NDArray[np.float64])
assert_type(np.linalg.matrix_power(AR_i8, 1), npt.NDArray[np.int64])
assert_type(np.linalg.matrix_power(AR_f8, 0), npt.NDArray[np.float64])
assert_type(np.linalg.matrix_power(AR_c16, 1), npt.NDArray[np.complex128])
assert_type(np.linalg.matrix_power(AR_O, 2), npt.NDArray[np.object_])

assert_type(np.linalg.cholesky(AR_i8), npt.NDArray[np.float64])
assert_type(np.linalg.cholesky(AR_f8), npt.NDArray[np.float64])
assert_type(np.linalg.cholesky(AR_c16), npt.NDArray[np.complex128])

assert_type(np.linalg.qr(AR_i8), QRResult[np.float64])
assert_type(np.linalg.qr(AR_i8, "r"), npt.NDArray[np.float64])
assert_type(np.linalg.qr(AR_i8, "raw"), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(np.linalg.qr(AR_f4), QRResult[np.float32])
assert_type(np.linalg.qr(AR_f4, "r"), npt.NDArray[np.float32])
assert_type(np.linalg.qr(AR_f4, "raw"), tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]])
assert_type(np.linalg.qr(AR_f8), QRResult[np.float64])
assert_type(np.linalg.qr(AR_f8, "r"), npt.NDArray[np.float64])
assert_type(np.linalg.qr(AR_f8, "raw"), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(np.linalg.qr(AR_c8), QRResult[np.complex64])
assert_type(np.linalg.qr(AR_c8, "r"), npt.NDArray[np.complex64])
assert_type(np.linalg.qr(AR_c8, "raw"), tuple[npt.NDArray[np.complex64], npt.NDArray[np.complex64]])
assert_type(np.linalg.qr(AR_c16), QRResult[np.complex128])
assert_type(np.linalg.qr(AR_c16, "r"), npt.NDArray[np.complex128])
assert_type(np.linalg.qr(AR_c16, "raw"), tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]])
# Mypy bug: `Expression is of type "QRResult[Any]", not "QRResult[Any]"`
assert_type(np.linalg.qr(AR_any), QRResult[Any])  # type: ignore[assert-type]
# Mypy bug: `Expression is of type "ndarray[Any, Any]", not "ndarray[tuple[Any, ...], dtype[Any]]"`
assert_type(np.linalg.qr(AR_any, "r"), npt.NDArray[Any])  # type: ignore[assert-type]
# Mypy bug: `Expression is of type "tuple[Any, ...]", <--snip-->"`
assert_type(np.linalg.qr(AR_any, "raw"), tuple[npt.NDArray[Any], npt.NDArray[Any]])  # type: ignore[assert-type]

assert_type(np.linalg.eigvals(AR_i8), npt.NDArray[np.float64] | npt.NDArray[np.complex128])
assert_type(np.linalg.eigvals(AR_f8), npt.NDArray[np.float64] | npt.NDArray[np.complex128])
assert_type(np.linalg.eigvals(AR_c16), npt.NDArray[np.complex128])

assert_type(np.linalg.eigvalsh(AR_i8), npt.NDArray[np.float64])
assert_type(np.linalg.eigvalsh(AR_f8), npt.NDArray[np.float64])
assert_type(np.linalg.eigvalsh(AR_c16), npt.NDArray[np.float64])

assert_type(np.linalg.eig(AR_i8), EigResult[np.float64] | EigResult[np.complex128])
assert_type(np.linalg.eig(AR_f4), EigResult[np.float32] | EigResult[np.complex64])
assert_type(np.linalg.eig(AR_f8), EigResult[np.float64] | EigResult[np.complex128])
assert_type(np.linalg.eig(AR_c8), EigResult[np.complex64])
assert_type(np.linalg.eig(AR_c16), EigResult[np.complex128])
# Mypy bug: `Expression is of type "EigResult[Any]", not "EigResult[Any]"`
assert_type(np.linalg.eig(AR_f_), EigResult[Any])  # type: ignore[assert-type]
assert_type(np.linalg.eig(AR_c_), EigResult[Any])  # type: ignore[assert-type]
assert_type(np.linalg.eig(AR_any), EigResult[Any])  # type: ignore[assert-type]

assert_type(np.linalg.eigh(AR_i8), EighResult[np.float64, np.float64])
assert_type(np.linalg.eigh(AR_f4), EighResult[np.float32, np.float32])
assert_type(np.linalg.eigh(AR_f8), EighResult[np.float64, np.float64])
assert_type(np.linalg.eigh(AR_c8), EighResult[np.float32, np.complex64])
assert_type(np.linalg.eigh(AR_c16), EighResult[np.float64, np.complex128])
# Mypy bug: `Expression is of type "EighResult[Any, Any]", not "EighResult[Any, Any]"`
assert_type(np.linalg.eigh(AR_any), EighResult[Any, Any])  # type: ignore[assert-type]

assert_type(np.linalg.svd(AR_i8), SVDResult[np.float64, np.float64])
assert_type(np.linalg.svd(AR_i8, compute_uv=False), npt.NDArray[np.float64])
assert_type(np.linalg.svd(AR_f4), SVDResult[np.float32, np.float32])
assert_type(np.linalg.svd(AR_f4, compute_uv=False), npt.NDArray[np.float32])
assert_type(np.linalg.svd(AR_f8), SVDResult[np.float64, np.float64])
assert_type(np.linalg.svd(AR_f8, compute_uv=False), npt.NDArray[np.float64])
assert_type(np.linalg.svd(AR_c8), SVDResult[np.float32, np.complex64])
assert_type(np.linalg.svd(AR_c8, compute_uv=False), npt.NDArray[np.float32])
assert_type(np.linalg.svd(AR_c16), SVDResult[np.float64, np.complex128])
assert_type(np.linalg.svd(AR_c16, compute_uv=False), npt.NDArray[np.float64])
assert_type(np.linalg.svd(int_list_2d), SVDResult[np.float64, np.float64])
assert_type(np.linalg.svd(int_list_2d, compute_uv=False), npt.NDArray[np.float64])
assert_type(np.linalg.svd(float_list_2d), SVDResult[np.float64, np.float64])
assert_type(np.linalg.svd(float_list_2d, compute_uv=False), npt.NDArray[np.float64])
assert_type(np.linalg.svd(complex_list_2d), SVDResult[np.float64, np.complex128])
assert_type(np.linalg.svd(complex_list_2d, compute_uv=False), npt.NDArray[np.float64])
# Mypy bug: `Expression is of type "SVDResult[Any, Any]", not "SVDResult[Any, Any]"`
assert_type(np.linalg.svd(AR_any), SVDResult[Any, Any])  # type: ignore[assert-type]
# Mypy bug: `Expression is of type "ndarray[Any, Any]", not "ndarray[tuple[Any, ...], dtype[Any]]"`
assert_type(np.linalg.svd(AR_any, compute_uv=False), npt.NDArray[Any])  # type: ignore[assert-type]

assert_type(np.linalg.svdvals(AR_b), npt.NDArray[np.float64])
assert_type(np.linalg.svdvals(AR_i8), npt.NDArray[np.float64])
assert_type(np.linalg.svdvals(AR_f4), npt.NDArray[np.float32])
assert_type(np.linalg.svdvals(AR_f8), npt.NDArray[np.float64])
assert_type(np.linalg.svdvals(AR_c8), npt.NDArray[np.float32])
assert_type(np.linalg.svdvals(AR_c16), npt.NDArray[np.float64])
assert_type(np.linalg.svdvals(int_list_2d), npt.NDArray[np.float64])
assert_type(np.linalg.svdvals(float_list_2d), npt.NDArray[np.float64])
assert_type(np.linalg.svdvals(complex_list_2d), npt.NDArray[np.float64])

assert_type(np.linalg.matrix_rank(AR_i8), Any)
assert_type(np.linalg.matrix_rank(AR_f8), Any)
assert_type(np.linalg.matrix_rank(AR_c16), Any)
assert_type(np.linalg.matrix_rank(SC_f8), Literal[0, 1])
assert_type(np.linalg.matrix_rank(AR_f8_1d), Literal[0, 1])
assert_type(np.linalg.matrix_rank(float_list_1d), Literal[0, 1])
assert_type(np.linalg.matrix_rank(AR_f8_2d), np.int_)
assert_type(np.linalg.matrix_rank(float_list_2d), np.int_)
assert_type(np.linalg.matrix_rank(AR_f8_3d), _Array1D[np.int_])
assert_type(np.linalg.matrix_rank(float_list_3d), _Array1D[np.int_])
assert_type(np.linalg.matrix_rank(AR_f8_4d), npt.NDArray[np.int_])
assert_type(np.linalg.matrix_rank(float_list_4d), npt.NDArray[np.int_])

assert_type(np.linalg.cond(AR_i8), Any)
assert_type(np.linalg.cond(AR_f8), Any)
assert_type(np.linalg.cond(AR_c16), Any)
assert_type(np.linalg.cond(AR_f4_2d), np.float32)
assert_type(np.linalg.cond(AR_f8_2d), np.float64)
assert_type(np.linalg.cond(AR_f4_3d), npt.NDArray[np.float32])
assert_type(np.linalg.cond(AR_f8_3d), npt.NDArray[np.float64])

assert_type(np.linalg.slogdet(AR_i8), SlogdetResult)
assert_type(np.linalg.slogdet(AR_f8), SlogdetResult)
assert_type(np.linalg.slogdet(AR_c16), SlogdetResult)
assert_type(np.linalg.slogdet(AR_f4_2d), SlogdetResult[np.float32, np.float32])
assert_type(np.linalg.slogdet(AR_f8_2d), SlogdetResult[np.float64, np.float64])
assert_type(np.linalg.slogdet(AR_f4_3d), SlogdetResult[npt.NDArray[np.float32], npt.NDArray[np.float32]])
assert_type(np.linalg.slogdet(AR_f8_3d), SlogdetResult[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(np.linalg.slogdet(complex_list_2d), SlogdetResult[np.float64, np.complex128])
assert_type(np.linalg.slogdet(complex_list_3d), SlogdetResult[npt.NDArray[np.float64], npt.NDArray[np.complex128]])

assert_type(np.linalg.det(AR_i8), Any)
assert_type(np.linalg.det(AR_f8), Any)
assert_type(np.linalg.det(AR_c16), Any)
assert_type(np.linalg.det(AR_f4_2d), np.float32)
assert_type(np.linalg.det(AR_f8_2d), np.float64)
assert_type(np.linalg.det(AR_f4_3d), npt.NDArray[np.float32])
assert_type(np.linalg.det(AR_f8_3d), npt.NDArray[np.float64])
assert_type(np.linalg.det(complex_list_2d), np.complex128)
assert_type(np.linalg.det(complex_list_3d), npt.NDArray[np.complex128])

assert_type(
    np.linalg.lstsq(AR_i8, AR_i8),
    tuple[npt.NDArray[np.float64], _Array1D[np.float64], np.int32, _Array1D[np.float64]],
)
assert_type(
    np.linalg.lstsq(AR_f4, AR_f4),
    tuple[npt.NDArray[np.float32], _Array1D[np.float32], np.int32, _Array1D[np.float32]],
)
assert_type(
    np.linalg.lstsq(AR_i8, AR_f8),
    tuple[npt.NDArray[np.float64], _Array1D[np.float64], np.int32, _Array1D[np.float64]],
)
assert_type(
    np.linalg.lstsq(AR_f4, AR_f8),
    tuple[npt.NDArray[np.float64], _Array1D[np.float64], np.int32, _Array1D[np.float64]],
)
assert_type(
    np.linalg.lstsq(AR_f8, AR_i8),
    tuple[npt.NDArray[np.float64], _Array1D[np.float64], np.int32, _Array1D[np.float64]],
)
assert_type(
    np.linalg.lstsq(AR_f8, AR_f4),
    tuple[npt.NDArray[np.float64], _Array1D[np.float64], np.int32, _Array1D[np.float64]],
)
assert_type(
    np.linalg.lstsq(AR_c8, AR_c8),
    tuple[npt.NDArray[np.complex64], _Array1D[np.float32], np.int32, _Array1D[np.float32]],
)
assert_type(
    np.linalg.lstsq(AR_c8, AR_c16),
    tuple[npt.NDArray[np.complex128], _Array1D[np.float64], np.int32, _Array1D[np.float64]],
)
assert_type(
    np.linalg.lstsq(AR_c16, AR_c8),
    tuple[npt.NDArray[np.complex128], _Array1D[np.float64], np.int32, _Array1D[np.float64]],
)
assert_type(
    np.linalg.lstsq(AR_f8, AR_f8_1d),
    tuple[_Array1D[np.float64], _Array1D[np.float64], np.int32, _Array1D[np.float64]],
)
assert_type(
    np.linalg.lstsq(AR_f4, AR_f4_1d),
    tuple[_Array1D[np.float32], _Array1D[np.float32], np.int32, _Array1D[np.float32]],
)
assert_type(
    np.linalg.lstsq(AR_f8, AR_f8_2d),
    tuple[_Array2D[np.float64], _Array1D[np.float64], np.int32, _Array1D[np.float64]],
)
assert_type(
    np.linalg.lstsq(AR_f4, AR_f4_2d),
    tuple[_Array2D[np.float32], _Array1D[np.float32], np.int32, _Array1D[np.float32]],
)

assert_type(np.linalg.norm(AR_i8), np.float64)
assert_type(np.linalg.norm(AR_f8), np.float64)
assert_type(np.linalg.norm(AR_c16), np.float64)
# Mypy incorrectly infers `Any` for datetime64 and timedelta64, but pyright behaves correctly.
assert_type(np.linalg.norm(AR_M), np.float64)  # type: ignore[assert-type]
assert_type(np.linalg.norm(AR_m), np.float64)  # type: ignore[assert-type]
assert_type(np.linalg.norm(AR_U), np.float64)
assert_type(np.linalg.norm(AR_S), np.float64)
assert_type(np.linalg.norm(AR_f8, 0, 1), npt.NDArray[np.float64])
assert_type(np.linalg.norm(AR_f8, axis=0), npt.NDArray[np.float64])
assert_type(np.linalg.norm(AR_f8, keepdims=True), npt.NDArray[np.float64])
assert_type(np.linalg.norm(AR_f8_2d, keepdims=True), _Array2D[np.float64])
assert_type(np.linalg.norm(AR_f2), np.float16)
assert_type(np.linalg.norm(AR_f2, 0, 1), npt.NDArray[np.float16])
assert_type(np.linalg.norm(AR_f2, axis=1), npt.NDArray[np.float16])
assert_type(np.linalg.norm(AR_f2, keepdims=True), npt.NDArray[np.float16])
assert_type(np.linalg.norm(AR_f2_2d, keepdims=True), _Array2D[np.float16])
assert_type(np.linalg.norm(AR_f4), np.float32)
assert_type(np.linalg.norm(AR_c8), np.float32)
assert_type(np.linalg.norm(AR_f4, 0, 1), npt.NDArray[np.float32])
assert_type(np.linalg.norm(AR_f4, axis=1), npt.NDArray[np.float32])
assert_type(np.linalg.norm(AR_f4, keepdims=True), npt.NDArray[np.float32])
assert_type(np.linalg.norm(AR_f4_2d, keepdims=True), _Array2D[np.float32])
assert_type(np.linalg.norm(AR_f10), np.longdouble)
assert_type(np.linalg.norm(AR_c20), np.longdouble)
assert_type(np.linalg.norm(AR_f10, 0, 1), npt.NDArray[np.longdouble])
assert_type(np.linalg.norm(AR_f10, axis=1), npt.NDArray[np.longdouble])
assert_type(np.linalg.norm(AR_f10, keepdims=True), npt.NDArray[np.longdouble])
assert_type(np.linalg.norm(AR_f10_2d, keepdims=True), _Array2D[np.longdouble])

assert_type(np.linalg.matrix_norm(AR_i8), npt.NDArray[np.float64] | Any)
assert_type(np.linalg.matrix_norm(AR_f8), npt.NDArray[np.float64] | Any)
assert_type(np.linalg.matrix_norm(AR_c16), npt.NDArray[np.float64] | Any)
assert_type(np.linalg.matrix_norm(AR_U), npt.NDArray[np.float64] | Any)
assert_type(np.linalg.matrix_norm(AR_S), npt.NDArray[np.float64] | Any)
assert_type(np.linalg.matrix_norm(AR_f8_2d), np.float64)
assert_type(np.linalg.matrix_norm(AR_f8_3d), npt.NDArray[np.float64])
assert_type(np.linalg.matrix_norm(AR_f8_2d, keepdims=True), _Array2D[np.float64])
assert_type(np.linalg.matrix_norm(AR_f4), npt.NDArray[np.float32] | Any)
assert_type(np.linalg.matrix_norm(AR_c8), npt.NDArray[np.float32] | Any)
assert_type(np.linalg.matrix_norm(AR_f4_2d), np.float32)
assert_type(np.linalg.matrix_norm(AR_f4_3d), npt.NDArray[np.float32])
assert_type(np.linalg.matrix_norm(AR_f4_2d, keepdims=True), _Array2D[np.float32])
assert_type(np.linalg.matrix_norm(AR_f10), npt.NDArray[np.longdouble] | Any)
assert_type(np.linalg.matrix_norm(AR_c20), npt.NDArray[np.longdouble] | Any)
assert_type(np.linalg.matrix_norm(AR_f10_2d), np.longdouble)
assert_type(np.linalg.matrix_norm(AR_f10_3d), npt.NDArray[np.longdouble])
assert_type(np.linalg.matrix_norm(AR_f10_2d, keepdims=True), _Array2D[np.longdouble])
assert_type(np.linalg.matrix_norm(complex_list_2d), np.float64)
assert_type(np.linalg.matrix_norm(complex_list_3d), npt.NDArray[np.float64])
assert_type(np.linalg.matrix_norm(complex_list_2d, keepdims=True), npt.NDArray[np.float64])

assert_type(np.linalg.vector_norm(AR_i8), np.float64)
assert_type(np.linalg.vector_norm(AR_f8), np.float64)
assert_type(np.linalg.vector_norm(AR_c16), np.float64)
# Mypy incorrectly infers `Any` for datetime64 and timedelta64, but pyright behaves correctly.
assert_type(np.linalg.vector_norm(AR_M), np.float64)  # type: ignore[assert-type]
assert_type(np.linalg.vector_norm(AR_m), np.float64)  # type: ignore[assert-type]
assert_type(np.linalg.vector_norm(AR_U), np.float64)
assert_type(np.linalg.vector_norm(AR_S), np.float64)
assert_type(np.linalg.vector_norm(AR_f8, axis=0), npt.NDArray[np.float64])
assert_type(np.linalg.vector_norm(AR_f8, keepdims=True), npt.NDArray[np.float64])
assert_type(np.linalg.vector_norm(AR_f8_2d, keepdims=True), _Array2D[np.float64])
assert_type(np.linalg.vector_norm(AR_f2), np.float16)
assert_type(np.linalg.vector_norm(AR_f2, axis=1), npt.NDArray[np.float16])
assert_type(np.linalg.vector_norm(AR_f2, keepdims=True), npt.NDArray[np.float16])
assert_type(np.linalg.vector_norm(AR_f2_2d, keepdims=True), _Array2D[np.float16])
assert_type(np.linalg.vector_norm(AR_f4), np.float32)
assert_type(np.linalg.vector_norm(AR_c8), np.float32)
assert_type(np.linalg.vector_norm(AR_f4, axis=1), npt.NDArray[np.float32])
assert_type(np.linalg.vector_norm(AR_f4, keepdims=True), npt.NDArray[np.float32])
assert_type(np.linalg.vector_norm(AR_f4_2d, keepdims=True), _Array2D[np.float32])
assert_type(np.linalg.vector_norm(AR_f10), np.longdouble)
assert_type(np.linalg.vector_norm(AR_c20), np.longdouble)
assert_type(np.linalg.vector_norm(AR_f10, axis=1), npt.NDArray[np.longdouble])
assert_type(np.linalg.vector_norm(AR_f10, keepdims=True), npt.NDArray[np.longdouble])
assert_type(np.linalg.vector_norm(AR_f10_2d, keepdims=True), _Array2D[np.longdouble])

assert_type(np.linalg.tensordot(AR_b, AR_b), npt.NDArray[np.bool])
assert_type(np.linalg.tensordot(AR_i8, AR_i8), npt.NDArray[np.int64])
assert_type(np.linalg.tensordot(AR_f8, AR_f8), npt.NDArray[np.float64])
assert_type(np.linalg.tensordot(AR_c16, AR_c16), npt.NDArray[np.complex128])
assert_type(np.linalg.tensordot(AR_m, AR_m), npt.NDArray[np.timedelta64])
assert_type(np.linalg.tensordot(AR_O, AR_O), npt.NDArray[np.object_])

assert_type(np.linalg.multi_dot([AR_i8, AR_i8]), Any)
assert_type(np.linalg.multi_dot([AR_i8, AR_f8]), Any)
assert_type(np.linalg.multi_dot([AR_f8, AR_c16]), Any)
assert_type(np.linalg.multi_dot([AR_O, AR_O]), Any)
assert_type(np.linalg.multi_dot([AR_m, AR_m]), Any)

# Mypy incorrectly infers `ndarray[Any, Any]`, but pyright behaves correctly.
assert_type(np.linalg.diagonal(AR_any), np.ndarray)  # type: ignore[assert-type]
assert_type(np.linalg.diagonal(AR_f4), npt.NDArray[np.float32])
assert_type(np.linalg.diagonal(AR_f4_2d), _Array1D[np.float32])
assert_type(np.linalg.diagonal(AR_f8_2d), _Array1D[np.float64])
assert_type(np.linalg.diagonal(bool_list_2d), npt.NDArray[np.bool])
assert_type(np.linalg.diagonal(int_list_2d), npt.NDArray[np.int_])
assert_type(np.linalg.diagonal(float_list_2d), npt.NDArray[np.float64])
assert_type(np.linalg.diagonal(complex_list_2d), npt.NDArray[np.complex128])
assert_type(np.linalg.diagonal(bytes_list_2d), npt.NDArray[np.bytes_])
assert_type(np.linalg.diagonal(str_list_2d), npt.NDArray[np.str_])

assert_type(np.linalg.trace(AR_any), Any)
assert_type(np.linalg.trace(AR_f4), Any)
assert_type(np.linalg.trace(AR_f4_2d), np.float32)
assert_type(np.linalg.trace(AR_f8_2d), np.float64)
assert_type(np.linalg.trace(AR_f4_3d), _Array1D[np.float32])
assert_type(np.linalg.trace(AR_f8_3d), _Array1D[np.float64])
assert_type(np.linalg.trace(AR_f8_4d), np.ndarray[tuple[int, *tuple[Any, ...]], np.dtype[np.float64]])
assert_type(np.linalg.trace(bool_list_2d), np.bool)
assert_type(np.linalg.trace(int_list_2d), np.int_)
assert_type(np.linalg.trace(float_list_2d), np.float64)
assert_type(np.linalg.trace(complex_list_2d), np.complex128)
assert_type(np.linalg.trace(float_list_3d), npt.NDArray[np.float64])

assert_type(np.linalg.outer(bool_list_1d, bool_list_1d), _Array2D[np.bool])
assert_type(np.linalg.outer(int_list_1d, int_list_1d), _Array2D[np.int64])
assert_type(np.linalg.outer(float_list_1d, float_list_1d), _Array2D[np.float64])
assert_type(np.linalg.outer(complex_list_1d, complex_list_1d), _Array2D[np.complex128])
assert_type(np.linalg.outer(AR_i8, AR_i8), _Array2D[np.int64])
assert_type(np.linalg.outer(AR_f8, AR_f8), _Array2D[np.float64])
assert_type(np.linalg.outer(AR_c16, AR_c16), _Array2D[np.complex128])
assert_type(np.linalg.outer(AR_b, AR_b), _Array2D[np.bool])
assert_type(np.linalg.outer(AR_O, AR_O), _Array2D[np.object_])
assert_type(np.linalg.outer(AR_i8, AR_m), _Array2D[np.timedelta64])

assert_type(np.linalg.cross(AR_i8, AR_i8), npt.NDArray[np.signedinteger])
assert_type(np.linalg.cross(AR_f8, AR_f8), npt.NDArray[np.floating])
assert_type(np.linalg.cross(AR_c16, AR_c16), npt.NDArray[np.complexfloating])

assert_type(np.linalg.matmul(AR_i8, AR_i8), npt.NDArray[np.int64])
assert_type(np.linalg.matmul(AR_f8, AR_f8), npt.NDArray[np.float64])
assert_type(np.linalg.matmul(AR_c16, AR_c16), npt.NDArray[np.complex128])
