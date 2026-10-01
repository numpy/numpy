from collections.abc import Iterator
from fractions import Fraction
from typing import Any, NoReturn, assert_type

import numpy as np
import numpy.typing as npt

AR_b: npt.NDArray[np.bool]
AR_u4: npt.NDArray[np.uint32]
AR_i8: npt.NDArray[np.int64]
AR_f4: npt.NDArray[np.float32]
AR_f8: npt.NDArray[np.float64]
AR_f8_1d: npt.Array1D[np.float64]
AR_f8_2d: npt.Array2D[np.float64]
AR_c8: npt.NDArray[np.complex64]
AR_c16: npt.NDArray[np.complex128]
AR_c16_1d: npt.Array1D[np.complex128]
AR_O: npt.NDArray[np.object_]

_f8: np.float64

_py_b_1d: list[bool]
_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]
_py_f_2d: list[list[float]]

_poly: np.poly1d
_poly_b: np.poly1d[np.bool]
_poly_f4: np.poly1d[np.float32]
_poly_f8: np.poly1d[np.float64]
_poly_c8: np.poly1d[np.complex64]
_poly_c16: np.poly1d[np.complex128]
_poly_O: np.poly1d[np.object_[Fraction]]

assert_type(np.poly1d(AR_f4), np.poly1d[np.float32])
assert_type(np.poly1d(_py_b_1d), np.poly1d[np.bool])
assert_type(np.poly1d(_py_i_1d), np.poly1d[np.int_])
assert_type(np.poly1d(_py_f_1d), np.poly1d[np.float64])
assert_type(np.poly1d(_py_c_1d), np.poly1d[np.complex128])
assert_type(np.poly1d(_py_f_1d, r=True), np.poly1d[Any])

assert_type(_poly.variable, str)
assert_type(_poly.order, int)
assert_type(_poly.o, int)
assert_type(_poly.roots, npt.Array1D[Any])
assert_type(_poly.r, npt.Array1D[Any])
assert_type(_poly_f8.coeffs, npt.Array1D[np.float64])
assert_type(_poly_f8.c, npt.Array1D[np.float64])
assert_type(_poly_f8.coef, npt.Array1D[np.float64])
assert_type(_poly_f8.coefficients, npt.Array1D[np.float64])
assert_type(_poly.__hash__, None)

assert_type(_poly_f8.__array__(), npt.Array1D[np.float64])

assert_type(_poly_f8(_poly_f8), np.poly1d[np.float64])
assert_type(_poly_f8(AR_f8_2d), npt.Array2D[np.float64])
assert_type(_poly_f8(_f8), np.float64)
assert_type(_poly_f8([_f8]), npt.Array1D[np.float64])
assert_type(_poly_f8([[_f8]]), npt.Array2D[np.float64])
assert_type(_poly(_f8), Any)
assert_type(_poly(AR_f8_2d), npt.Array2D[Any])
assert_type(_poly_f8(1.0), np.float64)
assert_type(_poly_f8(_py_f_1d), npt.Array1D[np.float64])
assert_type(_poly_f8(_py_f_2d), npt.Array2D[np.float64])
assert_type(_poly_c16(1j), np.complex128)
assert_type(_poly_c16(_py_c_1d), npt.Array1D[np.complex128])
assert_type(_poly_c16(_py_f_2d), npt.Array2D[np.complex128])
assert_type(_poly_f8(_poly_f4), np.poly1d[Any])
assert_type(_poly_b(_poly_b), np.poly1d[Any])
assert_type(_poly_f4(AR_f8_2d), npt.Array2D[Any])
assert_type(_poly_f4(1.0), Any)
assert_type(_poly_f4(_py_f_1d), npt.Array1D[Any])
assert_type(_poly_f4(_py_f_2d), npt.Array2D[Any])
assert_type(_poly_f4([AR_f8]), npt.NDArray[Any])

assert_type(len(_poly), int)

assert_type(iter(_poly_O), Iterator[Fraction])
assert_type(iter(_poly_f8), Iterator[np.float64])

assert_type(_poly_O[0], Fraction)
assert_type(_poly_f8[0], np.float64)
_poly[0] = 5

assert_type(-_poly_f8, np.poly1d[np.float64])
assert_type(-_poly_O, np.poly1d[np.object_[Fraction]])
assert_type(+_poly, np.poly1d[Any])

assert_type(_poly_f8 + _poly_f8, np.poly1d[np.float64])
assert_type(_poly_f8 + 5, np.poly1d[np.float64])
assert_type(_poly_c16 + 5j, np.poly1d[np.complex128])
assert_type(_poly_f4 + 5, np.poly1d[Any])

assert_type([_f8] + _poly_f8, np.poly1d[np.float64])
assert_type(5 + _poly_f8, np.poly1d[np.float64])
assert_type(5j + _poly_c16, np.poly1d[np.complex128])
assert_type(5 + _poly_f4, np.poly1d[Any])

assert_type(_poly_f8 - _poly_f8, np.poly1d[np.float64])
assert_type(_poly_f8 - 5, np.poly1d[np.float64])
assert_type(_poly_c16 - 5j, np.poly1d[np.complex128])
assert_type(_poly_f4 - 5, np.poly1d[Any])
assert_type(_poly_b - _poly_b, np.poly1d[Any])

assert_type([_f8] - _poly_f8, np.poly1d[np.float64])
assert_type(5 - _poly_f8, np.poly1d[np.float64])
assert_type(5j - _poly_c16, np.poly1d[np.complex128])
assert_type(5 - _poly_f4, np.poly1d[Any])

assert_type(_poly_f8 * _poly_f8, np.poly1d[np.float64])
assert_type(_poly_f4 * _f8, np.poly1d[Any])
assert_type(_poly_f4 * 5, np.poly1d[np.float32])
assert_type(_poly_f4 * 5.0, np.poly1d[np.float32])
assert_type(_poly_c16 * 5j, np.poly1d[np.complex128])
assert_type(_poly_f8 * _py_f_1d, np.poly1d[np.float64])
assert_type(_poly_c16 * _py_c_1d, np.poly1d[np.complex128])
assert_type(_poly_f4 * _py_f_1d, np.poly1d[Any])

assert_type(_poly_f8.__rmul__(_f8), np.poly1d[np.float64])
assert_type(_poly_f4.__rmul__(_f8), np.poly1d[Any])
assert_type(5 * _poly_f4, np.poly1d[np.float32])
assert_type(5.0 * _poly_f4, np.poly1d[np.float32])
assert_type(5j * _poly_c16, np.poly1d[np.complex128])
assert_type(_py_f_1d * _poly_f8, np.poly1d[np.float64])
assert_type(_py_c_1d * _poly_c16, np.poly1d[np.complex128])
assert_type(_py_f_1d * _poly_f4, np.poly1d[Any])

assert_type(_poly_f8**2, np.poly1d[np.float64])
assert_type(_poly_b**2, np.poly1d[np.int_])
assert_type(_poly_f4**2, np.poly1d[np.float64])  # type: ignore[assert-type]
assert_type(_poly_c8**2, np.poly1d[np.complex128])  # type: ignore[assert-type]
assert_type(_poly_O**2, np.poly1d[Any])

assert_type(_poly_f8 / _f8, np.poly1d[np.float64])
assert_type(_poly_f4 / _f8, np.poly1d[Any])
assert_type(_poly_f4 / 5.0, np.poly1d[np.float32])
assert_type(_poly_b / 5, np.poly1d[np.float64])
assert_type(_poly_c8 / 5j, np.poly1d[np.complex64])
assert_type(_poly_f8 / 5j, np.poly1d[Any])
assert_type(_poly_f8 / _poly_f8, tuple[np.poly1d[np.float64], np.poly1d[np.float64]])
assert_type(_poly_f4 / _py_f_1d, tuple[np.poly1d[np.float64], np.poly1d[np.float64]])
assert_type(_poly_c8 / _py_f_1d, tuple[np.poly1d[np.complex128], np.poly1d[np.complex128]])
assert_type(_poly_f4 / AR_f8, tuple[np.poly1d[Any], np.poly1d[Any]])

assert_type(_poly_f8.__rtruediv__(_f8), np.poly1d[np.float64])
assert_type(_poly_f4.__rtruediv__(_f8), np.poly1d[Any])
assert_type(5.0 / _poly_f4, np.poly1d[np.float32])
assert_type(5 / _poly_b, np.poly1d[np.float64])
assert_type(5j / _poly_c8, np.poly1d[np.complex64])
assert_type(5j / _poly_f8, np.poly1d[Any])
assert_type(_poly_f8.__rtruediv__(_poly_f8), tuple[np.poly1d[np.float64], np.poly1d[np.float64]])
assert_type(_py_f_1d / _poly_f4, tuple[np.poly1d[np.float64], np.poly1d[np.float64]])
assert_type(_py_f_1d / _poly_c8, tuple[np.poly1d[np.complex128], np.poly1d[np.complex128]])
assert_type(_poly_f4.__rtruediv__(AR_f8), tuple[np.poly1d[Any], np.poly1d[Any]])

assert_type(_poly_f8.deriv(), np.poly1d[np.float64])
assert_type(_poly_b.deriv(), np.poly1d[np.int_])
assert_type(_poly_f4.deriv(), np.poly1d[np.float64])  # type: ignore[assert-type]
assert_type(_poly_c8.deriv(), np.poly1d[np.complex128])  # type: ignore[assert-type]
assert_type(_poly_O.deriv(), np.poly1d[Any])

assert_type(_poly.integ(), np.poly1d[Any])

assert_type(np.poly(_poly_f8), npt.Array1D[Any])
assert_type(np.poly(AR_f4), npt.Array1D[np.float32])
assert_type(np.poly(AR_c8), npt.Array1D[np.float32 | np.complex64])
assert_type(np.poly(AR_f8_2d), npt.Array1D[np.float64])
assert_type(np.poly(AR_c16), npt.Array1D[np.float64 | np.complex128])
assert_type(np.poly(AR_O), npt.Array1D[np.object_])
assert_type(np.poly(_py_f_2d), npt.Array1D[Any])

assert_type(np.roots(_poly), npt.Array1D[Any])
assert_type(np.roots(AR_c8), npt.Array1D[np.complex64])
assert_type(np.roots(AR_c16), npt.Array1D[np.complex128])
assert_type(np.roots(AR_f4), npt.Array1D[np.float32 | np.complex64])
assert_type(np.roots([1, 2.0]), npt.Array1D[np.float64 | np.complex128])
assert_type(np.roots([1j, 2]), npt.Array1D[np.complex128])
assert_type(np.roots(AR_f8_2d), npt.Array1D[Any])

assert_type(np.polyint(_poly_f8), np.poly1d[Any])
assert_type(np.polyint(AR_i8), npt.Array1D[np.float64])
assert_type(np.polyint(AR_f8, 3, k=_py_i_1d), npt.Array1D[np.float64])
assert_type(np.polyint(AR_c8), npt.Array1D[np.complex128])
assert_type(np.polyint(AR_f8, k=AR_c16), npt.Array1D[Any])

assert_type(np.polyder(_poly_f8), np.poly1d[Any])
assert_type(np.polyder(AR_i8), npt.Array1D[np.int_])
assert_type(np.polyder(AR_f4), npt.Array1D[np.float64])
assert_type(np.polyder(AR_f8, m=2), npt.Array1D[np.float64])
assert_type(np.polyder(AR_c8), npt.Array1D[np.complex128])
assert_type(np.polyder(AR_f8_2d), npt.Array1D[Any])

assert_type(np.polyfit(AR_f8, AR_f8, 2), npt.NDArray[np.float64])
assert_type(np.polyfit(AR_f8, AR_f8_1d, 2), npt.Array1D[np.float64])
assert_type(np.polyfit(AR_f8, AR_f8_2d, 2), npt.NDArray[np.float64])
assert_type(np.polyfit(AR_u4, AR_f8, 1.0, cov="unscaled"), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(np.polyfit(AR_f8, AR_f8_1d, 1, cov=True), tuple[npt.Array1D[np.float64], npt.Array2D[np.float64]])
assert_type(np.polyfit(AR_f8, AR_f8_2d, 1, cov=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(
    np.polyfit(AR_f8, AR_i8, 1, None, True),
    tuple[
        npt.NDArray[np.float64],
        npt.Array1D[np.float64],
        np.int32,
        npt.Array1D[np.float64],
        float | np.floating,
    ],
)
assert_type(
    np.polyfit(AR_f4, AR_i8, 1, full=True),
    tuple[
        npt.NDArray[np.float64],
        npt.Array1D[np.float64],
        np.int32,
        npt.Array1D[np.float64],
        float | np.floating,
    ],
)
assert_type(np.polyfit(AR_c16, AR_f8, 2), npt.NDArray[np.complex128 | Any])
assert_type(np.polyfit(AR_f8, AR_c16_1d, 2), npt.Array1D[np.complex128 | Any])
assert_type(np.polyfit(AR_c16, AR_f8_2d, 2), npt.NDArray[np.complex128 | Any])
assert_type(np.polyfit(AR_u4, AR_c16, 1.0, cov=True), tuple[npt.NDArray[np.complex128 | Any], npt.NDArray[Any]])
assert_type(np.polyfit(AR_c16, AR_c16_1d, 1, cov=True), tuple[npt.Array1D[np.complex128 | Any], npt.Array2D[Any]])
assert_type(np.polyfit(AR_c16, AR_f8_2d, 1, cov=True), tuple[npt.NDArray[np.complex128 | Any], npt.NDArray[Any]])
assert_type(
    np.polyfit(AR_f8, AR_c16, 1, None, True),
    tuple[
        npt.NDArray[np.complex128 | Any],
        npt.Array1D[np.float64],
        np.int32,
        npt.Array1D[np.float64],
        float | np.floating,
    ],
)
assert_type(
    np.polyfit(AR_f8, AR_c16, 1, full=True),
    tuple[
        npt.NDArray[np.complex128 | Any],
        npt.Array1D[np.float64],
        np.int32,
        npt.Array1D[np.float64],
        float | np.floating,
    ],
)

assert_type(np.polyval(AR_i8, 1), np.int_ | Any)
assert_type(np.polyval(AR_i8, AR_b), npt.NDArray[np.int_ | Any])
assert_type(np.polyval(AR_i8, _py_c_1d), npt.NDArray[np.complex128 | Any])
assert_type(np.polyval(AR_f8, _poly_f8), np.poly1d[Any])
assert_type(np.polyval(AR_f8, _f8), np.float64)
assert_type(np.polyval(AR_f8, 1.0), np.float64 | Any)
assert_type(np.polyval(AR_f8, AR_i8), npt.NDArray[np.float64 | Any])
assert_type(np.polyval(AR_f8, AR_O), npt.NDArray[Any] | Any)
assert_type(np.polyval(AR_f8_1d, AR_f8_2d), npt.Array2D[np.float64])
assert_type(np.polyval(AR_c16, 1j), np.complex128 | Any)
assert_type(np.polyval(AR_c16_1d, AR_f8_2d), npt.Array2D[np.complex128 | Any])
assert_type(np.polyval(_py_i_1d, _py_i_1d), npt.NDArray[np.int_ | Any])
assert_type(np.polyval(_py_f_1d, _py_i_1d), npt.NDArray[np.float64 | Any])

assert_type(np.polyadd(_poly_f8, AR_i8), np.poly1d[Any])
assert_type(np.polyadd(AR_f8, _poly_f8), np.poly1d[Any])
assert_type(np.polyadd(AR_f8_1d, AR_f8_1d), npt.Array1D[np.float64])
assert_type(np.polyadd(_py_b_1d, _py_b_1d), npt.Array1D[np.bool])
assert_type(np.polyadd(_py_i_1d, _py_i_1d), npt.Array1D[np.int_ | Any])
assert_type(np.polyadd(_py_f_1d, _py_f_1d), npt.Array1D[np.float64 | Any])
assert_type(np.polyadd(_py_c_1d, _py_c_1d), npt.Array1D[np.complex128 | Any])
assert_type(np.polyadd(AR_O, AR_f8), npt.Array1D[np.object_])
assert_type(np.polyadd(AR_f8, AR_O), npt.Array1D[np.object_])

assert_type(np.polysub(_poly_f8, AR_i8), np.poly1d[Any])
assert_type(np.polysub(AR_f8, _poly_f8), np.poly1d[Any])
assert_type(np.polysub(AR_f8_1d, AR_f8_1d), npt.Array1D[np.float64])

def test_invalid_polysub() -> None:
    assert_type(np.polysub(_py_b_1d, _py_b_1d), NoReturn)

assert_type(np.polysub(_py_i_1d, _py_i_1d), npt.Array1D[np.int_ | Any])
assert_type(np.polysub(_py_f_1d, _py_f_1d), npt.Array1D[np.float64 | Any])
assert_type(np.polysub(_py_c_1d, _py_c_1d), npt.Array1D[np.complex128 | Any])
assert_type(np.polysub(AR_O, AR_f8), npt.Array1D[np.object_])
assert_type(np.polysub(AR_f8, AR_O), npt.Array1D[np.object_])

assert_type(np.polymul(_poly_f8, AR_i8), np.poly1d[Any])
assert_type(np.polymul(AR_f8, _poly_f8), np.poly1d[Any])
assert_type(np.polymul(AR_f8_1d, AR_f8_1d), npt.Array1D[np.float64])
assert_type(np.polymul(AR_f8, AR_O), npt.Array1D[np.object_])
assert_type(np.polymul(AR_O, AR_f8), npt.Array1D[np.object_])
assert_type(np.polymul(_py_b_1d, _py_b_1d), npt.Array1D[np.bool])
assert_type(np.polymul(_py_i_1d, _py_i_1d), npt.Array1D[np.int_ | Any])
assert_type(np.polymul(_py_f_1d, _py_f_1d), npt.Array1D[np.float64 | Any])
assert_type(np.polymul(_py_c_1d, _py_c_1d), npt.Array1D[np.complex128 | Any])

assert_type(np.polydiv(_poly_f8, _poly_f8), tuple[np.poly1d[Any], np.poly1d[Any]])
assert_type(np.polydiv(_poly_f8, AR_i8), tuple[np.poly1d[Any], np.poly1d[Any]])
assert_type(np.polydiv(AR_f8, _poly_f8), tuple[np.poly1d[Any], np.poly1d[Any]])
assert_type(np.polydiv(_poly_f8, AR_O), tuple[np.poly1d[Any], np.poly1d[Any]])
assert_type(np.polydiv(AR_O, _poly_f8), tuple[np.poly1d[Any], np.poly1d[Any]])
assert_type(np.polydiv(AR_f4, AR_f4), tuple[npt.Array1D[np.float32], npt.Array1D[np.float32]])
assert_type(np.polydiv(AR_i8, AR_i8), tuple[npt.Array1D[np.float64 | Any], npt.Array1D[np.float64 | Any]])
assert_type(np.polydiv(AR_f8, AR_i8), tuple[npt.Array1D[np.float64 | Any], npt.Array1D[np.float64 | Any]])
assert_type(np.polydiv(AR_i8, AR_c16), tuple[npt.Array1D[np.complex128 | Any], npt.Array1D[np.complex128 | Any]])
