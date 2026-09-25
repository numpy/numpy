from collections.abc import Iterator
from typing import Any, NoReturn, assert_type

import numpy as np
import numpy.typing as npt

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

AR_b: npt.NDArray[np.bool]
AR_u4: npt.NDArray[np.uint32]
AR_i8: npt.NDArray[np.int64]
AR_f4: npt.NDArray[np.float32]
AR_f8: npt.NDArray[np.float64]
AR_f8_1d: _Array1D[np.float64]
AR_f8_2d: _Array2D[np.float64]
AR_c8: npt.NDArray[np.complex64]
AR_c16: npt.NDArray[np.complex128]
AR_c16_1d: _Array1D[np.complex128]
AR_O: npt.NDArray[np.object_]

_f8: np.float64

_py_b_1d: list[bool]
_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]
_py_f_2d: list[list[float]]

poly_obj: np.poly1d

assert_type(poly_obj.variable, str)
assert_type(poly_obj.order, int)
assert_type(poly_obj.o, int)
assert_type(poly_obj.roots, _Array1D[Any])
assert_type(poly_obj.r, _Array1D[Any])
assert_type(poly_obj.coeffs, _Array1D[Any])
assert_type(poly_obj.c, _Array1D[Any])
assert_type(poly_obj.coef, _Array1D[Any])
assert_type(poly_obj.coefficients, _Array1D[Any])
assert_type(poly_obj.__hash__, None)

assert_type(poly_obj(poly_obj), np.poly1d)
assert_type(poly_obj(AR_f8_2d), _Array2D[Any])
assert_type(poly_obj(1), Any)
assert_type(poly_obj([1]), _Array1D[Any])
assert_type(poly_obj([[1.0]]), _Array2D[Any])
assert_type(poly_obj([AR_f8]), npt.NDArray[Any])

assert_type(len(poly_obj), int)
assert_type(-poly_obj, np.poly1d)
assert_type(+poly_obj, np.poly1d)

assert_type(poly_obj * 5, np.poly1d)
assert_type(5 * poly_obj, np.poly1d)
assert_type(poly_obj + 5, np.poly1d)
assert_type(5 + poly_obj, np.poly1d)
assert_type(poly_obj - 5, np.poly1d)
assert_type(5 - poly_obj, np.poly1d)
assert_type(poly_obj**1, np.poly1d)
assert_type(poly_obj**1.0, np.poly1d)
assert_type(poly_obj / 5, np.poly1d)
assert_type(5 / poly_obj, np.poly1d)

assert_type(poly_obj[0], Any)
poly_obj[0] = 5
assert_type(iter(poly_obj), Iterator[Any])
assert_type(poly_obj.deriv(), np.poly1d)
assert_type(poly_obj.integ(), np.poly1d)

assert_type(np.poly(poly_obj), _Array1D[Any])
assert_type(np.poly(AR_f4), _Array1D[np.float32])
assert_type(np.poly(AR_c8), _Array1D[np.float32 | np.complex64])
assert_type(np.poly(AR_f8_2d), _Array1D[np.float64])
assert_type(np.poly(AR_c16), _Array1D[np.float64 | np.complex128])
assert_type(np.poly(AR_O), _Array1D[np.object_])
assert_type(np.poly(_py_f_2d), _Array1D[Any])

assert_type(np.roots(poly_obj), _Array1D[Any])
assert_type(np.roots(AR_c8), _Array1D[np.complex64])
assert_type(np.roots(AR_c16), _Array1D[np.complex128])
assert_type(np.roots(AR_f4), _Array1D[np.float32 | np.complex64])
assert_type(np.roots([1, 2.0]), _Array1D[np.float64 | np.complex128])
assert_type(np.roots([1j, 2]), _Array1D[np.complex128])
assert_type(np.roots(AR_f8_2d), _Array1D[Any])

assert_type(np.polyint(poly_obj), np.poly1d)
assert_type(np.polyint(AR_i8), _Array1D[np.float64])
assert_type(np.polyint(AR_f8, 3, k=_py_i_1d), _Array1D[np.float64])
assert_type(np.polyint(AR_c8), _Array1D[np.complex128])
assert_type(np.polyint(AR_f8, k=AR_c16), _Array1D[Any])

assert_type(np.polyder(poly_obj), np.poly1d)
assert_type(np.polyder(AR_i8), _Array1D[np.int_])
assert_type(np.polyder(AR_f4), _Array1D[np.float64])
assert_type(np.polyder(AR_f8, m=2), _Array1D[np.float64])
assert_type(np.polyder(AR_c8), _Array1D[np.complex128])
assert_type(np.polyder(AR_f8_2d), _Array1D[Any])

assert_type(np.polyfit(AR_f8, AR_f8, 2), npt.NDArray[np.float64])
assert_type(np.polyfit(AR_f8, AR_f8_1d, 2), _Array1D[np.float64])
assert_type(np.polyfit(AR_f8, AR_f8_2d, 2), npt.NDArray[np.float64])
assert_type(np.polyfit(AR_u4, AR_f8, 1.0, cov="unscaled"), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(np.polyfit(AR_f8, AR_f8_1d, 1, cov=True), tuple[_Array1D[np.float64], _Array2D[np.float64]])
assert_type(np.polyfit(AR_f8, AR_f8_2d, 1, cov=True), tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]])
assert_type(
    np.polyfit(AR_f8, AR_i8, 1, None, True),
    tuple[
        npt.NDArray[np.float64],
        _Array1D[np.float64],
        np.int32,
        _Array1D[np.float64],
        float | np.floating,
    ],
)
assert_type(
    np.polyfit(AR_f4, AR_i8, 1, full=True),
    tuple[
        npt.NDArray[np.float64],
        _Array1D[np.float64],
        np.int32,
        _Array1D[np.float64],
        float | np.floating,
    ],
)
assert_type(np.polyfit(AR_c16, AR_f8, 2), npt.NDArray[np.complex128 | Any])
assert_type(np.polyfit(AR_f8, AR_c16_1d, 2), _Array1D[np.complex128 | Any])
assert_type(np.polyfit(AR_c16, AR_f8_2d, 2), npt.NDArray[np.complex128 | Any])
assert_type(np.polyfit(AR_u4, AR_c16, 1.0, cov=True), tuple[npt.NDArray[np.complex128 | Any], npt.NDArray[Any]])
assert_type(np.polyfit(AR_c16, AR_c16_1d, 1, cov=True), tuple[_Array1D[np.complex128 | Any], _Array2D[Any]])
assert_type(np.polyfit(AR_c16, AR_f8_2d, 1, cov=True), tuple[npt.NDArray[np.complex128 | Any], npt.NDArray[Any]])
assert_type(
    np.polyfit(AR_f8, AR_c16, 1, None, True),
    tuple[
        npt.NDArray[np.complex128 | Any],
        _Array1D[np.float64],
        np.int32,
        _Array1D[np.float64],
        float | np.floating,
    ],
)
assert_type(
    np.polyfit(AR_f8, AR_c16, 1, full=True),
    tuple[
        npt.NDArray[np.complex128 | Any],
        _Array1D[np.float64],
        np.int32,
        _Array1D[np.float64],
        float | np.floating,
    ],
)

assert_type(np.polyval(AR_i8, 1), np.int_ | Any)
assert_type(np.polyval(AR_i8, AR_b), npt.NDArray[np.int_ | Any])
assert_type(np.polyval(AR_i8, _py_c_1d), npt.NDArray[np.complex128 | Any])
assert_type(np.polyval(AR_f8, poly_obj), np.poly1d)
assert_type(np.polyval(AR_f8, _f8), np.float64)
assert_type(np.polyval(AR_f8, 1.0), np.float64 | Any)
assert_type(np.polyval(AR_f8, AR_i8), npt.NDArray[np.float64 | Any])
assert_type(np.polyval(AR_f8, AR_O), npt.NDArray[Any] | Any)
assert_type(np.polyval(AR_f8_1d, AR_f8_2d), _Array2D[np.float64])
assert_type(np.polyval(AR_c16, 1j), np.complex128 | Any)
assert_type(np.polyval(AR_c16_1d, AR_f8_2d), _Array2D[np.complex128 | Any])
assert_type(np.polyval(_py_i_1d, _py_i_1d), npt.NDArray[np.int_ | Any])
assert_type(np.polyval(_py_f_1d, _py_i_1d), npt.NDArray[np.float64 | Any])

assert_type(np.polyadd(poly_obj, AR_i8), np.poly1d)
assert_type(np.polyadd(AR_f8, poly_obj), np.poly1d)
assert_type(np.polyadd(AR_f8_1d, AR_f8_1d), _Array1D[np.float64])
assert_type(np.polyadd(_py_b_1d, _py_b_1d), _Array1D[np.bool])
assert_type(np.polyadd(_py_i_1d, _py_i_1d), _Array1D[np.int_ | Any])
assert_type(np.polyadd(_py_f_1d, _py_f_1d), _Array1D[np.float64 | Any])
assert_type(np.polyadd(_py_c_1d, _py_c_1d), _Array1D[np.complex128 | Any])
assert_type(np.polyadd(AR_O, AR_f8), _Array1D[np.object_])
assert_type(np.polyadd(AR_f8, AR_O), _Array1D[np.object_])

assert_type(np.polysub(poly_obj, AR_i8), np.poly1d)
assert_type(np.polysub(AR_f8, poly_obj), np.poly1d)
assert_type(np.polysub(AR_f8_1d, AR_f8_1d), _Array1D[np.float64])

def test_invalid_polysub() -> None:
    assert_type(np.polysub(_py_b_1d, _py_b_1d), NoReturn)

assert_type(np.polysub(_py_i_1d, _py_i_1d), _Array1D[np.int_ | Any])
assert_type(np.polysub(_py_f_1d, _py_f_1d), _Array1D[np.float64 | Any])
assert_type(np.polysub(_py_c_1d, _py_c_1d), _Array1D[np.complex128 | Any])
assert_type(np.polysub(AR_O, AR_f8), _Array1D[np.object_])
assert_type(np.polysub(AR_f8, AR_O), _Array1D[np.object_])

assert_type(np.polymul(poly_obj, AR_i8), np.poly1d)
assert_type(np.polymul(AR_f8, poly_obj), np.poly1d)
assert_type(np.polymul(AR_f8_1d, AR_f8_1d), _Array1D[np.float64])
assert_type(np.polymul(AR_f8, AR_O), _Array1D[np.object_])
assert_type(np.polymul(AR_O, AR_f8), _Array1D[np.object_])
assert_type(np.polymul(_py_b_1d, _py_b_1d), _Array1D[np.bool])
assert_type(np.polymul(_py_i_1d, _py_i_1d), _Array1D[np.int_ | Any])
assert_type(np.polymul(_py_f_1d, _py_f_1d), _Array1D[np.float64 | Any])
assert_type(np.polymul(_py_c_1d, _py_c_1d), _Array1D[np.complex128 | Any])

assert_type(np.polydiv(poly_obj, poly_obj), tuple[np.poly1d, np.poly1d])
assert_type(np.polydiv(poly_obj, AR_i8), tuple[np.poly1d, np.poly1d])
assert_type(np.polydiv(AR_f8, poly_obj), tuple[np.poly1d, np.poly1d])
assert_type(np.polydiv(poly_obj, AR_O), tuple[np.poly1d, np.poly1d])  # type: ignore[assert-type]
assert_type(np.polydiv(AR_O, poly_obj), tuple[np.poly1d, np.poly1d])  # type: ignore[assert-type]
assert_type(np.polydiv(AR_f4, AR_f4), tuple[_Array1D[np.float32], _Array1D[np.float32]])
assert_type(np.polydiv(AR_i8, AR_i8), tuple[_Array1D[np.float64 | Any], _Array1D[np.float64 | Any]])
assert_type(np.polydiv(AR_f8, AR_i8), tuple[_Array1D[np.float64 | Any], _Array1D[np.float64 | Any]])
assert_type(np.polydiv(AR_i8, AR_c16), tuple[_Array1D[np.complex128 | Any], _Array1D[np.complex128 | Any]])
