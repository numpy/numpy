from decimal import Decimal
from typing import Any, assert_type

import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt

type _ArrFloat1D = np.ndarray[tuple[int], np.dtype[np.floating]]
type _ArrFloat1D64 = np.ndarray[tuple[int], np.dtype[np.float64]]
type _ArrComplex1D = np.ndarray[tuple[int], np.dtype[np.complexfloating]]
type _ArrComplex1D128 = np.ndarray[tuple[int], np.dtype[np.complex128]]
type _ArrObject1D = np.ndarray[tuple[int], np.dtype[np.object_]]

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

AR_b: npt.NDArray[np.bool]
AR_u4: npt.NDArray[np.uint32]
AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_O: npt.NDArray[np.object_[int]]
AR_f4_1d: _Array1D[np.float32]
AR_f8_2d: _Array2D[np.float64]
AR_f10_2d: _Array2D[np.longdouble]
AR_c16_2d: _Array2D[np.complex128]
AR_O_2d: _Array2D[np.object_[int]]

_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_c_1d: list[complex]
_py_decimal_1d: list[Decimal]
_py_f64_2d: list[_Array1D[np.float64]]

PS_poly: npp.Polynomial
PS_cheb: npp.Chebyshev
PS_leg: npp.Legendre
PS_lag: npp.Laguerre
PS_herm: npp.Hermite
PS_herme: npp.HermiteE

assert_type(npp.polynomial.polyroots(AR_f8), _ArrFloat1D64)
assert_type(npp.polynomial.polyroots(AR_c16), _ArrComplex1D128)
assert_type(npp.polynomial.polyroots(AR_O), _ArrObject1D)

assert_type(npp.polynomial.polyfromroots(AR_f8), _ArrFloat1D)
assert_type(npp.polynomial.polyfromroots(AR_c16), _ArrComplex1D)
assert_type(npp.polynomial.polyfromroots(AR_O), _ArrObject1D)

# assert_type(npp.polynomial.polyadd(AR_b, AR_b), NoReturn)
assert_type(npp.polynomial.polyadd(AR_u4, AR_b), _ArrFloat1D)
assert_type(npp.polynomial.polyadd(AR_i8, AR_i8), _ArrFloat1D)
assert_type(npp.polynomial.polyadd(AR_f8, AR_i8), _ArrFloat1D)
assert_type(npp.polynomial.polyadd(AR_i8, AR_c16), _ArrComplex1D)
assert_type(npp.polynomial.polyadd(AR_O, AR_O), _ArrObject1D)

assert_type(npp.polynomial.polymulx(AR_u4), _ArrFloat1D)
assert_type(npp.polynomial.polymulx(AR_i8), _ArrFloat1D)
assert_type(npp.polynomial.polymulx(AR_f8), _ArrFloat1D)
assert_type(npp.polynomial.polymulx(AR_c16), _ArrComplex1D)
assert_type(npp.polynomial.polymulx(AR_O), _ArrObject1D)

assert_type(npp.polynomial.polypow(AR_u4, 2), _ArrFloat1D)
assert_type(npp.polynomial.polypow(AR_i8, 2), _ArrFloat1D)
assert_type(npp.polynomial.polypow(AR_f8, 2), _ArrFloat1D)
assert_type(npp.polynomial.polypow(AR_c16, 2), _ArrComplex1D)
assert_type(npp.polynomial.polypow(AR_O, 2), _ArrObject1D)

# assert_type(npp.polynomial.polyder(PS_poly), npt.NDArray[np.object_])
assert_type(npp.polynomial.polyder(AR_f8), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyder(AR_c16), npt.NDArray[np.complexfloating])
assert_type(npp.polynomial.polyder(AR_O, m=2), npt.NDArray[np.object_])

# assert_type(npp.polynomial.polyint(PS_poly), npt.NDArray[np.object_])
assert_type(npp.polynomial.polyint(AR_f8), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyint(AR_f8, k=AR_c16), npt.NDArray[np.complexfloating])
assert_type(npp.polynomial.polyint(AR_O, m=2), npt.NDArray[np.object_])

assert_type(npp.polynomial.polyval(AR_f8_2d, _py_f_1d), _Array2D[np.float64])
assert_type(npp.polynomial.polyval(AR_f8_2d, _py_c_1d), _Array2D[np.complex128])
assert_type(npp.polynomial.polyval(AR_c16_2d, _py_f_1d), _Array2D[np.complex128])
assert_type(npp.polynomial.polyval(AR_O_2d, _py_f_1d), _Array2D[np.object_])
assert_type(npp.polynomial.polyval(AR_f10_2d, _py_f_1d), _Array2D[Any])
assert_type(npp.polynomial.polyval(1.0, _py_f_1d), np.float64)
assert_type(npp.polynomial.polyval(1j, _py_c_1d), np.complex128)
assert_type(npp.polynomial.polyval(_py_f_1d, _py_f_1d), _Array1D[np.float64])
assert_type(npp.polynomial.polyval(_py_c_1d, _py_f_1d), _Array1D[np.complex128])
assert_type(npp.polynomial.polyval([1.0, 2.0], AR_f4_1d), _Array1D[Any])
assert_type(npp.polynomial.polyval(AR_f8_2d, _py_f64_2d), npt.NDArray[Any] | Any)
assert_type(npp.polynomial.polyval(_py_decimal_1d, _py_decimal_1d), _Array1D[np.object_])
assert_type(npp.polynomial.polyval(PS_poly, _py_f_1d), npp.Polynomial)
assert_type(npp.polynomial.polyval(Decimal(), _py_decimal_1d), Decimal)

assert_type(npp.chebyshev.chebval(AR_f8_2d, _py_f_1d), _Array2D[np.float64])
assert_type(npp.chebyshev.chebval(AR_f8_2d, _py_c_1d), _Array2D[np.complex128])
assert_type(npp.chebyshev.chebval(AR_c16_2d, _py_f_1d), _Array2D[np.complex128])
assert_type(npp.chebyshev.chebval(AR_O_2d, _py_f_1d), _Array2D[np.object_])
assert_type(npp.chebyshev.chebval(AR_f10_2d, _py_f_1d), _Array2D[Any])
assert_type(npp.chebyshev.chebval(1.0, _py_f_1d), np.float64)
assert_type(npp.chebyshev.chebval(1j, _py_c_1d), np.complex128)
assert_type(npp.chebyshev.chebval(_py_f_1d, _py_f_1d), _Array1D[np.float64])
assert_type(npp.chebyshev.chebval(_py_c_1d, _py_f_1d), _Array1D[np.complex128])
assert_type(npp.chebyshev.chebval([1.0, 2.0], AR_f4_1d), _Array1D[Any])
assert_type(npp.chebyshev.chebval(AR_f8_2d, _py_f64_2d), npt.NDArray[Any] | Any)
assert_type(npp.chebyshev.chebval(_py_decimal_1d, _py_decimal_1d), _Array1D[np.object_])
assert_type(npp.chebyshev.chebval(PS_cheb, _py_f_1d), npp.Chebyshev)
assert_type(npp.chebyshev.chebval(Decimal(), _py_decimal_1d), Decimal)

assert_type(npp.legendre.legval(AR_f8_2d, _py_f_1d), _Array2D[np.float64])
assert_type(npp.legendre.legval(AR_f8_2d, _py_c_1d), _Array2D[np.complex128])
assert_type(npp.legendre.legval(AR_c16_2d, _py_f_1d), _Array2D[np.complex128])
assert_type(npp.legendre.legval(AR_O_2d, _py_f_1d), _Array2D[np.object_])
assert_type(npp.legendre.legval(AR_f10_2d, _py_f_1d), _Array2D[Any])
assert_type(npp.legendre.legval(1.0, _py_f_1d), np.float64)
assert_type(npp.legendre.legval(1j, _py_c_1d), np.complex128)
assert_type(npp.legendre.legval(_py_f_1d, _py_f_1d), _Array1D[np.float64])
assert_type(npp.legendre.legval(_py_c_1d, _py_f_1d), _Array1D[np.complex128])
assert_type(npp.legendre.legval([1.0, 2.0], AR_f4_1d), _Array1D[Any])
assert_type(npp.legendre.legval(AR_f8_2d, _py_f64_2d), npt.NDArray[Any] | Any)
assert_type(npp.legendre.legval(_py_decimal_1d, _py_decimal_1d), _Array1D[np.object_])
assert_type(npp.legendre.legval(PS_leg, _py_f_1d), npp.Legendre)
assert_type(npp.legendre.legval(Decimal(), _py_decimal_1d), Decimal | float)

assert_type(npp.laguerre.lagval(AR_f8_2d, _py_f_1d), _Array2D[np.float64])
assert_type(npp.laguerre.lagval(AR_f8_2d, _py_c_1d), _Array2D[np.complex128])
assert_type(npp.laguerre.lagval(AR_c16_2d, _py_f_1d), _Array2D[np.complex128])
assert_type(npp.laguerre.lagval(AR_O_2d, _py_f_1d), _Array2D[np.object_])
assert_type(npp.laguerre.lagval(AR_f10_2d, _py_f_1d), _Array2D[Any])
assert_type(npp.laguerre.lagval(1.0, _py_f_1d), np.float64)
assert_type(npp.laguerre.lagval(1j, _py_c_1d), np.complex128)
assert_type(npp.laguerre.lagval(_py_f_1d, _py_f_1d), _Array1D[np.float64])
assert_type(npp.laguerre.lagval(_py_c_1d, _py_f_1d), _Array1D[np.complex128])
assert_type(npp.laguerre.lagval([1.0, 2.0], AR_f4_1d), _Array1D[Any])
assert_type(npp.laguerre.lagval(AR_f8_2d, _py_f64_2d), npt.NDArray[Any] | Any)
assert_type(npp.laguerre.lagval(_py_decimal_1d, _py_decimal_1d), _Array1D[np.object_])
assert_type(npp.laguerre.lagval(PS_lag, _py_f_1d), npp.Laguerre)
assert_type(npp.laguerre.lagval(Decimal(), _py_decimal_1d), Decimal)

assert_type(npp.hermite.hermval(AR_f8_2d, _py_f_1d), _Array2D[np.float64])
assert_type(npp.hermite.hermval(AR_f8_2d, _py_c_1d), _Array2D[np.complex128])
assert_type(npp.hermite.hermval(AR_c16_2d, _py_f_1d), _Array2D[np.complex128])
assert_type(npp.hermite.hermval(AR_O_2d, _py_f_1d), _Array2D[np.object_])
assert_type(npp.hermite.hermval(AR_f10_2d, _py_f_1d), _Array2D[Any])
assert_type(npp.hermite.hermval(1.0, _py_f_1d), np.float64)
assert_type(npp.hermite.hermval(1j, _py_c_1d), np.complex128)
assert_type(npp.hermite.hermval(_py_f_1d, _py_f_1d), _Array1D[np.float64])
assert_type(npp.hermite.hermval(_py_c_1d, _py_f_1d), _Array1D[np.complex128])
assert_type(npp.hermite.hermval([1.0, 2.0], AR_f4_1d), _Array1D[Any])
assert_type(npp.hermite.hermval(AR_f8_2d, _py_f64_2d), npt.NDArray[Any] | Any)
assert_type(npp.hermite.hermval(_py_decimal_1d, _py_decimal_1d), _Array1D[np.object_])
assert_type(npp.hermite.hermval(PS_herm, _py_f_1d), npp.Hermite)
assert_type(npp.hermite.hermval(Decimal(), _py_decimal_1d), Decimal)

assert_type(npp.hermite_e.hermeval(AR_f8_2d, _py_f_1d), _Array2D[np.float64])
assert_type(npp.hermite_e.hermeval(AR_f8_2d, _py_c_1d), _Array2D[np.complex128])
assert_type(npp.hermite_e.hermeval(AR_c16_2d, _py_f_1d), _Array2D[np.complex128])
assert_type(npp.hermite_e.hermeval(AR_O_2d, _py_f_1d), _Array2D[np.object_])
assert_type(npp.hermite_e.hermeval(AR_f10_2d, _py_f_1d), _Array2D[Any])
assert_type(npp.hermite_e.hermeval(1.0, _py_f_1d), np.float64)
assert_type(npp.hermite_e.hermeval(1j, _py_c_1d), np.complex128)
assert_type(npp.hermite_e.hermeval(_py_f_1d, _py_f_1d), _Array1D[np.float64])
assert_type(npp.hermite_e.hermeval(_py_c_1d, _py_f_1d), _Array1D[np.complex128])
assert_type(npp.hermite_e.hermeval([1.0, 2.0], AR_f4_1d), _Array1D[Any])
assert_type(npp.hermite_e.hermeval(AR_f8_2d, _py_f64_2d), npt.NDArray[Any] | Any)
assert_type(npp.hermite_e.hermeval(_py_decimal_1d, _py_decimal_1d), _Array1D[np.object_])
assert_type(npp.hermite_e.hermeval(PS_herme, _py_f_1d), npp.HermiteE)
assert_type(npp.hermite_e.hermeval(Decimal(), _py_decimal_1d), Decimal)

assert_type(npp.polynomial.polyval2d(AR_b, AR_b, AR_b), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyval2d(AR_u4, AR_u4, AR_b), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyval2d(AR_i8, AR_i8, AR_i8), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyval2d(AR_f8, AR_f8, AR_i8), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyval2d(AR_i8, AR_i8, AR_c16), npt.NDArray[np.complexfloating])
assert_type(npp.polynomial.polyval2d(AR_O, AR_O, AR_O), npt.NDArray[np.object_])

assert_type(npp.polynomial.polyval3d(AR_b, AR_b, AR_b, AR_b), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyval3d(AR_u4, AR_u4, AR_u4, AR_b), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyval3d(AR_i8, AR_i8, AR_i8, AR_i8), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyval3d(AR_f8, AR_f8, AR_f8, AR_i8), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyval3d(AR_i8, AR_i8, AR_i8, AR_c16), npt.NDArray[np.complexfloating])
assert_type(npp.polynomial.polyval3d(AR_O, AR_O, AR_O, AR_O), npt.NDArray[np.object_])

assert_type(npp.polynomial.polyvalfromroots(AR_b, AR_b), npt.NDArray[np.float64 | Any])
assert_type(npp.polynomial.polyvalfromroots(AR_u4, AR_b), npt.NDArray[np.float64 | Any])
assert_type(npp.polynomial.polyvalfromroots(AR_i8, AR_i8), npt.NDArray[np.float64 | Any])
assert_type(npp.polynomial.polyvalfromroots(AR_f8, AR_i8), npt.NDArray[np.float64 | Any])
assert_type(npp.polynomial.polyvalfromroots(AR_i8, AR_c16), npt.NDArray[np.complex128 | Any])
assert_type(npp.polynomial.polyvalfromroots(AR_O, AR_O), npt.NDArray[np.object_ | Any])

assert_type(npp.polynomial.polyvander(AR_f8, 3), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyvander(AR_c16, 3), npt.NDArray[np.complexfloating])
assert_type(npp.polynomial.polyvander(AR_O, 3), npt.NDArray[np.object_])

assert_type(npp.polynomial.polyvander2d(AR_f8, AR_f8, [4, 2]), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyvander2d(AR_c16, AR_c16, [4, 2]), npt.NDArray[np.complexfloating])
assert_type(npp.polynomial.polyvander2d(AR_O, AR_O, [4, 2]), npt.NDArray[np.object_])

assert_type(npp.polynomial.polyvander3d(AR_f8, AR_f8, AR_f8, [4, 3, 2]), npt.NDArray[np.floating])
assert_type(npp.polynomial.polyvander3d(AR_c16, AR_c16, AR_c16, [4, 3, 2]), npt.NDArray[np.complexfloating])
assert_type(npp.polynomial.polyvander3d(AR_O, AR_O, AR_O, [4, 3, 2]), npt.NDArray[np.object_])

assert_type(npp.polynomial.polyfit(AR_f8, AR_f8_2d, 2), _Array2D[np.float64])
assert_type(npp.polynomial.polyfit(AR_f8, AR_f8_2d, AR_i8, full=True), tuple[_Array2D[np.float64], list[Any]])
assert_type(npp.polynomial.polyfit(AR_f8, _py_f_1d, 2), _Array1D[np.float64])
assert_type(npp.polynomial.polyfit(AR_f8, _py_f_1d, 2, full=True), tuple[_Array1D[np.float64], list[Any]])
assert_type(npp.polynomial.polyfit(AR_f8, _py_f_2d, 2), _Array2D[np.float64])
assert_type(npp.polynomial.polyfit(AR_f8, _py_f_2d, 2, full=True), tuple[_Array2D[np.float64], list[Any]])
assert_type(npp.polynomial.polyfit(AR_f8, AR_c16_2d, 2), _Array2D[Any])
assert_type(npp.polynomial.polyfit(AR_f8, AR_c16_2d, 2, full=True), tuple[_Array2D[Any], list[Any]])
assert_type(npp.polynomial.polyfit(AR_f8, _py_c_1d, 2), npt.NDArray[Any])
assert_type(npp.polynomial.polyfit(AR_f8, _py_c_1d, 2, full=True), tuple[npt.NDArray[Any], list[Any]])

assert_type(npp.chebyshev.chebgauss(2), tuple[_ArrFloat1D64, _ArrFloat1D64])

assert_type(npp.chebyshev.chebweight(AR_f8), npt.NDArray[np.float64])
assert_type(npp.chebyshev.chebweight(AR_c16), npt.NDArray[np.complex128])
assert_type(npp.chebyshev.chebweight(AR_O), npt.NDArray[np.object_])

assert_type(npp.chebyshev.poly2cheb(AR_f8), _ArrFloat1D)
assert_type(npp.chebyshev.poly2cheb(AR_c16), _ArrComplex1D)
assert_type(npp.chebyshev.poly2cheb(AR_O), _ArrObject1D)

assert_type(npp.chebyshev.cheb2poly(AR_f8), _ArrFloat1D)
assert_type(npp.chebyshev.cheb2poly(AR_c16), _ArrComplex1D)
assert_type(npp.chebyshev.cheb2poly(AR_O), _ArrObject1D)

assert_type(npp.chebyshev.chebpts1(6), _ArrFloat1D64)
assert_type(npp.chebyshev.chebpts2(6), _ArrFloat1D64)

assert_type(
    npp.chebyshev.chebinterpolate(np.tanh, 3),
    npt.NDArray[np.float64 | np.complex128 | np.object_],
)
