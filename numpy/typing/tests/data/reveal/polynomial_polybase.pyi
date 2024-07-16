import sys
from collections.abc import Sequence
from decimal import Decimal
from typing import Any, Literal as L, TypeAlias, TypeVar

import numpy as np
import numpy.polynomial as npp
import numpy.typing as npt

if sys.version_info >= (3, 11):
    from typing import LiteralString, assert_type
else:
    from typing_extensions import LiteralString, assert_type

_Ar_x: TypeAlias = npt.NDArray[np.inexact[Any] | np.object_]
_Ar_f: TypeAlias = npt.NDArray[np.floating[Any]]
_Ar_c: TypeAlias = npt.NDArray[np.complexfloating[Any, Any]]
_Ar_O: TypeAlias = npt.NDArray[np.object_]

_Ar_x_n: TypeAlias = np.ndarray[tuple[int], np.dtype[np.inexact[Any] | np.object_]]
_Ar_f_n: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating[Any]]]
_Ar_c_n: TypeAlias = np.ndarray[tuple[int], np.dtype[np.complexfloating[Any, Any]]]
_Ar_O_n: TypeAlias = np.ndarray[tuple[int], np.dtype[np.object_]]

_Ar_x_2: TypeAlias = np.ndarray[tuple[L[2]], np.dtype[np.inexact[Any] | np.object_]]
_Ar_f_2: TypeAlias = np.ndarray[tuple[L[2]], np.dtype[np.floating[Any]]]
_Ar_c_2: TypeAlias = np.ndarray[tuple[L[2]], np.dtype[np.complexfloating[Any, Any]]]
_Ar_O_2: TypeAlias = np.ndarray[tuple[L[2]], np.dtype[np.object_]]

_SCT = TypeVar("_SCT", bound=np.generic)
_Ar_1d: TypeAlias = np.ndarray[tuple[int], np.dtype[_SCT]]

_BasisName: TypeAlias = L["X"]

SC_i: np.integer[Any]
SC_i_co: int | np.integer[Any]
SC_f: np.floating[Any]
SC_f_co: float | np.floating[Any] | np.integer[Any]
SC_c: np.complexfloating[Any, Any]
SC_c_co: complex | np.number[Any]
SC_O: np.object_
SC_O_co: np.object_ | np.number[Any] | object

AR_i: npt.NDArray[np.integer[Any]]
AR_f: npt.NDArray[np.floating[Any]]
AR_f_co: npt.NDArray[np.floating[Any] | np.integer[Any]]
AR_c: npt.NDArray[np.complexfloating[Any, Any]]
AR_c_co: npt.NDArray[np.number[Any]]
AR_O: npt.NDArray[np.object_]
AR_O_co: npt.NDArray[np.object_ | np.number[Any]]

SQ_i: Sequence[int]
SQ_f: Sequence[float]
SQ_c: Sequence[complex]
SQ_O: Sequence[Decimal]

PS_poly: npp.Polynomial
PS_cheb: npp.Chebyshev
PS_herm: npp.Hermite
PS_herme: npp.HermiteE
PS_lag: npp.Laguerre
PS_leg: npp.Legendre
PS_all: (
    npp.Polynomial
    | npp.Chebyshev
    | npp.Hermite
    | npp.HermiteE
    | npp.Laguerre
    | npp.Legendre
)

assert_type(type(PS_poly).basis_name, None)
assert_type(type(PS_cheb).basis_name, L['T'])
assert_type(type(PS_herm).basis_name, L['H'])
assert_type(type(PS_herme).basis_name, L['He'])
assert_type(type(PS_lag).basis_name, L['L'])
assert_type(type(PS_leg).basis_name, L['P'])

assert_type(type(PS_all).__hash__, None)
assert_type(type(PS_all).__array_ufunc__, None)
assert_type(type(PS_all).maxpower, L[100])

assert_type(PS_all.coef, _Ar_x_n)
assert_type(PS_all.domain, _Ar_x_2)
assert_type(PS_all.window, _Ar_x_2)
assert_type(PS_all.symbol, LiteralString)

assert_type(PS_all.has_samecoef(PS_all), bool)
assert_type(PS_all.has_samedomain(PS_all), bool)
assert_type(PS_all.has_samewindow(PS_all), bool)
assert_type(PS_all.has_sametype(PS_all), bool)
assert_type(PS_poly.has_sametype(PS_poly), bool)
assert_type(PS_poly.has_sametype(PS_leg), bool)
assert_type(PS_poly.has_sametype(NotADirectoryError), L[False])

assert_type(PS_poly.copy(), npp.Polynomial)
assert_type(PS_cheb.copy(), npp.Chebyshev)
assert_type(PS_herm.copy(), npp.Hermite)
assert_type(PS_herme.copy(), npp.HermiteE)
assert_type(PS_lag.copy(), npp.Laguerre)
assert_type(PS_leg.copy(), npp.Legendre)

assert_type(PS_leg.cutdeg(), npp.Legendre)
assert_type(PS_leg.trim(), npp.Legendre)
assert_type(PS_leg.trim(tol=SC_f_co), npp.Legendre)
assert_type(PS_leg.truncate(SC_i_co), npp.Legendre)

assert_type(PS_all.convert(None, npp.Chebyshev), npp.Chebyshev)
assert_type(PS_all.convert((0, 1), npp.Laguerre), npp.Laguerre)
assert_type(PS_all.convert([0, 1], npp.Hermite, [-1, 1]), npp.Hermite)

assert_type(PS_all.degree(), int)
assert_type(PS_all.mapparms(), tuple[Any, Any])

assert_type(PS_poly.integ(), npp.Polynomial)
assert_type(PS_herme.integ(SC_i_co), npp.HermiteE)
assert_type(PS_lag.integ(SC_i_co, SC_f_co), npp.Laguerre)
assert_type(PS_poly.deriv(), npp.Polynomial)
assert_type(PS_herm.deriv(SC_i_co), npp.Hermite)

assert_type(PS_poly.roots(), _Ar_x_n)

assert_type(
    PS_poly.linspace(),
    tuple[_Ar_1d[np.float64 | np.complex128], _Ar_1d[np.float64 | np.complex128]],
)

assert_type(
    PS_poly.linspace(9),
    tuple[_Ar_1d[np.float64 | np.complex128], _Ar_1d[np.float64 | np.complex128]],
)

assert_type(PS_cheb.fit(AR_c_co, AR_c_co, SC_i_co), npp.Chebyshev)
assert_type(PS_leg.fit(AR_c_co, AR_c_co, AR_i), npp.Legendre)
assert_type(PS_herm.fit(AR_c_co, AR_c_co, SQ_i), npp.Hermite)
assert_type(PS_poly.fit(AR_c_co, SQ_c, SQ_i), npp.Polynomial)
assert_type(PS_lag.fit(SQ_c, SQ_c, SQ_i, full=False), npp.Laguerre)
assert_type(
    PS_herme.fit(SQ_c, AR_c_co, SC_i_co, full=True),
    tuple[npp.HermiteE, Sequence[np.inexact[Any] | np.int32]],
)

assert_type(type(PS_poly).fromroots(SC_i), npp.Polynomial)
assert_type(type(PS_poly).fromroots(SQ_i), npp.Polynomial)
assert_type(type(PS_poly).fromroots(AR_i), npp.Polynomial)
assert_type(type(PS_cheb).fromroots(SC_f), npp.Chebyshev)
assert_type(type(PS_cheb).fromroots(SQ_f), npp.Chebyshev)
assert_type(type(PS_cheb).fromroots(AR_f_co), npp.Chebyshev)
assert_type(type(PS_herm).fromroots(SC_c), npp.Hermite)
assert_type(type(PS_herm).fromroots(SQ_c), npp.Hermite)
assert_type(type(PS_herm).fromroots(AR_c_co), npp.Hermite)
assert_type(type(PS_leg).fromroots(SC_O), npp.Legendre)
assert_type(type(PS_leg).fromroots(SQ_O), npp.Legendre)
assert_type(type(PS_leg).fromroots(AR_O_co), npp.Legendre)

assert_type(type(PS_poly).identity(), npp.Polynomial)
assert_type(type(PS_cheb).identity(symbol='z'), npp.Chebyshev)

assert_type(type(PS_lag).basis(SC_i), npp.Laguerre)
assert_type(type(PS_leg).basis(32, symbol='u'), npp.Legendre)

assert_type(type(PS_herm).cast(PS_poly), npp.Hermite)
assert_type(type(PS_herme).cast(PS_leg), npp.HermiteE)

# TODO: ABCPolyBase operators
