from collections.abc import Sequence
from typing import Any, ClassVar, Final, Literal as L, SupportsIndex, overload

import numpy as np
import numpy.typing as npt
from numpy._typing import (
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _NestedSequence,
    _NumberLike_co,
    _Shape,
    _SupportsArray,
)

from ._polybase import ABCPolyBase
from ._polytypes import (
    _Array1,
    _Array2,
    _FuncBinOp,
    _FuncCompanion,
    _FuncDer,
    _FuncFromRoots,
    _FuncGauss,
    _FuncInteg,
    _FuncLine,
    _FuncPoly2Ortho,
    _FuncPow,
    _FuncRoots,
    _FuncUnOp,
    _FuncVal2D,
    _FuncVal3D,
    _FuncValND,
    _FuncVander,
    _FuncVander2D,
    _FuncVander3D,
    _PolyScalar,
    _SupportsCoefOps,
)
from .polyutils import trimcoef as lagtrim

__all__ = [
    "lagzero",
    "lagone",
    "lagx",
    "lagdomain",
    "lagline",
    "lagadd",
    "lagsub",
    "lagmulx",
    "lagmul",
    "lagdiv",
    "lagpow",
    "lagval",
    "lagder",
    "lagint",
    "lag2poly",
    "poly2lag",
    "lagfromroots",
    "lagvander",
    "lagfit",
    "lagtrim",
    "lagroots",
    "Laguerre",
    "lagval2d",
    "lagval3d",
    "lagvalnd",
    "laggrid2d",
    "laggrid3d",
    "lagvander2d",
    "lagvander3d",
    "lagcompanion",
    "laggauss",
    "lagweight",
]

###

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _ToArray1D[ScalarT: np.generic, T] = _Array1D[ScalarT] | Sequence[T]

type _AsFloat64 = np.float64 | np.integer | np.bool
type _ToFloat64 = np.float64 | np.float32 | np.float16 | np.integer | np.bool

type _ToComplex128_1D = _SupportsArray[np.dtype[np.number | np.bool]] | Sequence[_NumberLike_co]
type _ToInt_1D = _SupportsArray[np.dtype[np.integer]] | Sequence[SupportsIndex]

###

poly2lag: Final[_FuncPoly2Ortho] = ...
lag2poly: Final[_FuncUnOp] = ...

lagdomain: Final[_Array2[np.float64]] = ...
lagzero: Final[_Array1[np.int_]] = ...
lagone: Final[_Array1[np.int_]] = ...
lagx: Final[_Array2[np.int_]] = ...

lagline: Final[_FuncLine] = ...
lagfromroots: Final[_FuncFromRoots] = ...
lagadd: Final[_FuncBinOp] = ...
lagsub: Final[_FuncBinOp] = ...
lagmulx: Final[_FuncUnOp] = ...
lagmul: Final[_FuncBinOp] = ...
lagdiv: Final[_FuncBinOp] = ...
lagpow: Final[_FuncPow] = ...
lagder: Final[_FuncDer] = ...
lagint: Final[_FuncInteg] = ...
lagval2d: Final[_FuncVal2D] = ...
lagval3d: Final[_FuncVal3D] = ...
lagvalnd: Final[_FuncValND] = ...

# keep in sync with `polynomial.*val`
@overload  # Nd +f64, 1d +f64
def lagval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f64, 1d ~c128
def lagval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, 1d +c128
def lagval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    c: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, 1d +O
def lagval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, 1d ? (fallback)
def lagval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 1d +f64
def lagval(
    x: float | _ToFloat64,
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.float64: ...
@overload  # 0d +c128, 1d ~c128
def lagval(
    x: complex | np.complex64 | _ToFloat64,
    c: _Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64
def lagval(
    x: Sequence[float],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> _Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128
def lagval(
    x: list[complex],
    c: _ToArray1D[np.complex128 | _AsFloat64, complex],
    tensor: bool = True,
) -> _Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?  (fallback)
def lagval(
    x: Sequence[_NumberLike_co],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> _Array1D[Any]: ...
@overload  # ?d ?, ?d ?  (fallback)
def lagval(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, ?d ~O
def lagval(
    x: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> _Array1D[np.object_]: ...
@overload  # poly, 1d ?
def lagval[PolyT: ABCPolyBase](
    x: PolyT,
    c: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> PolyT: ...
@overload  # 0d T, ?d ~O
def lagval[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> CoefT: ...

laggrid2d: Final[_FuncVal2D] = ...
laggrid3d: Final[_FuncVal3D] = ...
lagvander: Final[_FuncVander] = ...
lagvander2d: Final[_FuncVander2D] = ...
lagvander3d: Final[_FuncVander3D] = ...

# keep in sync with `polynomial.*fit`
@overload  # Nd +f64
def lagfit[ShapeT: _Shape](
    x: _ToArray1D[_ToFloat64, float],
    y: np.ndarray[ShapeT, np.dtype[_AsFloat64]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f64, full=True
def lagfit[ShapeT: _Shape](
    x: _ToArray1D[_ToFloat64, float],
    y: np.ndarray[ShapeT, np.dtype[_AsFloat64]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[np.ndarray[ShapeT, np.dtype[np.float64]], list[Any]]: ...
@overload  # 1d +f64
def lagfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> _Array1D[np.float64]: ...
@overload  # 1d +f64, full=True
def lagfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[_Array1D[np.float64], list[Any]]: ...
@overload  # 2d +f64
def lagfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> _Array2D[np.float64]: ...
@overload  # 2d +f64, full=True
def lagfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[_Array2D[np.float64], list[Any]]: ...
@overload  # Nd
def lagfit[ShapeT: _Shape](
    x: _ToComplex128_1D,
    y: np.ndarray[ShapeT, np.dtype[np.number | np.bool]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToComplex128_1D | None = None,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # Nd, full=True
def lagfit[ShapeT: _Shape](
    x: _ToComplex128_1D,
    y: np.ndarray[ShapeT, np.dtype[np.number | np.bool]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToComplex128_1D | None = None,
) -> tuple[np.ndarray[ShapeT, np.dtype[Any]], list[Any]]: ...
@overload  # ?d  (fallback)
def lagfit(
    x: _ToComplex128_1D,
    y: _ArrayLikeNumber_co,
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToComplex128_1D | None = None,
) -> npt.NDArray[Any]: ...
@overload  # ?d, full=True
def lagfit(
    x: _ToComplex128_1D,
    y: _ArrayLikeNumber_co,
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToComplex128_1D | None = None,
) -> tuple[npt.NDArray[Any], list[Any]]: ...

lagcompanion: Final[_FuncCompanion] = ...
lagroots: Final[_FuncRoots] = ...
laggauss: Final[_FuncGauss] = ...

@overload  # Nd T
def lagweight[ShapeT: _Shape, ScalarT: np.inexact](
    x: np.ndarray[ShapeT, np.dtype[ScalarT]],
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # Nd +f64
def lagweight[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.int64 | np.int32 | np.uint64 | np.uint32]],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f32
def lagweight[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.int16 | np.uint16]],
) -> np.ndarray[ShapeT, np.dtype[np.float32]]: ...
@overload  # Nd +f16
def lagweight[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.int8 | np.uint8]],
) -> np.ndarray[ShapeT, np.dtype[np.float16]]: ...
@overload  # 0d T
def lagweight[ScalarT: np.inexact](x: ScalarT) -> ScalarT: ...
@overload  # 0d +f64
def lagweight(x: float | np.int64 | np.int32 | np.uint64 | np.uint32) -> np.float64: ...
@overload  # 0d +f32
def lagweight(x: np.int16 | np.uint16) -> np.float32: ...
@overload  # 0d +f16
def lagweight(x: np.int8 | np.uint8) -> np.float16: ...
@overload  # 0d ~c128
def lagweight(x: complex) -> np.complex128 | Any: ...

class Laguerre(ABCPolyBase[L["L"]]):
    basis_name: ClassVar[L["L"]] = "L"  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]
    domain: _Array2[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
    window: _Array2[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
