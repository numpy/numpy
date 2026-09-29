from collections.abc import Sequence
from typing import Any, ClassVar, Final, Literal, Never, SupportsIndex, overload

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
    _CanArray,
    _FuncBinOp,
    _FuncCompanion,
    _FuncDer,
    _FuncFromRoots,
    _FuncInteg,
    _FuncLine,
    _FuncPow,
    _FuncRoots,
    _FuncUnOp,
    _FuncVal2D,
    _FuncVal3D,
    _FuncValND,
    _FuncVander2D,
    _FuncVander3D,
    _PolyScalar,
    _SupportsCoefOps,
)
from .polyutils import trimcoef as polytrim

__all__ = [
    "polyzero",
    "polyone",
    "polyx",
    "polydomain",
    "polyline",
    "polyadd",
    "polysub",
    "polymulx",
    "polymul",
    "polydiv",
    "polypow",
    "polyval",
    "polyvalfromroots",
    "polyder",
    "polyint",
    "polyfromroots",
    "polyvander",
    "polyfit",
    "polytrim",
    "polyroots",
    "Polynomial",
    "polyval2d",
    "polyval3d",
    "polyvalnd",
    "polygrid2d",
    "polygrid3d",
    "polyvander2d",
    "polyvander3d",
    "polycompanion",
]

###

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]

# workaround for mypy and pyright not following the typing spec for overloads
type _ArrayJustND[ScalarT: np.generic] = np.ndarray[tuple[Never, Never, Never, Never], np.dtype[ScalarT]]

type _ToArray1D[ScalarT: np.generic, T] = _Array1D[ScalarT] | Sequence[T]

type _AsFloat64 = np.float64 | np.integer | np.bool
type _ToFloat64 = np.float64 | np.float32 | np.float16 | np.integer | np.bool

type _ToComplex128_1D = _SupportsArray[np.dtype[np.number | np.bool]] | Sequence[_NumberLike_co]
type _ToInt_1D = _SupportsArray[np.dtype[np.integer]] | Sequence[SupportsIndex]

###

polydomain: Final[_Array2[np.float64]] = ...
polyzero: Final[_Array1[np.int_]] = ...
polyone: Final[_Array1[np.int_]] = ...
polyx: Final[_Array2[np.int_]] = ...

polyline: Final[_FuncLine] = ...
polyfromroots: Final[_FuncFromRoots] = ...
polyadd: Final[_FuncBinOp] = ...
polysub: Final[_FuncBinOp] = ...
polymulx: Final[_FuncUnOp] = ...
polymul: Final[_FuncBinOp] = ...
polydiv: Final[_FuncBinOp] = ...
polypow: Final[_FuncPow] = ...
polyder: Final[_FuncDer] = ...
polyint: Final[_FuncInteg] = ...
polyval2d: Final[_FuncVal2D] = ...
polyval3d: Final[_FuncVal3D] = ...
polyvalnd: Final[_FuncValND] = ...

# keep in sync with `polynomial.*val`
@overload  # Nd +f64, 1d +f64
def polyval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f64, 1d ~c128
def polyval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, 1d +c128
def polyval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    c: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, 1d +O
def polyval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, 1d ? (fallback)
def polyval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 1d +f64
def polyval(
    x: float | _ToFloat64,
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.float64: ...
@overload  # 0d +c128, 1d ~c128
def polyval(
    x: complex | np.complex64 | _ToFloat64,
    c: _Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64
def polyval(
    x: Sequence[float],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> _Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128
def polyval(
    x: list[complex],
    c: _ToArray1D[np.complex128 | _AsFloat64, complex],
    tensor: bool = True,
) -> _Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?  (fallback)
def polyval(
    x: Sequence[_NumberLike_co],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> _Array1D[Any]: ...
@overload  # ?d ?, ?d ?  (fallback)
def polyval(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, ?d ~O
def polyval(
    x: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> _Array1D[np.object_]: ...
@overload  # poly, 1d ?
def polyval[PolyT: ABCPolyBase](
    x: PolyT,
    c: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> PolyT: ...
@overload  # 0d T, ?d ~O
def polyval[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> CoefT: ...

# keep in sync with `polyval` (minus the `PolyT` overload)
@overload  # Nd +f64, 1d +f64
def polyvalfromroots[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    r: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f64, 1d ~c128
def polyvalfromroots[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    r: _Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, 1d +c128
def polyvalfromroots[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    r: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, 1d +O
def polyvalfromroots[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    r: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, 1d ? (fallback)
def polyvalfromroots[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    r: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 1d +f64
def polyvalfromroots(
    x: float | _ToFloat64,
    r: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.float64: ...
@overload  # 0d +c128, 1d ~c128
def polyvalfromroots(
    x: complex | np.complex64 | _ToFloat64,
    r: _Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64
def polyvalfromroots(
    x: Sequence[float],
    r: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> _Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128
def polyvalfromroots(
    x: list[complex],
    r: _ToArray1D[np.complex128 | _AsFloat64, complex],
    tensor: bool = True,
) -> _Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?  (fallback)
def polyvalfromroots(
    x: Sequence[_NumberLike_co],
    r: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> _Array1D[Any]: ...
@overload  # ?d ?, ?d ?  (fallback)
def polyvalfromroots(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    r: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, ?d ~O
def polyvalfromroots(
    x: Sequence[_SupportsCoefOps[Any]],
    r: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> _Array1D[np.object_]: ...
@overload  # 0d T, ?d ~O
def polyvalfromroots[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    r: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> CoefT: ...

polygrid2d: Final[_FuncVal2D] = ...
polygrid3d: Final[_FuncVal3D] = ...

# keep in sync with `polynomial.*vander`
@overload  # ?d T  (workaround)
def polyvander[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    deg: SupportsIndex,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def polyvander(
    x: _ArrayJustND[np.integer | np.bool],
    deg: SupportsIndex,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def polyvander(
    x: _ArrayJustND[np.object_],
    deg: SupportsIndex,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def polyvander[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: SupportsIndex,
) -> _Array2D[ScalarT]: ...
@overload  # <=1d +f64
def polyvander(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    deg: SupportsIndex,
) -> _Array2D[np.float64]: ...
@overload  # <=1d ~c128
def polyvander(
    x: list[complex],
    deg: SupportsIndex,
) -> _Array2D[np.complex128]: ...
@overload  # <=1d ~O
def polyvander(
    x: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    deg: SupportsIndex,
) -> _Array2D[np.object_]: ...
@overload  # 2d T
def polyvander[ScalarT: np.inexact](
    x: _Array2D[ScalarT],
    deg: SupportsIndex,
) -> _Array3D[ScalarT]: ...
@overload  # 2d +f64
def polyvander(
    x: _Array2D[np.integer | np.bool] | Sequence[Sequence[float]],
    deg: SupportsIndex,
) -> _Array3D[np.float64]: ...
@overload  # 2d ~c128
def polyvander(
    x: Sequence[list[complex]],
    deg: SupportsIndex,
) -> _Array3D[np.complex128]: ...
@overload  # 2d ~O
def polyvander(
    x: _Array2D[np.object_],
    deg: SupportsIndex,
) -> _Array3D[np.object_]: ...
@overload  # ?d  (fallback)
def polyvander(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co | _SupportsCoefOps[Any] | _NestedSequence[_SupportsCoefOps[Any]],
    deg: SupportsIndex,
) -> npt.NDArray[Any]: ...

polyvander2d: Final[_FuncVander2D] = ...
polyvander3d: Final[_FuncVander3D] = ...

# keep in sync with `polynomial.*fit`
@overload  # Nd +f64
def polyfit[ShapeT: _Shape](
    x: _ToArray1D[_ToFloat64, float],
    y: np.ndarray[ShapeT, np.dtype[_AsFloat64]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: Literal[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f64, full=True
def polyfit[ShapeT: _Shape](
    x: _ToArray1D[_ToFloat64, float],
    y: np.ndarray[ShapeT, np.dtype[_AsFloat64]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: Literal[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[np.ndarray[ShapeT, np.dtype[np.float64]], list[Any]]: ...
@overload  # 1d +f64
def polyfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: Literal[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> _Array1D[np.float64]: ...
@overload  # 1d +f64, full=True
def polyfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: Literal[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[_Array1D[np.float64], list[Any]]: ...
@overload  # 2d +f64
def polyfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: Literal[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> _Array2D[np.float64]: ...
@overload  # 2d +f64, full=True
def polyfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: Literal[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[_Array2D[np.float64], list[Any]]: ...
@overload  # Nd
def polyfit[ShapeT: _Shape](
    x: _ToComplex128_1D,
    y: np.ndarray[ShapeT, np.dtype[np.number | np.bool]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: Literal[False] = False,
    w: _ToComplex128_1D | None = None,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # Nd, full=True
def polyfit[ShapeT: _Shape](
    x: _ToComplex128_1D,
    y: np.ndarray[ShapeT, np.dtype[np.number | np.bool]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: Literal[True],
    w: _ToComplex128_1D | None = None,
) -> tuple[np.ndarray[ShapeT, np.dtype[Any]], list[Any]]: ...
@overload  # ?d  (fallback)
def polyfit(
    x: _ToComplex128_1D,
    y: _ArrayLikeNumber_co,
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: Literal[False] = False,
    w: _ToComplex128_1D | None = None,
) -> npt.NDArray[Any]: ...
@overload  # ?d, full=True
def polyfit(
    x: _ToComplex128_1D,
    y: _ArrayLikeNumber_co,
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: Literal[True],
    w: _ToComplex128_1D | None = None,
) -> tuple[npt.NDArray[Any], list[Any]]: ...

polycompanion: Final[_FuncCompanion] = ...
polyroots: Final[_FuncRoots] = ...

class Polynomial(ABCPolyBase[None]):
    basis_name: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]
    domain: _Array2[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
    window: _Array2[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
