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
    _FuncFromRoots,
    _FuncLine,
    _FuncPow,
    _FuncRoots,
    _FuncUnOp,
    _FuncVander3D,
    _PolyScalar,
    _SupportsCoefOps,
    _ToCoef1D,
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
type _ToArray2D[ScalarT: np.generic, T] = _Array2D[ScalarT] | Sequence[Sequence[T]]
type _ToArray3D[ScalarT: np.generic, T] = _Array3D[ScalarT] | Sequence[Sequence[Sequence[T]]]

type _AsFloat64 = np.float64 | np.integer | np.bool
type _ToFloat64 = np.float64 | np.float32 | np.float16 | np.integer | np.bool

type _ToFloat64_ND = np.ndarray[Any, np.dtype[_ToFloat64]] | _NestedSequence[float]
type _ToComplex128_ND = np.ndarray[Any, np.dtype[np.complex128 | np.complex64 | _ToFloat64]] | _NestedSequence[complex]

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

# keep in sync with `polynomial.*der`
@overload  # ?d T  (workaround)
def polyder[ScalarT: np.inexact](
    c: _ArrayJustND[ScalarT],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def polyder(
    c: _ArrayJustND[np.integer | np.bool],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def polyder(
    c: _ArrayJustND[np.object_],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def polyder[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[ScalarT]: ...
@overload  # <=1d +f64
def polyder(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.float64]: ...
@overload  # <=1d ~c128
def polyder(
    c: list[complex],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.complex128]: ...
@overload  # <=1d ~O
def polyder(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.object_]: ...
@overload  # 2d T
def polyder[ScalarT: np.inexact](
    c: _Array2D[ScalarT],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[ScalarT]: ...
@overload  # 2d +f64
def polyder(
    c: _Array2D[np.integer | np.bool] | Sequence[Sequence[float]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.float64]: ...
@overload  # 2d ~c128
def polyder(
    c: Sequence[list[complex]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.complex128]: ...
@overload  # 2d ~O
def polyder(
    c: _Array2D[np.object_],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.object_]: ...
@overload  # ?d  (fallback)
def polyder(
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _SupportsCoefOps[Any] | _NestedSequence[_SupportsCoefOps[Any]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*int`
@overload  # ?d T  (workaround)
def polyint[ScalarT: np.inexact](
    c: _ArrayJustND[ScalarT],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def polyint(
    c: _ArrayJustND[np.integer | np.bool],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def polyint(
    c: _ArrayJustND[np.object_],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def polyint[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[ScalarT]: ...
@overload  # <=1d +f64
def polyint(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.float64]: ...
@overload  # <=1d ~c128
def polyint(
    c: list[complex],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.complex128]: ...
@overload  # <=1d ~O
def polyint(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.object_]: ...
@overload  # 2d T
def polyint[ScalarT: np.inexact](
    c: _Array2D[ScalarT],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[ScalarT]: ...
@overload  # 2d +f64
def polyint(
    c: _Array2D[np.integer | np.bool] | Sequence[Sequence[float]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.float64]: ...
@overload  # 2d ~c128
def polyint(
    c: Sequence[list[complex]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.complex128]: ...
@overload  # 2d ~O
def polyint(
    c: _Array2D[np.object_],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.object_]: ...
@overload  # ?d  (fallback)
def polyint(
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _SupportsCoefOps[Any] | _NestedSequence[_SupportsCoefOps[Any]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*val2d`
@overload  # Nd +f64, Nd +f64, 2d +f64
def polyval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray2D[_AsFloat64, float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, Nd +c128, 2d ~c128
def polyval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _Array2D[np.complex128] | Sequence[list[complex]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, Nd +c128, 2d +c128
def polyval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, Nd ~O, 2d +O
def polyval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    y: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, Nd ?, 2d ?  (fallback)
def polyval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    y: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 0d +f64, 2d +f64
def polyval2d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    c: _ToArray2D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 2d ~c128
def polyval2d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    c: _Array2D[np.complex128] | Sequence[list[complex]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 2d +f64
def polyval2d(
    x: Sequence[float],
    y: Sequence[float],
    c: _ToArray2D[_AsFloat64, float],
) -> _Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128, 2d +c128
def polyval2d(
    x: list[complex],
    y: Sequence[complex],
    c: _ToArray2D[np.complex128 | _AsFloat64, complex],
) -> _Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?, 2d ?  (fallback)
def polyval2d(
    x: Sequence[_NumberLike_co],
    y: Sequence[_NumberLike_co],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> _Array1D[Any]: ...
@overload  # ?d ?, ?d ?, ?d ?  (fallback)
def polyval2d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, 1d ~O, ?d ~O
def polyval2d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> _Array1D[np.object_]: ...
@overload  # poly, poly, 2d ?
def polyval2d[PolyT: ABCPolyBase](
    x: PolyT,
    y: PolyT,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # 0d T, 0d T, ?d ~O
def polyval2d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> CoefT: ...

# keep in sync with `polynomial.*val3d`
@overload  # Nd +f64, Nd +f64, Nd +f64, 3d +f64
def polyval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray3D[_AsFloat64, float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, Nd +c128, Nd +c128, 3d ~c128
def polyval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, Nd +c128, Nd +c128, 3d +c128
def polyval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, Nd ~O, Nd ~O, 3d +O
def polyval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    y: np.ndarray[ShapeT, np.dtype[np.object_]],
    z: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, Nd ?, Nd ?, 3d ?  (fallback)
def polyval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    y: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    z: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 0d +f64, 0d +f64, 3d +f64
def polyval3d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    z: float | _ToFloat64,
    c: _ToArray3D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 0d +c128, 3d ~c128
def polyval3d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    z: complex | np.complex64 | _ToFloat64,
    c: _Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 1d +f64, 3d +f64
def polyval3d(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    c: _ToArray3D[_AsFloat64, float],
) -> _Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128, 1d +c128, 3d +c128
def polyval3d(
    x: list[complex],
    y: Sequence[complex],
    z: Sequence[complex],
    c: _ToArray3D[np.complex128 | _AsFloat64, complex],
) -> _Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?, 1d ?, 3d ?  (fallback)
def polyval3d(
    x: Sequence[_NumberLike_co],
    y: Sequence[_NumberLike_co],
    z: Sequence[_NumberLike_co],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> _Array1D[Any]: ...
@overload  # ?d ?, ?d ?, ?d ?, ?d ?  (fallback)
def polyval3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, ?d ~O
def polyval3d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    z: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> _Array1D[np.object_]: ...
@overload  # 0d T, 0d T, 0d T, ?d ~O
def polyval3d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    z: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> CoefT: ...

# keep in sync with `polynomial.*valnd`
@overload  # *Nd +f64, ?d +f64
def polyvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[_ToFloat64]]],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # *Nd +c128, ?d ~c128
def polyvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]]],
    c: _SupportsArray[np.dtype[np.complex128]] | list[complex] | _NestedSequence[list[complex]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # *Nd ~c128, ?d +c128
def polyvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.complex128]]],
    c: _SupportsArray[np.dtype[np.complex128 | np.complex64 | _ToFloat64]] | _NestedSequence[complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # *Nd ~O, ?d +O
def polyvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.object_]]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # *Nd ?, ?d ?  (fallback)
def polyvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[_PolyScalar]]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # *0d +f64, ?d +f64
def polyvalnd(
    pts: Sequence[float | _ToFloat64],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> np.float64: ...
@overload  # *0d +c128, ?d ~c128
def polyvalnd(
    pts: Sequence[complex | np.complex64 | _ToFloat64],
    c: _SupportsArray[np.dtype[np.complex128]] | list[complex] | _NestedSequence[list[complex]],
) -> np.complex128: ...
@overload  # *1d +f64, ?d +f64
def polyvalnd(
    pts: Sequence[Sequence[float]],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> _Array1D[np.float64]: ...
@overload  # *1d ~c128, ?d +c128
def polyvalnd(
    pts: Sequence[list[complex]],
    c: _SupportsArray[np.dtype[np.complex128 | _AsFloat64]] | _NestedSequence[complex],
) -> _Array1D[np.complex128]: ...
@overload  # *1d ?, ?d ?  (fallback)
def polyvalnd(
    pts: Sequence[Sequence[_NumberLike_co]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co],
) -> _Array1D[Any]: ...
@overload  # *poly, ?d ?
def polyvalnd[PolyT: ABCPolyBase](
    pts: Sequence[PolyT],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # *1d ~O, ?d ~O
def polyvalnd(
    pts: Sequence[Sequence[_SupportsCoefOps[Any]]],
    c: _SupportsArray[np.dtype[np.object_]] | _NestedSequence[_SupportsCoefOps[Any]],
) -> _Array1D[np.object_]: ...
@overload  # *?d ?, ?d ?  (fallback)
def polyvalnd(
    pts: Sequence[_ArrayLikeNumber_co | _ArrayLikeObject_co | _SupportsCoefOps[Any] | _NestedSequence[_SupportsCoefOps[Any]]],
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
) -> npt.NDArray[Any] | Any: ...

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

# keep in sync with `polynomial.*grid2d`
@overload  # ?d +f64, Nd +f64, 2d +f64  (workaround)
def polygrid2d(
    x: _ArrayJustND[_ToFloat64],
    y: _ToFloat64_ND,
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, ?d +f64, 2d +f64  (workaround)
def polygrid2d(
    x: _ToFloat64_ND,
    y: _ArrayJustND[_ToFloat64],
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ?, Nd ?, 2d ?  (workaround)
def polygrid2d(
    x: _ArrayJustND[_PolyScalar],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, ?d ?, 2d ?  (workaround)
def polygrid2d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayJustND[_PolyScalar],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # 0d +f64, 0d +f64, 2d +f64
def polygrid2d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    c: _ToArray2D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 2d ~c128
def polygrid2d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    c: _Array2D[np.complex128] | Sequence[list[complex]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 2d +f64
def polygrid2d(
    x: _ToArray1D[_ToFloat64, float],
    y: _ToArray1D[_ToFloat64, float],
    c: _ToArray2D[_AsFloat64, float],
) -> _Array2D[np.float64]: ...
@overload  # 1d +c128, 1d +c128, 2d ~c128
def polygrid2d(
    x: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _Array2D[np.complex128] | Sequence[list[complex]],
) -> _Array2D[np.complex128]: ...
@overload  # 1d ~c128, 1d +c128, 2d +c128
def polygrid2d(
    x: _Array1D[np.complex128] | list[complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> _Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 2d +O
def polygrid2d(
    x: _Array1D[np.object_],
    y: _Array1D[np.object_],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> _Array2D[np.object_]: ...
@overload  # 1d ?, 1d ?, 2d ?  (fallback)
def polygrid2d(
    x: _ToArray1D[_PolyScalar, _NumberLike_co],
    y: _ToArray1D[_PolyScalar, _NumberLike_co],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> _Array2D[Any]: ...
@overload  # ?d +f64, ?d +f64, 2d +f64
def polygrid2d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d +c128, ?d +c128, 2d ~c128
def polygrid2d(
    x: _ToComplex128_ND,
    y: _ToComplex128_ND,
    c: _Array2D[np.complex128] | Sequence[list[complex]],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~c128, ?d +c128, 2d +c128
def polygrid2d(
    x: np.ndarray[Any, np.dtype[np.complex128]] | _NestedSequence[list[complex]] | list[complex],
    y: _ToComplex128_ND,
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~O, ?d ~O, 2d +O
def polygrid2d(
    x: _ArrayLikeObject_co,
    y: _ArrayLikeObject_co,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[np.object_]: ...
@overload  # 1d ~O, 1d ~O, ?d ~O
def polygrid2d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> _Array2D[np.object_]: ...
@overload  # poly, poly, 2d ?
def polygrid2d[PolyT: ABCPolyBase](
    x: PolyT,
    y: PolyT,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # ?d ?, ?d ?, ?d ?  (fallback)
def polygrid2d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
) -> npt.NDArray[Any] | Any: ...
@overload  # 0d T, 0d T, ?d ~O
def polygrid2d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> CoefT: ...

# keep in sync with `polynomial.*grid3d`
@overload  # ?d +f64, Nd +f64, Nd +f64, 3d +f64  (workaround)
def polygrid3d(
    x: _ArrayJustND[_ToFloat64],
    y: _ToFloat64_ND,
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, ?d +f64, Nd +f64, 3d +f64  (workaround)
def polygrid3d(
    x: _ToFloat64_ND,
    y: _ArrayJustND[_ToFloat64],
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, Nd +f64, ?d +f64, 3d +f64  (workaround)
def polygrid3d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    z: _ArrayJustND[_ToFloat64],
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ?, Nd ?, Nd ?, 3d ?  (workaround)
def polygrid3d(
    x: _ArrayJustND[_PolyScalar],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, ?d ?, Nd ?, 3d ?  (workaround)
def polygrid3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayJustND[_PolyScalar],
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, Nd ?, ?d ?, 3d ?  (workaround)
def polygrid3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayJustND[_PolyScalar],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # 0d +f64, 0d +f64, 0d +f64, 3d +f64
def polygrid3d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    z: float | _ToFloat64,
    c: _ToArray3D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 0d +c128, 3d ~c128
def polygrid3d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    z: complex | np.complex64 | _ToFloat64,
    c: _Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 1d +f64, 3d +f64
def polygrid3d(
    x: _ToArray1D[_ToFloat64, float],
    y: _ToArray1D[_ToFloat64, float],
    z: _ToArray1D[_ToFloat64, float],
    c: _ToArray3D[_AsFloat64, float],
) -> _Array3D[np.float64]: ...
@overload  # 1d +c128, 1d +c128, 1d +c128, 3d ~c128
def polygrid3d(
    x: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    z: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> _Array3D[np.complex128]: ...
@overload  # 1d ~c128, 1d +c128, 1d +c128, 3d +c128
def polygrid3d(
    x: _Array1D[np.complex128] | list[complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    z: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> _Array3D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, 3d +O
def polygrid3d(
    x: _Array1D[np.object_],
    y: _Array1D[np.object_],
    z: _Array1D[np.object_],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> _Array3D[np.object_]: ...
@overload  # 1d ?, 1d ?, 1d ?, 3d ?  (fallback)
def polygrid3d(
    x: _ToArray1D[_PolyScalar, _NumberLike_co],
    y: _ToArray1D[_PolyScalar, _NumberLike_co],
    z: _ToArray1D[_PolyScalar, _NumberLike_co],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> _Array3D[Any]: ...
@overload  # ?d +f64, ?d +f64, ?d +f64, 3d +f64
def polygrid3d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d +c128, ?d +c128, ?d +c128, 3d ~c128
def polygrid3d(
    x: _ToComplex128_ND,
    y: _ToComplex128_ND,
    z: _ToComplex128_ND,
    c: _Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~c128, ?d +c128, ?d +c128, 3d +c128
def polygrid3d(
    x: np.ndarray[Any, np.dtype[np.complex128]] | _NestedSequence[list[complex]] | list[complex],
    y: _ToComplex128_ND,
    z: _ToComplex128_ND,
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~O, ?d ~O, ?d ~O, 3d +O
def polygrid3d(
    x: _ArrayLikeObject_co,
    y: _ArrayLikeObject_co,
    z: _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[np.object_]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, ?d ~O
def polygrid3d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    z: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> _Array3D[np.object_]: ...
@overload  # ?d ?, ?d ?, ?d ?, ?d ?  (fallback)
def polygrid3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
) -> npt.NDArray[Any] | Any: ...
@overload  # 0d T, 0d T, 0d T, ?d ~O
def polygrid3d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    z: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> CoefT: ...

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

# keep in sync with `polynomial.*vander2d`
@overload  # ?d T, ?d T  (workaround)
def polyvander2d[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    y: _ArrayJustND[ScalarT],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64, ?d +f64  (workaround)
def polyvander2d(
    x: _ArrayJustND[_AsFloat64],
    y: _ArrayJustND[_AsFloat64],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O, ?d ~O  (workaround)
def polyvander2d(
    x: _ArrayJustND[np.object_],
    y: _ArrayJustND[np.object_],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T, <=1d T
def polyvander2d[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: Sequence[SupportsIndex],
) -> _Array2D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64
def polyvander2d(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    deg: Sequence[SupportsIndex],
) -> _Array2D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128
def polyvander2d(
    x: list[complex],
    y: Sequence[complex] | complex,
    deg: Sequence[SupportsIndex],
) -> _Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O
def polyvander2d(
    x: _Array1D[np.object_],
    y: _Array1D[np.object_],
    deg: Sequence[SupportsIndex],
) -> _Array2D[np.object_]: ...
@overload  # 2d T, 2d T
def polyvander2d[ScalarT: np.inexact](
    x: _Array2D[ScalarT],
    y: _Array2D[ScalarT],
    deg: Sequence[SupportsIndex],
) -> _Array3D[ScalarT]: ...
@overload  # 2d +f64, 2d +f64
def polyvander2d(
    x: _Array2D[_AsFloat64] | Sequence[Sequence[float]],
    y: _Array2D[_AsFloat64] | Sequence[Sequence[float]],
    deg: Sequence[SupportsIndex],
) -> _Array3D[np.float64]: ...
@overload  # 2d ~c128, 2d +c128
def polyvander2d(
    x: Sequence[list[complex]],
    y: Sequence[Sequence[complex]],
    deg: Sequence[SupportsIndex],
) -> _Array3D[np.complex128]: ...
@overload  # 2d ~O, 2d ~O
def polyvander2d(
    x: _Array2D[np.object_],
    y: _Array2D[np.object_],
    deg: Sequence[SupportsIndex],
) -> _Array3D[np.object_]: ...
@overload  # ?d, ?d  (fallback)
def polyvander2d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co | _SupportsCoefOps[Any] | _NestedSequence[_SupportsCoefOps[Any]],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co | _SupportsCoefOps[Any] | _NestedSequence[_SupportsCoefOps[Any]],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[Any]: ...

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
