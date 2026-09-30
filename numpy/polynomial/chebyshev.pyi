from _typeshed import ConvertibleToInt
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    ClassVar,
    Concatenate,
    Final,
    Literal as L,
    Never,
    Self,
    SupportsIndex,
    overload,
)

import numpy as np
import numpy.typing as npt
from numpy._typing import (
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _IntLike_co,
    _NestedSequence,
    _NumberLike_co,
    _Shape,
    _SupportsArray,
)

from ._polybase import ABCPolyBase
from ._polytypes import (
    _AnyInt,
    _CanArray,
    _CoefSeries,
    _FuncBinOp,
    _PolyScalar,
    _Series,
    _SeriesLikeCoef_co,
    _SupportsCoefOps,
    _ToCoef1D,
    _ToCoefND,
)
from .polyutils import trimcoef as chebtrim

__all__ = [
    "chebzero",
    "chebone",
    "chebx",
    "chebdomain",
    "chebline",
    "chebadd",
    "chebsub",
    "chebmulx",
    "chebmul",
    "chebdiv",
    "chebpow",
    "chebval",
    "chebder",
    "chebint",
    "cheb2poly",
    "poly2cheb",
    "chebfromroots",
    "chebvander",
    "chebfit",
    "chebtrim",
    "chebroots",
    "chebpts1",
    "chebpts2",
    "Chebyshev",
    "chebval2d",
    "chebval3d",
    "chebvalnd",
    "chebgrid2d",
    "chebgrid3d",
    "chebvander2d",
    "chebvander3d",
    "chebcompanion",
    "chebgauss",
    "chebweight",
    "chebinterpolate",
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

def _cseries_to_zseries[ScalarT: np.number | np.object_](c: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...
def _zseries_to_cseries[ScalarT: np.number | np.object_](zs: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...
def _zseries_mul[ScalarT: np.number | np.object_](z1: npt.NDArray[ScalarT], z2: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...
def _zseries_div[ScalarT: np.number | np.object_](z1: npt.NDArray[ScalarT], z2: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...
def _zseries_der[ScalarT: np.number | np.object_](zs: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...
def _zseries_int[ScalarT: np.number | np.object_](zs: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...

# keep in sync with `polynomial.poly2*`
@overload  # <=1d T
def poly2cheb[ScalarT: np.longdouble | np.clongdouble](
    pol: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> _Array1D[ScalarT]: ...
@overload  # <=1d +f64
def poly2cheb(
    pol: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_ToFloat64]]] | Sequence[float] | float,
) -> _Array1D[np.float64]: ...
@overload  # <=1d +c128
def poly2cheb(
    pol: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64]]] | list[complex],
) -> _Array1D[np.complex128]: ...
@overload  # <=1d  (fallback)
def poly2cheb(
    pol: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> _Array1D[Any]: ...

# keep in sync with `polynomial.*2poly`
@overload  # <=1d T
def cheb2poly[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> _Array1D[ScalarT]: ...
@overload  # <=1d +f64
def cheb2poly(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer]]] | Sequence[float] | float,
) -> _Array1D[np.float64]: ...
@overload  # <=1d ~c128
def cheb2poly(c: list[complex]) -> _Array1D[np.complex128]: ...
@overload  # <=1d  (fallback)
def cheb2poly(
    c: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> _Array1D[Any]: ...

chebdomain: Final[_Array1D[np.float64]] = ...
chebzero: Final[_Array1D[np.int_]] = ...
chebone: Final[_Array1D[np.int_]] = ...
chebx: Final[_Array1D[np.int_]] = ...

# keep in sync with `polynomial.*line`
@overload  # 0d T, 0d T
def chebline[ScalarT: np.number | np.bool](
    off: ScalarT,
    scl: ScalarT,
) -> _Array1D[ScalarT]: ...
@overload  # 0d ~i8, 0d ~i8
def chebline(
    off: int,
    scl: int,
) -> _Array1D[np.int_]: ...
@overload  # 0d +f64, 0d +f64
def chebline(
    off: float | np.float64 | np.float32 | np.float16 | np.integer,
    scl: float | np.float64 | np.float32 | np.float16 | np.integer,
) -> _Array1D[np.float64 | Any]: ...
@overload  # 0d +c128, 0d +c128
def chebline(
    off: complex | np.complex128 | np.complex64 | np.float64 | np.float32 | np.float16 | np.integer,
    scl: complex | np.complex128 | np.complex64 | np.float64 | np.float32 | np.float16 | np.integer,
) -> _Array1D[np.complex128 | Any]: ...
@overload  # 0d, 0d  (fallback)
def chebline(
    off: _NumberLike_co | _SupportsCoefOps[Any] | np.object_,
    scl: _NumberLike_co | _SupportsCoefOps[Any] | np.object_,
) -> _Array1D[Any]: ...

# keep in sync with `polynomial.*fromroots`
@overload  # 1d T
def chebfromroots[ScalarT: np.longdouble | np.clongdouble](
    roots: _CanArray[_Array1D[ScalarT]],
) -> _Array1D[ScalarT]: ...
@overload  # 1d +f64
def chebfromroots(
    roots: _CanArray[_Array1D[np.float64 | np.float32 | np.float16 | np.integer]] | Sequence[float],
) -> _Array1D[np.float64]: ...
@overload  # 1d +c128
def chebfromroots(
    roots: _CanArray[_Array1D[np.complex128 | np.complex64]] | list[complex],
) -> _Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def chebfromroots(
    roots: _CanArray[_Array1D[np.number | np.object_]] | Sequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> _Array1D[Any]: ...

chebadd: Final[_FuncBinOp] = ...
chebsub: Final[_FuncBinOp] = ...

# keep in sync with `polynomial.*mulx`
@overload  # <=1d T
def chebmulx[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> _Array1D[ScalarT]: ...
@overload  # <=1d +f64
def chebmulx(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer]]] | Sequence[float] | float,
) -> _Array1D[np.float64]: ...
@overload  # <=1d ~c128
def chebmulx(c: list[complex]) -> _Array1D[np.complex128]: ...
@overload  # <=1d ~O
def chebmulx(c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]]) -> _Array1D[np.object_]: ...
@overload  # <=1d  (fallback)
def chebmulx(
    c: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> _Array1D[Any]: ...

chebmul: Final[_FuncBinOp] = ...
chebdiv: Final[_FuncBinOp] = ...

# keep in sync with `polynomial.*pow`
@overload  # <=1d T
def chebpow[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> _Array1D[ScalarT]: ...
@overload  # <=1d +f64
def chebpow(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer]]] | Sequence[float] | float,
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> _Array1D[np.float64]: ...
@overload  # <=1d ~c128
def chebpow(
    c: list[complex],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> _Array1D[np.complex128]: ...
@overload  # <=1d ~O
def chebpow(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> _Array1D[np.object_]: ...
@overload  # <=1d  (fallback)
def chebpow(
    c: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> _Array1D[Any]: ...

# keep in sync with `polynomial.*der`
@overload  # ?d T  (workaround)
def chebder[ScalarT: np.inexact](
    c: _ArrayJustND[ScalarT],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def chebder(
    c: _ArrayJustND[np.integer | np.bool],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def chebder(
    c: _ArrayJustND[np.object_],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def chebder[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[ScalarT]: ...
@overload  # <=1d +f64
def chebder(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.float64]: ...
@overload  # <=1d ~c128
def chebder(
    c: list[complex],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.complex128]: ...
@overload  # <=1d ~O
def chebder(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.object_]: ...
@overload  # 2d T
def chebder[ScalarT: np.inexact](
    c: _Array2D[ScalarT],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[ScalarT]: ...
@overload  # 2d +f64
def chebder(
    c: _ToArray2D[np.integer | np.bool, float],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.float64]: ...
@overload  # 2d ~c128
def chebder(
    c: Sequence[list[complex]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.complex128]: ...
@overload  # 2d ~O
def chebder(
    c: _Array2D[np.object_],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.object_]: ...
@overload  # ?d  (fallback)
def chebder(
    c: _ToCoefND | _SupportsCoefOps[Any],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*int`
@overload  # ?d T  (workaround)
def chebint[ScalarT: np.inexact](
    c: _ArrayJustND[ScalarT],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def chebint(
    c: _ArrayJustND[np.integer | np.bool],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def chebint(
    c: _ArrayJustND[np.object_],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def chebint[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[ScalarT]: ...
@overload  # <=1d +f64
def chebint(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.float64]: ...
@overload  # <=1d ~c128
def chebint(
    c: list[complex],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.complex128]: ...
@overload  # <=1d ~O
def chebint(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array1D[np.object_]: ...
@overload  # 2d T
def chebint[ScalarT: np.inexact](
    c: _Array2D[ScalarT],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[ScalarT]: ...
@overload  # 2d +f64
def chebint(
    c: _ToArray2D[np.integer | np.bool, float],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.float64]: ...
@overload  # 2d ~c128
def chebint(
    c: Sequence[list[complex]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.complex128]: ...
@overload  # 2d ~O
def chebint(
    c: _Array2D[np.object_],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> _Array2D[np.object_]: ...
@overload  # ?d  (fallback)
def chebint(
    c: _ToCoefND | _SupportsCoefOps[Any],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*val2d`
@overload  # Nd +f64, Nd +f64, 2d +f64
def chebval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray2D[_AsFloat64, float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, Nd +c128, 2d ~c128
def chebval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _Array2D[np.complex128] | Sequence[list[complex]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, Nd +c128, 2d +c128
def chebval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, Nd ~O, 2d +O
def chebval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    y: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, Nd ?, 2d ?  (fallback)
def chebval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    y: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 0d +f64, 2d +f64
def chebval2d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    c: _ToArray2D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 2d ~c128
def chebval2d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    c: _Array2D[np.complex128] | Sequence[list[complex]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 2d +f64
def chebval2d(
    x: Sequence[float],
    y: Sequence[float],
    c: _ToArray2D[_AsFloat64, float],
) -> _Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128, 2d +c128
def chebval2d(
    x: list[complex],
    y: Sequence[complex],
    c: _ToArray2D[np.complex128 | _AsFloat64, complex],
) -> _Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?, 2d ?  (fallback)
def chebval2d(
    x: Sequence[_NumberLike_co],
    y: Sequence[_NumberLike_co],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> _Array1D[Any]: ...
@overload  # ?d ?, ?d ?, ?d ?  (fallback)
def chebval2d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, 1d ~O, ?d ~O
def chebval2d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> _Array1D[np.object_]: ...
@overload  # poly, poly, 2d ?
def chebval2d[PolyT: ABCPolyBase](
    x: PolyT,
    y: PolyT,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # 0d T, 0d T, ?d ~O
def chebval2d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> CoefT: ...

# keep in sync with `polynomial.*val3d`
@overload  # Nd +f64, Nd +f64, Nd +f64, 3d +f64
def chebval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray3D[_AsFloat64, float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, Nd +c128, Nd +c128, 3d ~c128
def chebval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, Nd +c128, Nd +c128, 3d +c128
def chebval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, Nd ~O, Nd ~O, 3d +O
def chebval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    y: np.ndarray[ShapeT, np.dtype[np.object_]],
    z: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, Nd ?, Nd ?, 3d ?  (fallback)
def chebval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    y: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    z: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 0d +f64, 0d +f64, 3d +f64
def chebval3d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    z: float | _ToFloat64,
    c: _ToArray3D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 0d +c128, 3d ~c128
def chebval3d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    z: complex | np.complex64 | _ToFloat64,
    c: _Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 1d +f64, 3d +f64
def chebval3d(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    c: _ToArray3D[_AsFloat64, float],
) -> _Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128, 1d +c128, 3d +c128
def chebval3d(
    x: list[complex],
    y: Sequence[complex],
    z: Sequence[complex],
    c: _ToArray3D[np.complex128 | _AsFloat64, complex],
) -> _Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?, 1d ?, 3d ?  (fallback)
def chebval3d(
    x: Sequence[_NumberLike_co],
    y: Sequence[_NumberLike_co],
    z: Sequence[_NumberLike_co],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> _Array1D[Any]: ...
@overload  # ?d ?, ?d ?, ?d ?, ?d ?  (fallback)
def chebval3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, ?d ~O
def chebval3d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    z: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> _Array1D[np.object_]: ...
@overload  # 0d T, 0d T, 0d T, ?d ~O
def chebval3d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    z: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> CoefT: ...

# keep in sync with `polynomial.*valnd`
@overload  # *Nd +f64, ?d +f64
def chebvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[_ToFloat64]]],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # *Nd +c128, ?d ~c128
def chebvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]]],
    c: _SupportsArray[np.dtype[np.complex128]] | list[complex] | _NestedSequence[list[complex]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # *Nd ~c128, ?d +c128
def chebvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.complex128]]],
    c: _SupportsArray[np.dtype[np.complex128 | np.complex64 | _ToFloat64]] | _NestedSequence[complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # *Nd ~O, ?d +O
def chebvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.object_]]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # *Nd ?, ?d ?  (fallback)
def chebvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[_PolyScalar]]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # *0d +f64, ?d +f64
def chebvalnd(
    pts: Sequence[float | _ToFloat64],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> np.float64: ...
@overload  # *0d +c128, ?d ~c128
def chebvalnd(
    pts: Sequence[complex | np.complex64 | _ToFloat64],
    c: _SupportsArray[np.dtype[np.complex128]] | list[complex] | _NestedSequence[list[complex]],
) -> np.complex128: ...
@overload  # *1d +f64, ?d +f64
def chebvalnd(
    pts: Sequence[Sequence[float]],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> _Array1D[np.float64]: ...
@overload  # *1d ~c128, ?d +c128
def chebvalnd(
    pts: Sequence[list[complex]],
    c: _SupportsArray[np.dtype[np.complex128 | _AsFloat64]] | _NestedSequence[complex],
) -> _Array1D[np.complex128]: ...
@overload  # *1d ?, ?d ?  (fallback)
def chebvalnd(
    pts: Sequence[Sequence[_NumberLike_co]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co],
) -> _Array1D[Any]: ...
@overload  # *poly, ?d ?
def chebvalnd[PolyT: ABCPolyBase](
    pts: Sequence[PolyT],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # *1d ~O, ?d ~O
def chebvalnd(
    pts: Sequence[Sequence[_SupportsCoefOps[Any]]],
    c: _SupportsArray[np.dtype[np.object_]] | _NestedSequence[_SupportsCoefOps[Any]],
) -> _Array1D[np.object_]: ...
@overload  # *?d ?, ?d ?  (fallback)
def chebvalnd(
    pts: Sequence[_ToCoefND | _SupportsCoefOps[Any]],
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...

# keep in sync with `polynomial.*val`
@overload  # Nd +f64, 1d +f64
def chebval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, 1d ~c128
def chebval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, 1d +c128
def chebval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    c: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, 1d +O
def chebval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, 1d ? (fallback)
def chebval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 1d +f64
def chebval(
    x: float | _ToFloat64,
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.float64: ...
@overload  # 0d +c128, 1d ~c128
def chebval(
    x: complex | np.complex64 | _ToFloat64,
    c: _Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64
def chebval(
    x: Sequence[float],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> _Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128
def chebval(
    x: list[complex],
    c: _ToArray1D[np.complex128 | _AsFloat64, complex],
    tensor: bool = True,
) -> _Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?  (fallback)
def chebval(
    x: Sequence[_NumberLike_co],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> _Array1D[Any]: ...
@overload  # ?d ?, ?d ?  (fallback)
def chebval(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToCoefND,
    tensor: bool = True,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, ?d ~O
def chebval(
    x: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> _Array1D[np.object_]: ...
@overload  # poly, 1d ?
def chebval[PolyT: ABCPolyBase](
    x: PolyT,
    c: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> PolyT: ...
@overload  # 0d T, ?d ~O
def chebval[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> CoefT: ...

# keep in sync with `polynomial.*grid2d`
@overload  # ?d +f64, Nd +f64, 2d +f64  (workaround)
def chebgrid2d(
    x: _ArrayJustND[_ToFloat64],
    y: _ToFloat64_ND,
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, ?d +f64, 2d +f64  (workaround)
def chebgrid2d(
    x: _ToFloat64_ND,
    y: _ArrayJustND[_ToFloat64],
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ?, Nd ?, 2d ?  (workaround)
def chebgrid2d(
    x: _ArrayJustND[_PolyScalar],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, ?d ?, 2d ?  (workaround)
def chebgrid2d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayJustND[_PolyScalar],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # 0d +f64, 0d +f64, 2d +f64
def chebgrid2d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    c: _ToArray2D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 2d ~c128
def chebgrid2d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    c: _Array2D[np.complex128] | Sequence[list[complex]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 2d +f64
def chebgrid2d(
    x: _ToArray1D[_ToFloat64, float],
    y: _ToArray1D[_ToFloat64, float],
    c: _ToArray2D[_AsFloat64, float],
) -> _Array2D[np.float64]: ...
@overload  # 1d +c128, 1d +c128, 2d ~c128
def chebgrid2d(
    x: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _Array2D[np.complex128] | Sequence[list[complex]],
) -> _Array2D[np.complex128]: ...
@overload  # 1d ~c128, 1d +c128, 2d +c128
def chebgrid2d(
    x: _Array1D[np.complex128] | list[complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> _Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 2d +O
def chebgrid2d(
    x: _Array1D[np.object_],
    y: _Array1D[np.object_],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> _Array2D[np.object_]: ...
@overload  # 1d ?, 1d ?, 2d ?  (fallback)
def chebgrid2d(
    x: _ToArray1D[_PolyScalar, _NumberLike_co],
    y: _ToArray1D[_PolyScalar, _NumberLike_co],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> _Array2D[Any]: ...
@overload  # ?d +f64, ?d +f64, 2d +f64
def chebgrid2d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d +c128, ?d +c128, 2d ~c128
def chebgrid2d(
    x: _ToComplex128_ND,
    y: _ToComplex128_ND,
    c: _Array2D[np.complex128] | Sequence[list[complex]],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~c128, ?d +c128, 2d +c128
def chebgrid2d(
    x: np.ndarray[Any, np.dtype[np.complex128]] | _NestedSequence[list[complex]] | list[complex],
    y: _ToComplex128_ND,
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~O, ?d ~O, 2d +O
def chebgrid2d(
    x: _ArrayLikeObject_co,
    y: _ArrayLikeObject_co,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[np.object_]: ...
@overload  # 1d ~O, 1d ~O, ?d ~O
def chebgrid2d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> _Array2D[np.object_]: ...
@overload  # poly, poly, 2d ?
def chebgrid2d[PolyT: ABCPolyBase](
    x: PolyT,
    y: PolyT,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # ?d ?, ?d ?, ?d ?  (fallback)
def chebgrid2d(
    x: _ToCoefND,
    y: _ToCoefND,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 0d T, 0d T, ?d ~O
def chebgrid2d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> CoefT: ...

# keep in sync with `polynomial.*grid3d`
@overload  # ?d +f64, Nd +f64, Nd +f64, 3d +f64  (workaround)
def chebgrid3d(
    x: _ArrayJustND[_ToFloat64],
    y: _ToFloat64_ND,
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, ?d +f64, Nd +f64, 3d +f64  (workaround)
def chebgrid3d(
    x: _ToFloat64_ND,
    y: _ArrayJustND[_ToFloat64],
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, Nd +f64, ?d +f64, 3d +f64  (workaround)
def chebgrid3d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    z: _ArrayJustND[_ToFloat64],
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ?, Nd ?, Nd ?, 3d ?  (workaround)
def chebgrid3d(
    x: _ArrayJustND[_PolyScalar],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, ?d ?, Nd ?, 3d ?  (workaround)
def chebgrid3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayJustND[_PolyScalar],
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, Nd ?, ?d ?, 3d ?  (workaround)
def chebgrid3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayJustND[_PolyScalar],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # 0d +f64, 0d +f64, 0d +f64, 3d +f64
def chebgrid3d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    z: float | _ToFloat64,
    c: _ToArray3D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 0d +c128, 3d ~c128
def chebgrid3d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    z: complex | np.complex64 | _ToFloat64,
    c: _Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 1d +f64, 3d +f64
def chebgrid3d(
    x: _ToArray1D[_ToFloat64, float],
    y: _ToArray1D[_ToFloat64, float],
    z: _ToArray1D[_ToFloat64, float],
    c: _ToArray3D[_AsFloat64, float],
) -> _Array3D[np.float64]: ...
@overload  # 1d +c128, 1d +c128, 1d +c128, 3d ~c128
def chebgrid3d(
    x: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    z: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> _Array3D[np.complex128]: ...
@overload  # 1d ~c128, 1d +c128, 1d +c128, 3d +c128
def chebgrid3d(
    x: _Array1D[np.complex128] | list[complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    z: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> _Array3D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, 3d +O
def chebgrid3d(
    x: _Array1D[np.object_],
    y: _Array1D[np.object_],
    z: _Array1D[np.object_],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> _Array3D[np.object_]: ...
@overload  # 1d ?, 1d ?, 1d ?, 3d ?  (fallback)
def chebgrid3d(
    x: _ToArray1D[_PolyScalar, _NumberLike_co],
    y: _ToArray1D[_PolyScalar, _NumberLike_co],
    z: _ToArray1D[_PolyScalar, _NumberLike_co],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> _Array3D[Any]: ...
@overload  # ?d +f64, ?d +f64, ?d +f64, 3d +f64
def chebgrid3d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d +c128, ?d +c128, ?d +c128, 3d ~c128
def chebgrid3d(
    x: _ToComplex128_ND,
    y: _ToComplex128_ND,
    z: _ToComplex128_ND,
    c: _Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~c128, ?d +c128, ?d +c128, 3d +c128
def chebgrid3d(
    x: np.ndarray[Any, np.dtype[np.complex128]] | _NestedSequence[list[complex]] | list[complex],
    y: _ToComplex128_ND,
    z: _ToComplex128_ND,
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~O, ?d ~O, ?d ~O, 3d +O
def chebgrid3d(
    x: _ArrayLikeObject_co,
    y: _ArrayLikeObject_co,
    z: _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[np.object_]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, ?d ~O
def chebgrid3d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    z: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> _Array3D[np.object_]: ...
@overload  # ?d ?, ?d ?, ?d ?, ?d ?  (fallback)
def chebgrid3d(
    x: _ToCoefND,
    y: _ToCoefND,
    z: _ToCoefND,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 0d T, 0d T, 0d T, ?d ~O
def chebgrid3d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    z: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> CoefT: ...

# keep in sync with `polynomial.*vander`
@overload  # ?d T  (workaround)
def chebvander[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    deg: SupportsIndex,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def chebvander(
    x: _ArrayJustND[np.integer | np.bool],
    deg: SupportsIndex,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def chebvander(
    x: _ArrayJustND[np.object_],
    deg: SupportsIndex,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def chebvander[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: SupportsIndex,
) -> _Array2D[ScalarT]: ...
@overload  # <=1d +f64
def chebvander(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    deg: SupportsIndex,
) -> _Array2D[np.float64]: ...
@overload  # <=1d ~c128
def chebvander(
    x: list[complex],
    deg: SupportsIndex,
) -> _Array2D[np.complex128]: ...
@overload  # <=1d ~O
def chebvander(
    x: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    deg: SupportsIndex,
) -> _Array2D[np.object_]: ...
@overload  # 2d T
def chebvander[ScalarT: np.inexact](
    x: _Array2D[ScalarT],
    deg: SupportsIndex,
) -> _Array3D[ScalarT]: ...
@overload  # 2d +f64
def chebvander(
    x: _ToArray2D[np.integer | np.bool, float],
    deg: SupportsIndex,
) -> _Array3D[np.float64]: ...
@overload  # 2d ~c128
def chebvander(
    x: Sequence[list[complex]],
    deg: SupportsIndex,
) -> _Array3D[np.complex128]: ...
@overload  # 2d ~O
def chebvander(
    x: _Array2D[np.object_],
    deg: SupportsIndex,
) -> _Array3D[np.object_]: ...
@overload  # ?d  (fallback)
def chebvander(
    x: _ToCoefND | _SupportsCoefOps[Any],
    deg: SupportsIndex,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*vander2d`
@overload  # ?d T, ?d T  (workaround)
def chebvander2d[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    y: _ArrayJustND[ScalarT],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64, ?d +f64  (workaround)
def chebvander2d(
    x: _ArrayJustND[_AsFloat64],
    y: _ArrayJustND[_AsFloat64],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O, ?d ~O  (workaround)
def chebvander2d(
    x: _ArrayJustND[np.object_],
    y: _ArrayJustND[np.object_],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T, <=1d T
def chebvander2d[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: Sequence[SupportsIndex],
) -> _Array2D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64
def chebvander2d(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    deg: Sequence[SupportsIndex],
) -> _Array2D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128
def chebvander2d(
    x: list[complex],
    y: Sequence[complex] | complex,
    deg: Sequence[SupportsIndex],
) -> _Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O
def chebvander2d(
    x: _Array1D[np.object_],
    y: _Array1D[np.object_],
    deg: Sequence[SupportsIndex],
) -> _Array2D[np.object_]: ...
@overload  # 2d T, 2d T
def chebvander2d[ScalarT: np.inexact](
    x: _Array2D[ScalarT],
    y: _Array2D[ScalarT],
    deg: Sequence[SupportsIndex],
) -> _Array3D[ScalarT]: ...
@overload  # 2d +f64, 2d +f64
def chebvander2d(
    x: _ToArray2D[_AsFloat64, float],
    y: _ToArray2D[_AsFloat64, float],
    deg: Sequence[SupportsIndex],
) -> _Array3D[np.float64]: ...
@overload  # 2d ~c128, 2d +c128
def chebvander2d(
    x: Sequence[list[complex]],
    y: Sequence[Sequence[complex]],
    deg: Sequence[SupportsIndex],
) -> _Array3D[np.complex128]: ...
@overload  # 2d ~O, 2d ~O
def chebvander2d(
    x: _Array2D[np.object_],
    y: _Array2D[np.object_],
    deg: Sequence[SupportsIndex],
) -> _Array3D[np.object_]: ...
@overload  # ?d, ?d  (fallback)
def chebvander2d(
    x: _ToCoefND | _SupportsCoefOps[Any],
    y: _ToCoefND | _SupportsCoefOps[Any],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*vander3d`
@overload  # ?d T, ?d T, ?d T  (workaround)
def chebvander3d[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    y: _ArrayJustND[ScalarT],
    z: _ArrayJustND[ScalarT],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64, ?d +f64, ?d +f64  (workaround)
def chebvander3d(
    x: _ArrayJustND[_AsFloat64],
    y: _ArrayJustND[_AsFloat64],
    z: _ArrayJustND[_AsFloat64],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O, ?d ~O, ?d ~O  (workaround)
def chebvander3d(
    x: _ArrayJustND[np.object_],
    y: _ArrayJustND[np.object_],
    z: _ArrayJustND[np.object_],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T, <=1d T, <=1d T
def chebvander3d[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    z: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: Sequence[SupportsIndex],
) -> _Array2D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64, <=1d +f64
def chebvander3d(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    z: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    deg: Sequence[SupportsIndex],
) -> _Array2D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128, <=1d +c128
def chebvander3d(
    x: list[complex],
    y: Sequence[complex] | complex,
    z: Sequence[complex] | complex,
    deg: Sequence[SupportsIndex],
) -> _Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O
def chebvander3d(
    x: _Array1D[np.object_],
    y: _Array1D[np.object_],
    z: _Array1D[np.object_],
    deg: Sequence[SupportsIndex],
) -> _Array2D[np.object_]: ...
@overload  # 2d T, 2d T, 2d T
def chebvander3d[ScalarT: np.inexact](
    x: _Array2D[ScalarT],
    y: _Array2D[ScalarT],
    z: _Array2D[ScalarT],
    deg: Sequence[SupportsIndex],
) -> _Array3D[ScalarT]: ...
@overload  # 2d +f64, 2d +f64, 2d +f64
def chebvander3d(
    x: _ToArray2D[_AsFloat64, float],
    y: _ToArray2D[_AsFloat64, float],
    z: _ToArray2D[_AsFloat64, float],
    deg: Sequence[SupportsIndex],
) -> _Array3D[np.float64]: ...
@overload  # 2d ~c128, 2d +c128, 2d +c128
def chebvander3d(
    x: Sequence[list[complex]],
    y: Sequence[Sequence[complex]],
    z: Sequence[Sequence[complex]],
    deg: Sequence[SupportsIndex],
) -> _Array3D[np.complex128]: ...
@overload  # 2d ~O, 2d ~O, 2d ~O
def chebvander3d(
    x: _Array2D[np.object_],
    y: _Array2D[np.object_],
    z: _Array2D[np.object_],
    deg: Sequence[SupportsIndex],
) -> _Array3D[np.object_]: ...
@overload  # ?d, ?d, ?d  (fallback)
def chebvander3d(
    x: _ToCoefND | _SupportsCoefOps[Any],
    y: _ToCoefND | _SupportsCoefOps[Any],
    z: _ToCoefND | _SupportsCoefOps[Any],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*fit`
@overload  # Nd +f64
def chebfit[ShapeT: _Shape](
    x: _ToArray1D[_ToFloat64, float],
    y: np.ndarray[ShapeT, np.dtype[_AsFloat64]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f64, full=True
def chebfit[ShapeT: _Shape](
    x: _ToArray1D[_ToFloat64, float],
    y: np.ndarray[ShapeT, np.dtype[_AsFloat64]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[np.ndarray[ShapeT, np.dtype[np.float64]], list[Any]]: ...
@overload  # 1d +f64
def chebfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> _Array1D[np.float64]: ...
@overload  # 1d +f64, full=True
def chebfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[_Array1D[np.float64], list[Any]]: ...
@overload  # 2d +f64
def chebfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> _Array2D[np.float64]: ...
@overload  # 2d +f64, full=True
def chebfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[_Array2D[np.float64], list[Any]]: ...
@overload  # Nd
def chebfit[ShapeT: _Shape](
    x: _ToComplex128_1D,
    y: np.ndarray[ShapeT, np.dtype[np.number | np.bool]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToComplex128_1D | None = None,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # Nd, full=True
def chebfit[ShapeT: _Shape](
    x: _ToComplex128_1D,
    y: np.ndarray[ShapeT, np.dtype[np.number | np.bool]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToComplex128_1D | None = None,
) -> tuple[np.ndarray[ShapeT, np.dtype[Any]], list[Any]]: ...
@overload  # ?d  (fallback)
def chebfit(
    x: _ToComplex128_1D,
    y: _ArrayLikeNumber_co,
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToComplex128_1D | None = None,
) -> npt.NDArray[Any]: ...
@overload  # ?d, full=True
def chebfit(
    x: _ToComplex128_1D,
    y: _ArrayLikeNumber_co,
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToComplex128_1D | None = None,
) -> tuple[npt.NDArray[Any], list[Any]]: ...

# keep in sync with `polynomial.*companion`
@overload  # 1d T
def chebcompanion[ScalarT: np.inexact](c: _CanArray[_Array1D[ScalarT]]) -> _Array2D[ScalarT]: ...
@overload  # 1d +f64
def chebcompanion(c: _CanArray[_Array1D[np.integer]] | Sequence[float]) -> _Array2D[np.float64]: ...
@overload  # 1d ~c128
def chebcompanion(c: list[complex]) -> _Array2D[np.complex128]: ...
@overload  # 1d  (fallback)
def chebcompanion(c: _CanArray[_Array1D[_PolyScalar]] | Sequence[_NumberLike_co | _SupportsCoefOps[Any]]) -> _Array2D[Any]: ...

# keep in sync with `polynomial.*roots`
@overload  # 1d T
def chebroots[ScalarT: np.complexfloating](c: _CanArray[_Array1D[ScalarT]] | Sequence[ScalarT]) -> _Array1D[ScalarT]: ...
@overload  # 1d ~f32
def chebroots(c: _CanArray[_Array1D[np.float32]] | Sequence[np.float32]) -> _Array1D[np.float32 | np.complex64]: ...
@overload  # 1d +f64
def chebroots(c: _CanArray[_Array1D[np.float64 | np.integer]] | Sequence[float]) -> _Array1D[np.float64 | np.complex128]: ...
@overload  # 1d ~c128
def chebroots(c: list[complex]) -> _Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def chebroots(c: _CanArray[_Array1D[_PolyScalar]] | Sequence[_NumberLike_co | _SupportsCoefOps[Any]]) -> _Array1D[Any]: ...

#
def chebgauss(deg: SupportsIndex) -> tuple[_Array1D[np.float64], _Array1D[np.float64]]: ...

# keep in sync with `.hermite_e.hermeweight`
@overload  # Nd T
def chebweight[ShapeT: _Shape, ScalarT: np.inexact](
    x: np.ndarray[ShapeT, np.dtype[ScalarT]],
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # Nd +f64
def chebweight[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.integer | np.bool]],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # 0d T
def chebweight[ScalarT: np.inexact](x: ScalarT) -> ScalarT: ...
@overload  # 0d +f64
def chebweight(x: float | np.integer | np.bool) -> np.float64: ...
@overload  # 0d ~c128
def chebweight(x: complex) -> np.complex128 | Any: ...

#
def chebpts1(npts: ConvertibleToInt) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...
def chebpts2(npts: ConvertibleToInt) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...

#
@overload  # ?d +f64  (workaround)
def chebinterpolate(
    func: Callable[[_Array1D[np.float64]], _ArrayJustND[_ToFloat64]],
    deg: _IntLike_co,
    args: tuple[()] = (),
) -> npt.NDArray[np.float64]: ...
@overload  # ?d +f64, args=<given>  (workaround)
def chebinterpolate[*Ts](
    func: Callable[[_Array1D[np.float64], *Ts], _ArrayJustND[_ToFloat64]],
    deg: _IntLike_co,
    args: tuple[*Ts],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~c128  (workaround)
def chebinterpolate(
    func: Callable[[_Array1D[np.float64]], _ArrayJustND[np.complex128]],
    deg: _IntLike_co,
    args: tuple[()] = (),
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~c128, args=<given>  (workaround)
def chebinterpolate[*Ts](
    func: Callable[[_Array1D[np.float64], *Ts], _ArrayJustND[np.complex128]],
    deg: _IntLike_co,
    args: tuple[*Ts],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~O  (workaround)
def chebinterpolate(
    func: Callable[[_Array1D[np.float64]], _ArrayJustND[np.object_]],
    deg: _IntLike_co,
    args: tuple[()] = (),
) -> npt.NDArray[np.object_]: ...
@overload  # ?d ~O, args=<given>  (workaround)
def chebinterpolate[*Ts](
    func: Callable[[_Array1D[np.float64], *Ts], _ArrayJustND[np.object_]],
    deg: _IntLike_co,
    args: tuple[*Ts],
) -> npt.NDArray[np.object_]: ...
@overload  # 1d +f64
def chebinterpolate(
    func: Callable[[_Array1D[np.float64]], _Array1D[_ToFloat64]],
    deg: _IntLike_co,
    args: tuple[()] = (),
) -> _Array1D[np.float64]: ...
@overload  # 1d +f64, args=<given>
def chebinterpolate[*Ts](
    func: Callable[[_Array1D[np.float64], *Ts], _Array1D[_ToFloat64]],
    deg: _IntLike_co,
    args: tuple[*Ts],
) -> _Array1D[np.float64]: ...
@overload  # 1d ~c128
def chebinterpolate(
    func: Callable[[_Array1D[np.float64]], _Array1D[np.complex128]],
    deg: _IntLike_co,
    args: tuple[()] = (),
) -> _Array1D[np.complex128]: ...
@overload  # 1d ~c128, args=<given>
def chebinterpolate[*Ts](
    func: Callable[[_Array1D[np.float64], *Ts], _Array1D[np.complex128]],
    deg: _IntLike_co,
    args: tuple[*Ts],
) -> _Array1D[np.complex128]: ...
@overload  # 1d ~O
def chebinterpolate(
    func: Callable[[_Array1D[np.float64]], _Array1D[np.object_]],
    deg: _IntLike_co,
    args: tuple[()] = (),
) -> _Array1D[np.object_]: ...
@overload  # 1d ~O, args=<given>
def chebinterpolate[*Ts](
    func: Callable[[_Array1D[np.float64], *Ts], _Array1D[np.object_]],
    deg: _IntLike_co,
    args: tuple[*Ts],
) -> _Array1D[np.object_]: ...
@overload  # ?  (fallback)
def chebinterpolate(
    func: Callable[..., object],
    deg: _IntLike_co,
    args: Iterable[Any] = (),
) -> npt.NDArray[Any]: ...

class Chebyshev(ABCPolyBase[L["T"]]):
    basis_name: ClassVar[L["T"]] = "T"  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]
    domain: _Array1D[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
    window: _Array1D[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]

    @overload
    @classmethod
    def interpolate(
        cls,
        func: Callable[[npt.NDArray[np.float64]], _CoefSeries],
        deg: _IntLike_co,
        domain: _SeriesLikeCoef_co | None = None,
        args: tuple[()] = (),
    ) -> Self: ...
    @overload
    @classmethod
    def interpolate(
        cls,
        func: Callable[Concatenate[npt.NDArray[np.float64], ...], _CoefSeries],
        deg: _IntLike_co,
        domain: _SeriesLikeCoef_co | None = None,
        *,
        args: Iterable[Any],
    ) -> Self: ...
    @overload
    @classmethod
    def interpolate(
        cls,
        func: Callable[Concatenate[npt.NDArray[np.float64], ...], _CoefSeries],
        deg: _IntLike_co,
        domain: _SeriesLikeCoef_co | None,
        args: Iterable[Any],
    ) -> Self: ...
