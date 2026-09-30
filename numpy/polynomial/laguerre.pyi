from collections.abc import Sequence
from typing import Any, ClassVar, Final, Literal as L, Never, SupportsIndex, overload

import numpy as np
import numpy.typing as npt
from numpy._typing import (
    Array1D,
    Array2D,
    Array3D,
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
    _FuncBinOp,
    _PolyScalar,
    _SupportsCoefOps,
    _ToCoef1D,
    _ToCoefND,
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

# workaround for mypy and pyright not following the typing spec for overloads
type _ArrayJustND[ScalarT: np.generic] = np.ndarray[tuple[Never, Never, Never, Never], np.dtype[ScalarT]]

type _ToArray1D[ScalarT: np.generic, T] = Array1D[ScalarT] | Sequence[T]
type _ToArray2D[ScalarT: np.generic, T] = Array2D[ScalarT] | Sequence[Sequence[T]]
type _ToArray3D[ScalarT: np.generic, T] = Array3D[ScalarT] | Sequence[Sequence[Sequence[T]]]

type _AsFloat64 = np.float64 | np.integer | np.bool
type _ToFloat64 = np.float64 | np.float32 | np.float16 | np.integer | np.bool

type _ToFloat64_ND = np.ndarray[Any, np.dtype[_ToFloat64]] | _NestedSequence[float]
type _ToComplex128_ND = np.ndarray[Any, np.dtype[np.complex128 | np.complex64 | _ToFloat64]] | _NestedSequence[complex]

type _ToComplex128_1D = _SupportsArray[np.dtype[np.number | np.bool]] | Sequence[_NumberLike_co]
type _ToInt_1D = _SupportsArray[np.dtype[np.integer]] | Sequence[SupportsIndex]

###

# keep in sync with `polynomial.poly2*`
@overload  # <=1d T
def poly2lag[ScalarT: np.longdouble | np.clongdouble](
    pol: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def poly2lag(
    pol: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_ToFloat64]]] | Sequence[float] | float,
) -> Array1D[np.float64]: ...
@overload  # <=1d +c128
def poly2lag(
    pol: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64]]] | list[complex],
) -> Array1D[np.complex128]: ...
@overload  # <=1d  (fallback)
def poly2lag(
    pol: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*2poly`
@overload  # <=1d T
def lag2poly[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def lag2poly(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer]]] | Sequence[float] | float,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128
def lag2poly(c: list[complex]) -> Array1D[np.complex128]: ...
@overload  # <=1d  (fallback)
def lag2poly(
    c: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> Array1D[Any]: ...

lagdomain: Final[Array1D[np.float64]] = ...
lagzero: Final[Array1D[np.int_]] = ...
lagone: Final[Array1D[np.int_]] = ...
lagx: Final[Array1D[np.int_]] = ...

# keep in sync with `polynomial.*line`
@overload  # 0d T, 0d T
def lagline[ScalarT: np.number | np.bool](
    off: ScalarT,
    scl: ScalarT,
) -> Array1D[ScalarT]: ...
@overload  # 0d ~i8, 0d ~i8
def lagline(
    off: int,
    scl: int,
) -> Array1D[np.int_]: ...
@overload  # 0d +f64, 0d +f64
def lagline(
    off: float | np.float64 | np.float32 | np.float16 | np.integer,
    scl: float | np.float64 | np.float32 | np.float16 | np.integer,
) -> Array1D[np.float64 | Any]: ...
@overload  # 0d +c128, 0d +c128
def lagline(
    off: complex | np.complex128 | np.complex64 | np.float64 | np.float32 | np.float16 | np.integer,
    scl: complex | np.complex128 | np.complex64 | np.float64 | np.float32 | np.float16 | np.integer,
) -> Array1D[np.complex128 | Any]: ...
@overload  # 0d, 0d  (fallback)
def lagline(
    off: _NumberLike_co | _SupportsCoefOps[Any] | np.object_,
    scl: _NumberLike_co | _SupportsCoefOps[Any] | np.object_,
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*fromroots`
@overload  # 1d T
def lagfromroots[ScalarT: np.longdouble | np.clongdouble](
    roots: _CanArray[Array1D[ScalarT]],
) -> Array1D[ScalarT]: ...
@overload  # 1d +f64
def lagfromroots(
    roots: _CanArray[Array1D[np.float64 | np.float32 | np.float16 | np.integer]] | Sequence[float],
) -> Array1D[np.float64]: ...
@overload  # 1d +c128
def lagfromroots(
    roots: _CanArray[Array1D[np.complex128 | np.complex64]] | list[complex],
) -> Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def lagfromroots(
    roots: _CanArray[Array1D[np.number | np.object_]] | Sequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*add`
@overload  # <=1d T, <=1d T
def lagadd[ScalarT: (np.float16, np.float32, np.longdouble, np.complex64, np.clongdouble)](
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64
def lagadd(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64 | np.integer]]] | Sequence[float] | float,
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64 | np.integer]]] | Sequence[float] | float,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128
def lagadd(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128]]] | list[complex],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64 | _ToFloat64]]] | Sequence[complex],
) -> Array1D[np.complex128]: ...
@overload  # <=1d +c128, <=1d ~c128
def lagadd(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64 | _ToFloat64]]] | Sequence[complex],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128]]] | list[complex],
) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O, <=1d
def lagadd(
    c1: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    c2: _ToCoef1D,
) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d ~O
def lagadd(
    c1: _ToCoef1D,
    c2: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d  (fallback)
def lagadd(
    c1: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
    c2: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> Array1D[Any]: ...

lagsub: Final[_FuncBinOp] = ...

# keep in sync with `polynomial.*mulx`
@overload  # <=1d T
def lagmulx[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def lagmulx(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer]]] | Sequence[float] | float,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128
def lagmulx(c: list[complex]) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O
def lagmulx(c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]]) -> Array1D[np.object_]: ...
@overload  # <=1d  (fallback)
def lagmulx(
    c: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> Array1D[Any]: ...

lagmul: Final[_FuncBinOp] = ...
lagdiv: Final[_FuncBinOp] = ...

# keep in sync with `polynomial.*pow`
@overload  # <=1d T
def lagpow[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def lagpow(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer]]] | Sequence[float] | float,
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128
def lagpow(
    c: list[complex],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O
def lagpow(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> Array1D[np.object_]: ...
@overload  # <=1d  (fallback)
def lagpow(
    c: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*der`
@overload  # ?d T  (workaround)
def lagder[ScalarT: np.inexact](
    c: _ArrayJustND[ScalarT],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def lagder(
    c: _ArrayJustND[np.integer | np.bool],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def lagder(
    c: _ArrayJustND[np.object_],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def lagder[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def lagder(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128
def lagder(
    c: list[complex],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O
def lagder(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.object_]: ...
@overload  # 2d T
def lagder[ScalarT: np.inexact](
    c: Array2D[ScalarT],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[ScalarT]: ...
@overload  # 2d +f64
def lagder(
    c: _ToArray2D[np.integer | np.bool, float],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.float64]: ...
@overload  # 2d ~c128
def lagder(
    c: Sequence[list[complex]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.complex128]: ...
@overload  # 2d ~O
def lagder(
    c: Array2D[np.object_],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.object_]: ...
@overload  # ?d  (fallback)
def lagder(
    c: _ToCoefND | _SupportsCoefOps[Any],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*int`
@overload  # ?d T  (workaround)
def lagint[ScalarT: np.inexact](
    c: _ArrayJustND[ScalarT],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def lagint(
    c: _ArrayJustND[np.integer | np.bool],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def lagint(
    c: _ArrayJustND[np.object_],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def lagint[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def lagint(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128
def lagint(
    c: list[complex],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O
def lagint(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.object_]: ...
@overload  # 2d T
def lagint[ScalarT: np.inexact](
    c: Array2D[ScalarT],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[ScalarT]: ...
@overload  # 2d +f64
def lagint(
    c: _ToArray2D[np.integer | np.bool, float],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.float64]: ...
@overload  # 2d ~c128
def lagint(
    c: Sequence[list[complex]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.complex128]: ...
@overload  # 2d ~O
def lagint(
    c: Array2D[np.object_],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.object_]: ...
@overload  # ?d  (fallback)
def lagint(
    c: _ToCoefND | _SupportsCoefOps[Any],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*val2d`
@overload  # Nd +f64, Nd +f64, 2d +f64
def lagval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray2D[_AsFloat64, float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, Nd +c128, 2d ~c128
def lagval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: Array2D[np.complex128] | Sequence[list[complex]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, Nd +c128, 2d +c128
def lagval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, Nd ~O, 2d +O
def lagval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    y: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, Nd ?, 2d ?  (fallback)
def lagval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    y: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 0d +f64, 2d +f64
def lagval2d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    c: _ToArray2D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 2d ~c128
def lagval2d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    c: Array2D[np.complex128] | Sequence[list[complex]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 2d +f64
def lagval2d(
    x: Sequence[float],
    y: Sequence[float],
    c: _ToArray2D[_AsFloat64, float],
) -> Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128, 2d +c128
def lagval2d(
    x: list[complex],
    y: Sequence[complex],
    c: _ToArray2D[np.complex128 | _AsFloat64, complex],
) -> Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?, 2d ?  (fallback)
def lagval2d(
    x: Sequence[_NumberLike_co],
    y: Sequence[_NumberLike_co],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> Array1D[Any]: ...
@overload  # ?d ?, ?d ?, ?d ?  (fallback)
def lagval2d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, 1d ~O, ?d ~O
def lagval2d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> Array1D[np.object_]: ...
@overload  # poly, poly, 2d ?
def lagval2d[PolyT: ABCPolyBase](
    x: PolyT,
    y: PolyT,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # 0d T, 0d T, ?d ~O
def lagval2d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> CoefT: ...

# keep in sync with `polynomial.*val3d`
@overload  # Nd +f64, Nd +f64, Nd +f64, 3d +f64
def lagval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray3D[_AsFloat64, float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, Nd +c128, Nd +c128, 3d ~c128
def lagval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, Nd +c128, Nd +c128, 3d +c128
def lagval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, Nd ~O, Nd ~O, 3d +O
def lagval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    y: np.ndarray[ShapeT, np.dtype[np.object_]],
    z: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, Nd ?, Nd ?, 3d ?  (fallback)
def lagval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    y: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    z: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 0d +f64, 0d +f64, 3d +f64
def lagval3d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    z: float | _ToFloat64,
    c: _ToArray3D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 0d +c128, 3d ~c128
def lagval3d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    z: complex | np.complex64 | _ToFloat64,
    c: Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 1d +f64, 3d +f64
def lagval3d(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    c: _ToArray3D[_AsFloat64, float],
) -> Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128, 1d +c128, 3d +c128
def lagval3d(
    x: list[complex],
    y: Sequence[complex],
    z: Sequence[complex],
    c: _ToArray3D[np.complex128 | _AsFloat64, complex],
) -> Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?, 1d ?, 3d ?  (fallback)
def lagval3d(
    x: Sequence[_NumberLike_co],
    y: Sequence[_NumberLike_co],
    z: Sequence[_NumberLike_co],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> Array1D[Any]: ...
@overload  # ?d ?, ?d ?, ?d ?, ?d ?  (fallback)
def lagval3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, ?d ~O
def lagval3d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    z: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> Array1D[np.object_]: ...
@overload  # 0d T, 0d T, 0d T, ?d ~O
def lagval3d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    z: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> CoefT: ...

# keep in sync with `polynomial.*valnd`
@overload  # *Nd +f64, ?d +f64
def lagvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[_ToFloat64]]],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # *Nd +c128, ?d ~c128
def lagvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]]],
    c: _SupportsArray[np.dtype[np.complex128]] | list[complex] | _NestedSequence[list[complex]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # *Nd ~c128, ?d +c128
def lagvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.complex128]]],
    c: _SupportsArray[np.dtype[np.complex128 | np.complex64 | _ToFloat64]] | _NestedSequence[complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # *Nd ~O, ?d +O
def lagvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.object_]]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # *Nd ?, ?d ?  (fallback)
def lagvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[_PolyScalar]]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # *0d +f64, ?d +f64
def lagvalnd(
    pts: Sequence[float | _ToFloat64],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> np.float64: ...
@overload  # *0d +c128, ?d ~c128
def lagvalnd(
    pts: Sequence[complex | np.complex64 | _ToFloat64],
    c: _SupportsArray[np.dtype[np.complex128]] | list[complex] | _NestedSequence[list[complex]],
) -> np.complex128: ...
@overload  # *1d +f64, ?d +f64
def lagvalnd(
    pts: Sequence[Sequence[float]],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> Array1D[np.float64]: ...
@overload  # *1d ~c128, ?d +c128
def lagvalnd(
    pts: Sequence[list[complex]],
    c: _SupportsArray[np.dtype[np.complex128 | _AsFloat64]] | _NestedSequence[complex],
) -> Array1D[np.complex128]: ...
@overload  # *1d ?, ?d ?  (fallback)
def lagvalnd(
    pts: Sequence[Sequence[_NumberLike_co]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co],
) -> Array1D[Any]: ...
@overload  # *poly, ?d ?
def lagvalnd[PolyT: ABCPolyBase](
    pts: Sequence[PolyT],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # *1d ~O, ?d ~O
def lagvalnd(
    pts: Sequence[Sequence[_SupportsCoefOps[Any]]],
    c: _SupportsArray[np.dtype[np.object_]] | _NestedSequence[_SupportsCoefOps[Any]],
) -> Array1D[np.object_]: ...
@overload  # *?d ?, ?d ?  (fallback)
def lagvalnd(
    pts: Sequence[_ToCoefND | _SupportsCoefOps[Any]],
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...

# keep in sync with `polynomial.*val`
@overload  # Nd +f64, 1d +f64
def lagval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, 1d ~c128
def lagval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: Array1D[np.complex128] | list[complex],
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
    c: Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64
def lagval(
    x: Sequence[float],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128
def lagval(
    x: list[complex],
    c: _ToArray1D[np.complex128 | _AsFloat64, complex],
    tensor: bool = True,
) -> Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?  (fallback)
def lagval(
    x: Sequence[_NumberLike_co],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> Array1D[Any]: ...
@overload  # ?d ?, ?d ?  (fallback)
def lagval(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToCoefND,
    tensor: bool = True,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, ?d ~O
def lagval(
    x: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> Array1D[np.object_]: ...
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

# keep in sync with `polynomial.*grid2d`
@overload  # ?d +f64, Nd +f64, 2d +f64  (workaround)
def laggrid2d(
    x: _ArrayJustND[_ToFloat64],
    y: _ToFloat64_ND,
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, ?d +f64, 2d +f64  (workaround)
def laggrid2d(
    x: _ToFloat64_ND,
    y: _ArrayJustND[_ToFloat64],
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ?, Nd ?, 2d ?  (workaround)
def laggrid2d(
    x: _ArrayJustND[_PolyScalar],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, ?d ?, 2d ?  (workaround)
def laggrid2d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayJustND[_PolyScalar],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # 0d +f64, 0d +f64, 2d +f64
def laggrid2d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    c: _ToArray2D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 2d ~c128
def laggrid2d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    c: Array2D[np.complex128] | Sequence[list[complex]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 2d +f64
def laggrid2d(
    x: _ToArray1D[_ToFloat64, float],
    y: _ToArray1D[_ToFloat64, float],
    c: _ToArray2D[_AsFloat64, float],
) -> Array2D[np.float64]: ...
@overload  # 1d +c128, 1d +c128, 2d ~c128
def laggrid2d(
    x: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: Array2D[np.complex128] | Sequence[list[complex]],
) -> Array2D[np.complex128]: ...
@overload  # 1d ~c128, 1d +c128, 2d +c128
def laggrid2d(
    x: Array1D[np.complex128] | list[complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 2d +O
def laggrid2d(
    x: Array1D[np.object_],
    y: Array1D[np.object_],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> Array2D[np.object_]: ...
@overload  # 1d ?, 1d ?, 2d ?  (fallback)
def laggrid2d(
    x: _ToArray1D[_PolyScalar, _NumberLike_co],
    y: _ToArray1D[_PolyScalar, _NumberLike_co],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> Array2D[Any]: ...
@overload  # ?d +f64, ?d +f64, 2d +f64
def laggrid2d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d +c128, ?d +c128, 2d ~c128
def laggrid2d(
    x: _ToComplex128_ND,
    y: _ToComplex128_ND,
    c: Array2D[np.complex128] | Sequence[list[complex]],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~c128, ?d +c128, 2d +c128
def laggrid2d(
    x: np.ndarray[Any, np.dtype[np.complex128]] | _NestedSequence[list[complex]] | list[complex],
    y: _ToComplex128_ND,
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~O, ?d ~O, 2d +O
def laggrid2d(
    x: _ArrayLikeObject_co,
    y: _ArrayLikeObject_co,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[np.object_]: ...
@overload  # 1d ~O, 1d ~O, ?d ~O
def laggrid2d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> Array2D[np.object_]: ...
@overload  # poly, poly, 2d ?
def laggrid2d[PolyT: ABCPolyBase](
    x: PolyT,
    y: PolyT,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # ?d ?, ?d ?, ?d ?  (fallback)
def laggrid2d(
    x: _ToCoefND,
    y: _ToCoefND,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 0d T, 0d T, ?d ~O
def laggrid2d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> CoefT: ...

# keep in sync with `polynomial.*grid3d`
@overload  # ?d +f64, Nd +f64, Nd +f64, 3d +f64  (workaround)
def laggrid3d(
    x: _ArrayJustND[_ToFloat64],
    y: _ToFloat64_ND,
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, ?d +f64, Nd +f64, 3d +f64  (workaround)
def laggrid3d(
    x: _ToFloat64_ND,
    y: _ArrayJustND[_ToFloat64],
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, Nd +f64, ?d +f64, 3d +f64  (workaround)
def laggrid3d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    z: _ArrayJustND[_ToFloat64],
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ?, Nd ?, Nd ?, 3d ?  (workaround)
def laggrid3d(
    x: _ArrayJustND[_PolyScalar],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, ?d ?, Nd ?, 3d ?  (workaround)
def laggrid3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayJustND[_PolyScalar],
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, Nd ?, ?d ?, 3d ?  (workaround)
def laggrid3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayJustND[_PolyScalar],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # 0d +f64, 0d +f64, 0d +f64, 3d +f64
def laggrid3d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    z: float | _ToFloat64,
    c: _ToArray3D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 0d +c128, 3d ~c128
def laggrid3d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    z: complex | np.complex64 | _ToFloat64,
    c: Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 1d +f64, 3d +f64
def laggrid3d(
    x: _ToArray1D[_ToFloat64, float],
    y: _ToArray1D[_ToFloat64, float],
    z: _ToArray1D[_ToFloat64, float],
    c: _ToArray3D[_AsFloat64, float],
) -> Array3D[np.float64]: ...
@overload  # 1d +c128, 1d +c128, 1d +c128, 3d ~c128
def laggrid3d(
    x: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    z: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> Array3D[np.complex128]: ...
@overload  # 1d ~c128, 1d +c128, 1d +c128, 3d +c128
def laggrid3d(
    x: Array1D[np.complex128] | list[complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    z: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> Array3D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, 3d +O
def laggrid3d(
    x: Array1D[np.object_],
    y: Array1D[np.object_],
    z: Array1D[np.object_],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> Array3D[np.object_]: ...
@overload  # 1d ?, 1d ?, 1d ?, 3d ?  (fallback)
def laggrid3d(
    x: _ToArray1D[_PolyScalar, _NumberLike_co],
    y: _ToArray1D[_PolyScalar, _NumberLike_co],
    z: _ToArray1D[_PolyScalar, _NumberLike_co],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> Array3D[Any]: ...
@overload  # ?d +f64, ?d +f64, ?d +f64, 3d +f64
def laggrid3d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d +c128, ?d +c128, ?d +c128, 3d ~c128
def laggrid3d(
    x: _ToComplex128_ND,
    y: _ToComplex128_ND,
    z: _ToComplex128_ND,
    c: Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~c128, ?d +c128, ?d +c128, 3d +c128
def laggrid3d(
    x: np.ndarray[Any, np.dtype[np.complex128]] | _NestedSequence[list[complex]] | list[complex],
    y: _ToComplex128_ND,
    z: _ToComplex128_ND,
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~O, ?d ~O, ?d ~O, 3d +O
def laggrid3d(
    x: _ArrayLikeObject_co,
    y: _ArrayLikeObject_co,
    z: _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[np.object_]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, ?d ~O
def laggrid3d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    z: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> Array3D[np.object_]: ...
@overload  # ?d ?, ?d ?, ?d ?, ?d ?  (fallback)
def laggrid3d(
    x: _ToCoefND,
    y: _ToCoefND,
    z: _ToCoefND,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 0d T, 0d T, 0d T, ?d ~O
def laggrid3d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    z: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> CoefT: ...

# keep in sync with `polynomial.*vander`
@overload  # ?d T  (workaround)
def lagvander[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    deg: SupportsIndex,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def lagvander(
    x: _ArrayJustND[np.integer | np.bool],
    deg: SupportsIndex,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def lagvander(
    x: _ArrayJustND[np.object_],
    deg: SupportsIndex,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def lagvander[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: SupportsIndex,
) -> Array2D[ScalarT]: ...
@overload  # <=1d +f64
def lagvander(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    deg: SupportsIndex,
) -> Array2D[np.float64]: ...
@overload  # <=1d ~c128
def lagvander(
    x: list[complex],
    deg: SupportsIndex,
) -> Array2D[np.complex128]: ...
@overload  # <=1d ~O
def lagvander(
    x: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    deg: SupportsIndex,
) -> Array2D[np.object_]: ...
@overload  # 2d T
def lagvander[ScalarT: np.inexact](
    x: Array2D[ScalarT],
    deg: SupportsIndex,
) -> Array3D[ScalarT]: ...
@overload  # 2d +f64
def lagvander(
    x: _ToArray2D[np.integer | np.bool, float],
    deg: SupportsIndex,
) -> Array3D[np.float64]: ...
@overload  # 2d ~c128
def lagvander(
    x: Sequence[list[complex]],
    deg: SupportsIndex,
) -> Array3D[np.complex128]: ...
@overload  # 2d ~O
def lagvander(
    x: Array2D[np.object_],
    deg: SupportsIndex,
) -> Array3D[np.object_]: ...
@overload  # ?d  (fallback)
def lagvander(
    x: _ToCoefND | _SupportsCoefOps[Any],
    deg: SupportsIndex,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*vander2d`
@overload  # ?d T, ?d T  (workaround)
def lagvander2d[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    y: _ArrayJustND[ScalarT],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64, ?d +f64  (workaround)
def lagvander2d(
    x: _ArrayJustND[_AsFloat64],
    y: _ArrayJustND[_AsFloat64],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O, ?d ~O  (workaround)
def lagvander2d(
    x: _ArrayJustND[np.object_],
    y: _ArrayJustND[np.object_],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T, <=1d T
def lagvander2d[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: Sequence[SupportsIndex],
) -> Array2D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64
def lagvander2d(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    deg: Sequence[SupportsIndex],
) -> Array2D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128
def lagvander2d(
    x: list[complex],
    y: Sequence[complex] | complex,
    deg: Sequence[SupportsIndex],
) -> Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O
def lagvander2d(
    x: Array1D[np.object_],
    y: Array1D[np.object_],
    deg: Sequence[SupportsIndex],
) -> Array2D[np.object_]: ...
@overload  # 2d T, 2d T
def lagvander2d[ScalarT: np.inexact](
    x: Array2D[ScalarT],
    y: Array2D[ScalarT],
    deg: Sequence[SupportsIndex],
) -> Array3D[ScalarT]: ...
@overload  # 2d +f64, 2d +f64
def lagvander2d(
    x: _ToArray2D[_AsFloat64, float],
    y: _ToArray2D[_AsFloat64, float],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.float64]: ...
@overload  # 2d ~c128, 2d +c128
def lagvander2d(
    x: Sequence[list[complex]],
    y: Sequence[Sequence[complex]],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.complex128]: ...
@overload  # 2d ~O, 2d ~O
def lagvander2d(
    x: Array2D[np.object_],
    y: Array2D[np.object_],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.object_]: ...
@overload  # ?d, ?d  (fallback)
def lagvander2d(
    x: _ToCoefND | _SupportsCoefOps[Any],
    y: _ToCoefND | _SupportsCoefOps[Any],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*vander3d`
@overload  # ?d T, ?d T, ?d T  (workaround)
def lagvander3d[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    y: _ArrayJustND[ScalarT],
    z: _ArrayJustND[ScalarT],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64, ?d +f64, ?d +f64  (workaround)
def lagvander3d(
    x: _ArrayJustND[_AsFloat64],
    y: _ArrayJustND[_AsFloat64],
    z: _ArrayJustND[_AsFloat64],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O, ?d ~O, ?d ~O  (workaround)
def lagvander3d(
    x: _ArrayJustND[np.object_],
    y: _ArrayJustND[np.object_],
    z: _ArrayJustND[np.object_],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T, <=1d T, <=1d T
def lagvander3d[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    z: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: Sequence[SupportsIndex],
) -> Array2D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64, <=1d +f64
def lagvander3d(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    z: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    deg: Sequence[SupportsIndex],
) -> Array2D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128, <=1d +c128
def lagvander3d(
    x: list[complex],
    y: Sequence[complex] | complex,
    z: Sequence[complex] | complex,
    deg: Sequence[SupportsIndex],
) -> Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O
def lagvander3d(
    x: Array1D[np.object_],
    y: Array1D[np.object_],
    z: Array1D[np.object_],
    deg: Sequence[SupportsIndex],
) -> Array2D[np.object_]: ...
@overload  # 2d T, 2d T, 2d T
def lagvander3d[ScalarT: np.inexact](
    x: Array2D[ScalarT],
    y: Array2D[ScalarT],
    z: Array2D[ScalarT],
    deg: Sequence[SupportsIndex],
) -> Array3D[ScalarT]: ...
@overload  # 2d +f64, 2d +f64, 2d +f64
def lagvander3d(
    x: _ToArray2D[_AsFloat64, float],
    y: _ToArray2D[_AsFloat64, float],
    z: _ToArray2D[_AsFloat64, float],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.float64]: ...
@overload  # 2d ~c128, 2d +c128, 2d +c128
def lagvander3d(
    x: Sequence[list[complex]],
    y: Sequence[Sequence[complex]],
    z: Sequence[Sequence[complex]],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.complex128]: ...
@overload  # 2d ~O, 2d ~O, 2d ~O
def lagvander3d(
    x: Array2D[np.object_],
    y: Array2D[np.object_],
    z: Array2D[np.object_],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.object_]: ...
@overload  # ?d, ?d, ?d  (fallback)
def lagvander3d(
    x: _ToCoefND | _SupportsCoefOps[Any],
    y: _ToCoefND | _SupportsCoefOps[Any],
    z: _ToCoefND | _SupportsCoefOps[Any],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[Any]: ...

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
) -> Array1D[np.float64]: ...
@overload  # 1d +f64, full=True
def lagfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[Array1D[np.float64], list[Any]]: ...
@overload  # 2d +f64
def lagfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> Array2D[np.float64]: ...
@overload  # 2d +f64, full=True
def lagfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[Array2D[np.float64], list[Any]]: ...
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

# keep in sync with `polynomial.*companion`
@overload  # 1d T
def lagcompanion[ScalarT: np.inexact](c: _CanArray[Array1D[ScalarT]]) -> Array2D[ScalarT]: ...
@overload  # 1d +f64
def lagcompanion(c: _CanArray[Array1D[np.integer]] | Sequence[float]) -> Array2D[np.float64]: ...
@overload  # 1d ~c128
def lagcompanion(c: list[complex]) -> Array2D[np.complex128]: ...
@overload  # 1d  (fallback)
def lagcompanion(c: _CanArray[Array1D[_PolyScalar]] | Sequence[_NumberLike_co | _SupportsCoefOps[Any]]) -> Array2D[Any]: ...

# keep in sync with `polynomial.*roots`
@overload  # 1d T
def lagroots[ScalarT: np.complexfloating](c: _CanArray[Array1D[ScalarT]] | Sequence[ScalarT]) -> Array1D[ScalarT]: ...
@overload  # 1d ~f32
def lagroots(c: _CanArray[Array1D[np.float32]] | Sequence[np.float32]) -> Array1D[np.float32 | np.complex64]: ...
@overload  # 1d +f64
def lagroots(c: _CanArray[Array1D[np.float64 | np.integer]] | Sequence[float]) -> Array1D[np.float64 | np.complex128]: ...
@overload  # 1d ~c128
def lagroots(c: list[complex]) -> Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def lagroots(c: _CanArray[Array1D[_PolyScalar]] | Sequence[_NumberLike_co | _SupportsCoefOps[Any]]) -> Array1D[Any]: ...

#
def laggauss(deg: SupportsIndex) -> tuple[Array1D[np.float64], Array1D[np.float64]]: ...

# keep in sync with `.hermite.hermweight`  (minus `np.bool`)
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
    domain: Array1D[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
    window: Array1D[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
