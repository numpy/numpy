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
from .polyutils import trimcoef as legtrim

__all__ = [
    "legzero",
    "legone",
    "legx",
    "legdomain",
    "legline",
    "legadd",
    "legsub",
    "legmulx",
    "legmul",
    "legdiv",
    "legpow",
    "legval",
    "legder",
    "legint",
    "leg2poly",
    "poly2leg",
    "legfromroots",
    "legvander",
    "legfit",
    "legtrim",
    "legroots",
    "Legendre",
    "legval2d",
    "legval3d",
    "legvalnd",
    "leggrid2d",
    "leggrid3d",
    "legvander2d",
    "legvander3d",
    "legcompanion",
    "leggauss",
    "legweight",
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
def poly2leg[ScalarT: np.longdouble | np.clongdouble](
    pol: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def poly2leg(
    pol: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_ToFloat64]]] | Sequence[float] | float,
) -> Array1D[np.float64]: ...
@overload  # <=1d +c128
def poly2leg(
    pol: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64]]] | list[complex],
) -> Array1D[np.complex128]: ...
@overload  # <=1d  (fallback)
def poly2leg(
    pol: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*2poly`
@overload  # <=1d T
def leg2poly[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def leg2poly(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer]]] | Sequence[float] | float,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128
def leg2poly(c: list[complex]) -> Array1D[np.complex128]: ...
@overload  # <=1d  (fallback)
def leg2poly(
    c: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> Array1D[Any]: ...

legdomain: Final[Array1D[np.float64]] = ...
legzero: Final[Array1D[np.int_]] = ...
legone: Final[Array1D[np.int_]] = ...
legx: Final[Array1D[np.int_]] = ...

# keep in sync with `polynomial.*line`
@overload  # 0d T, 0d T
def legline[ScalarT: np.number | np.bool](
    off: ScalarT,
    scl: ScalarT,
) -> Array1D[ScalarT]: ...
@overload  # 0d ~i8, 0d ~i8
def legline(
    off: int,
    scl: int,
) -> Array1D[np.int_]: ...
@overload  # 0d +f64, 0d +f64
def legline(
    off: float | np.float64 | np.float32 | np.float16 | np.integer,
    scl: float | np.float64 | np.float32 | np.float16 | np.integer,
) -> Array1D[np.float64 | Any]: ...
@overload  # 0d +c128, 0d +c128
def legline(
    off: complex | np.complex128 | np.complex64 | np.float64 | np.float32 | np.float16 | np.integer,
    scl: complex | np.complex128 | np.complex64 | np.float64 | np.float32 | np.float16 | np.integer,
) -> Array1D[np.complex128 | Any]: ...
@overload  # 0d, 0d  (fallback)
def legline(
    off: _NumberLike_co | _SupportsCoefOps[Any] | np.object_,
    scl: _NumberLike_co | _SupportsCoefOps[Any] | np.object_,
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*fromroots`
@overload  # 1d T
def legfromroots[ScalarT: np.longdouble | np.clongdouble](
    roots: _CanArray[Array1D[ScalarT]],
) -> Array1D[ScalarT]: ...
@overload  # 1d +f64
def legfromroots(
    roots: _CanArray[Array1D[np.float64 | np.float32 | np.float16 | np.integer]] | Sequence[float],
) -> Array1D[np.float64]: ...
@overload  # 1d +c128
def legfromroots(
    roots: _CanArray[Array1D[np.complex128 | np.complex64]] | list[complex],
) -> Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def legfromroots(
    roots: _CanArray[Array1D[np.number | np.object_]] | Sequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*{add,sub,mul}`
@overload  # <=1d T, <=1d T
def legadd[ScalarT: (np.float16, np.float32, np.longdouble, np.complex64, np.clongdouble)](
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64
def legadd(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64 | np.integer]]] | Sequence[float] | float,
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64 | np.integer]]] | Sequence[float] | float,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128
def legadd(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128]]] | list[complex],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64 | _ToFloat64]]] | Sequence[complex],
) -> Array1D[np.complex128]: ...
@overload  # <=1d +c128, <=1d ~c128
def legadd(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64 | _ToFloat64]]] | Sequence[complex],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128]]] | list[complex],
) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O, <=1d
def legadd(
    c1: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    c2: _ToCoef1D,
) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d ~O
def legadd(
    c1: _ToCoef1D,
    c2: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d  (fallback)
def legadd(
    c1: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
    c2: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*{add,sub,mul}`
@overload  # <=1d T, <=1d T
def legsub[ScalarT: (np.float16, np.float32, np.longdouble, np.complex64, np.clongdouble)](
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64
def legsub(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64 | np.integer]]] | Sequence[float] | float,
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64 | np.integer]]] | Sequence[float] | float,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128
def legsub(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128]]] | list[complex],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64 | _ToFloat64]]] | Sequence[complex],
) -> Array1D[np.complex128]: ...
@overload  # <=1d +c128, <=1d ~c128
def legsub(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64 | _ToFloat64]]] | Sequence[complex],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128]]] | list[complex],
) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O, <=1d
def legsub(
    c1: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    c2: _ToCoef1D,
) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d ~O
def legsub(
    c1: _ToCoef1D,
    c2: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d  (fallback)
def legsub(
    c1: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
    c2: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*mulx`
@overload  # <=1d T
def legmulx[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def legmulx(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer]]] | Sequence[float] | float,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128
def legmulx(c: list[complex]) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O
def legmulx(c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]]) -> Array1D[np.object_]: ...
@overload  # <=1d  (fallback)
def legmulx(
    c: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*{add,sub,mul}`
@overload  # <=1d T, <=1d T
def legmul[ScalarT: (np.float16, np.float32, np.longdouble, np.complex64, np.clongdouble)](
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64
def legmul(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64 | np.integer]]] | Sequence[float] | float,
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.float64 | np.integer]]] | Sequence[float] | float,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128
def legmul(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128]]] | list[complex],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64 | _ToFloat64]]] | Sequence[complex],
) -> Array1D[np.complex128]: ...
@overload  # <=1d +c128, <=1d ~c128
def legmul(
    c1: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128 | np.complex64 | _ToFloat64]]] | Sequence[complex],
    c2: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.complex128]]] | list[complex],
) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O, <=1d
def legmul(
    c1: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    c2: _ToCoef1D,
) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d ~O
def legmul(
    c1: _ToCoef1D,
    c2: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d  (fallback)
def legmul(
    c1: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
    c2: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
) -> Array1D[Any]: ...

legdiv: Final[_FuncBinOp] = ...

# keep in sync with `polynomial.*pow`
@overload  # <=1d T
def legpow[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def legpow(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer]]] | Sequence[float] | float,
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128
def legpow(
    c: list[complex],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O
def legpow(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> Array1D[np.object_]: ...
@overload  # <=1d  (fallback)
def legpow(
    c: _ToCoef1D | _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.number | np.object_]]],
    pow: _AnyInt,
    maxpower: _IntLike_co | None = 16,
) -> Array1D[Any]: ...

# keep in sync with `polynomial.*der`
@overload  # ?d T  (workaround)
def legder[ScalarT: np.inexact](
    c: _ArrayJustND[ScalarT],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def legder(
    c: _ArrayJustND[np.integer | np.bool],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def legder(
    c: _ArrayJustND[np.object_],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def legder[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def legder(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128
def legder(
    c: list[complex],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O
def legder(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.object_]: ...
@overload  # 2d T
def legder[ScalarT: np.inexact](
    c: Array2D[ScalarT],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[ScalarT]: ...
@overload  # 2d +f64
def legder(
    c: _ToArray2D[np.integer | np.bool, float],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.float64]: ...
@overload  # 2d ~c128
def legder(
    c: Sequence[list[complex]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.complex128]: ...
@overload  # 2d ~O
def legder(
    c: Array2D[np.object_],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.object_]: ...
@overload  # ?d  (fallback)
def legder(
    c: _ToCoefND | _SupportsCoefOps[Any],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*int`
@overload  # ?d T  (workaround)
def legint[ScalarT: np.inexact](
    c: _ArrayJustND[ScalarT],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def legint(
    c: _ArrayJustND[np.integer | np.bool],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def legint(
    c: _ArrayJustND[np.object_],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def legint[ScalarT: np.inexact](
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[ScalarT]: ...
@overload  # <=1d +f64
def legint(
    c: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.float64]: ...
@overload  # <=1d ~c128
def legint(
    c: list[complex],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.complex128]: ...
@overload  # <=1d ~O
def legint(
    c: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array1D[np.object_]: ...
@overload  # 2d T
def legint[ScalarT: np.inexact](
    c: Array2D[ScalarT],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[ScalarT]: ...
@overload  # 2d +f64
def legint(
    c: _ToArray2D[np.integer | np.bool, float],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.float64]: ...
@overload  # 2d ~c128
def legint(
    c: Sequence[list[complex]],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.complex128]: ...
@overload  # 2d ~O
def legint(
    c: Array2D[np.object_],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> Array2D[np.object_]: ...
@overload  # ?d  (fallback)
def legint(
    c: _ToCoefND | _SupportsCoefOps[Any],
    m: SupportsIndex = 1,
    k: _ToCoef1D = [],
    lbnd: _NumberLike_co | _SupportsCoefOps[Any] = 0,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*val2d`
@overload  # Nd +f64, Nd +f64, 2d +f64
def legval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray2D[_AsFloat64, float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, Nd +c128, 2d ~c128
def legval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: Array2D[np.complex128] | Sequence[list[complex]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, Nd +c128, 2d +c128
def legval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, Nd ~O, 2d +O
def legval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    y: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, Nd ?, 2d ?  (fallback)
def legval2d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    y: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 0d +f64, 2d +f64
def legval2d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    c: _ToArray2D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 2d ~c128
def legval2d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    c: Array2D[np.complex128] | Sequence[list[complex]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 2d +f64
def legval2d(
    x: Sequence[float],
    y: Sequence[float],
    c: _ToArray2D[_AsFloat64, float],
) -> Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128, 2d +c128
def legval2d(
    x: list[complex],
    y: Sequence[complex],
    c: _ToArray2D[np.complex128 | _AsFloat64, complex],
) -> Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?, 2d ?  (fallback)
def legval2d(
    x: Sequence[_NumberLike_co],
    y: Sequence[_NumberLike_co],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> Array1D[Any]: ...
@overload  # ?d ?, ?d ?, ?d ?  (fallback)
def legval2d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, 1d ~O, ?d ~O
def legval2d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> Array1D[np.object_]: ...
@overload  # poly, poly, 2d ?
def legval2d[PolyT: ABCPolyBase](
    x: PolyT,
    y: PolyT,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # 0d T, 0d T, ?d ~O
def legval2d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> CoefT: ...

# keep in sync with `polynomial.*val3d`
@overload  # Nd +f64, Nd +f64, Nd +f64, 3d +f64
def legval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray3D[_AsFloat64, float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, Nd +c128, Nd +c128, 3d ~c128
def legval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, Nd +c128, Nd +c128, 3d +c128
def legval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    y: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    z: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, Nd ~O, Nd ~O, 3d +O
def legval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    y: np.ndarray[ShapeT, np.dtype[np.object_]],
    z: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, Nd ?, Nd ?, 3d ?  (fallback)
def legval3d[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    y: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    z: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 0d +f64, 0d +f64, 3d +f64
def legval3d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    z: float | _ToFloat64,
    c: _ToArray3D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 0d +c128, 3d ~c128
def legval3d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    z: complex | np.complex64 | _ToFloat64,
    c: Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 1d +f64, 3d +f64
def legval3d(
    x: Sequence[float],
    y: Sequence[float],
    z: Sequence[float],
    c: _ToArray3D[_AsFloat64, float],
) -> Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128, 1d +c128, 3d +c128
def legval3d(
    x: list[complex],
    y: Sequence[complex],
    z: Sequence[complex],
    c: _ToArray3D[np.complex128 | _AsFloat64, complex],
) -> Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?, 1d ?, 3d ?  (fallback)
def legval3d(
    x: Sequence[_NumberLike_co],
    y: Sequence[_NumberLike_co],
    z: Sequence[_NumberLike_co],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> Array1D[Any]: ...
@overload  # ?d ?, ?d ?, ?d ?, ?d ?  (fallback)
def legval3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, ?d ~O
def legval3d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    z: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> Array1D[np.object_]: ...
@overload  # 0d T, 0d T, 0d T, ?d ~O
def legval3d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    z: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> CoefT: ...

# keep in sync with `polynomial.*valnd`
@overload  # *Nd +f64, ?d +f64
def legvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[_ToFloat64]]],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # *Nd +c128, ?d ~c128
def legvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]]],
    c: _SupportsArray[np.dtype[np.complex128]] | list[complex] | _NestedSequence[list[complex]],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # *Nd ~c128, ?d +c128
def legvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.complex128]]],
    c: _SupportsArray[np.dtype[np.complex128 | np.complex64 | _ToFloat64]] | _NestedSequence[complex],
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # *Nd ~O, ?d +O
def legvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[np.object_]]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # *Nd ?, ?d ?  (fallback)
def legvalnd[ShapeT: _Shape](
    pts: Sequence[np.ndarray[ShapeT, np.dtype[_PolyScalar]]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co],
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # *0d +f64, ?d +f64
def legvalnd(
    pts: Sequence[float | _ToFloat64],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> np.float64: ...
@overload  # *0d +c128, ?d ~c128
def legvalnd(
    pts: Sequence[complex | np.complex64 | _ToFloat64],
    c: _SupportsArray[np.dtype[np.complex128]] | list[complex] | _NestedSequence[list[complex]],
) -> np.complex128: ...
@overload  # *1d +f64, ?d +f64
def legvalnd(
    pts: Sequence[Sequence[float]],
    c: _SupportsArray[np.dtype[_AsFloat64]] | _NestedSequence[float],
) -> Array1D[np.float64]: ...
@overload  # *1d ~c128, ?d +c128
def legvalnd(
    pts: Sequence[list[complex]],
    c: _SupportsArray[np.dtype[np.complex128 | _AsFloat64]] | _NestedSequence[complex],
) -> Array1D[np.complex128]: ...
@overload  # *1d ?, ?d ?  (fallback)
def legvalnd(
    pts: Sequence[Sequence[_NumberLike_co]],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co],
) -> Array1D[Any]: ...
@overload  # *poly, ?d ?
def legvalnd[PolyT: ABCPolyBase](
    pts: Sequence[PolyT],
    c: _SupportsArray[np.dtype[_PolyScalar]] | _NestedSequence[_NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # *1d ~O, ?d ~O
def legvalnd(
    pts: Sequence[Sequence[_SupportsCoefOps[Any]]],
    c: _SupportsArray[np.dtype[np.object_]] | _NestedSequence[_SupportsCoefOps[Any]],
) -> Array1D[np.object_]: ...
@overload  # *?d ?, ?d ?  (fallback)
def legvalnd(
    pts: Sequence[_ToCoefND | _SupportsCoefOps[Any]],
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...

# keep in sync with `polynomial.*val` (plus `float` in the last return)
@overload  # Nd +f64, 1d +f64
def legval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +c128, 1d ~c128
def legval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128 | np.complex64 | _ToFloat64]],
    c: Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, 1d +c128
def legval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    c: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, 1d +O
def legval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, 1d ? (fallback)
def legval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 1d +f64
def legval(
    x: float | _ToFloat64,
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.float64: ...
@overload  # 0d +c128, 1d ~c128
def legval(
    x: complex | np.complex64 | _ToFloat64,
    c: Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64
def legval(
    x: Sequence[float],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128
def legval(
    x: list[complex],
    c: _ToArray1D[np.complex128 | _AsFloat64, complex],
    tensor: bool = True,
) -> Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?  (fallback)
def legval(
    x: Sequence[_NumberLike_co],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> Array1D[Any]: ...
@overload  # ?d ?, ?d ?  (fallback)
def legval(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToCoefND,
    tensor: bool = True,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, ?d ~O
def legval(
    x: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> Array1D[np.object_]: ...
@overload  # poly, 1d ?
def legval[PolyT: ABCPolyBase](
    x: PolyT,
    c: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> PolyT: ...
@overload  # 0d T, ?d ~O
def legval[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> CoefT | float: ...

# keep in sync with `polynomial.*grid2d`
@overload  # ?d +f64, Nd +f64, 2d +f64  (workaround)
def leggrid2d(
    x: _ArrayJustND[_ToFloat64],
    y: _ToFloat64_ND,
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, ?d +f64, 2d +f64  (workaround)
def leggrid2d(
    x: _ToFloat64_ND,
    y: _ArrayJustND[_ToFloat64],
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ?, Nd ?, 2d ?  (workaround)
def leggrid2d(
    x: _ArrayJustND[_PolyScalar],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, ?d ?, 2d ?  (workaround)
def leggrid2d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayJustND[_PolyScalar],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # 0d +f64, 0d +f64, 2d +f64
def leggrid2d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    c: _ToArray2D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 2d ~c128
def leggrid2d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    c: Array2D[np.complex128] | Sequence[list[complex]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 2d +f64
def leggrid2d(
    x: _ToArray1D[_ToFloat64, float],
    y: _ToArray1D[_ToFloat64, float],
    c: _ToArray2D[_AsFloat64, float],
) -> Array2D[np.float64]: ...
@overload  # 1d +c128, 1d +c128, 2d ~c128
def leggrid2d(
    x: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: Array2D[np.complex128] | Sequence[list[complex]],
) -> Array2D[np.complex128]: ...
@overload  # 1d ~c128, 1d +c128, 2d +c128
def leggrid2d(
    x: Array1D[np.complex128] | list[complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 2d +O
def leggrid2d(
    x: Array1D[np.object_],
    y: Array1D[np.object_],
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> Array2D[np.object_]: ...
@overload  # 1d ?, 1d ?, 2d ?  (fallback)
def leggrid2d(
    x: _ToArray1D[_PolyScalar, _NumberLike_co],
    y: _ToArray1D[_PolyScalar, _NumberLike_co],
    c: _ToArray2D[_PolyScalar, _NumberLike_co],
) -> Array2D[Any]: ...
@overload  # ?d +f64, ?d +f64, 2d +f64
def leggrid2d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    c: _ToArray2D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d +c128, ?d +c128, 2d ~c128
def leggrid2d(
    x: _ToComplex128_ND,
    y: _ToComplex128_ND,
    c: Array2D[np.complex128] | Sequence[list[complex]],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~c128, ?d +c128, 2d +c128
def leggrid2d(
    x: np.ndarray[Any, np.dtype[np.complex128]] | _NestedSequence[list[complex]] | list[complex],
    y: _ToComplex128_ND,
    c: _ToArray2D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~O, ?d ~O, 2d +O
def leggrid2d(
    x: _ArrayLikeObject_co,
    y: _ArrayLikeObject_co,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[np.object_]: ...
@overload  # 1d ~O, 1d ~O, ?d ~O
def leggrid2d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> Array2D[np.object_]: ...
@overload  # poly, poly, 2d ?
def leggrid2d[PolyT: ABCPolyBase](
    x: PolyT,
    y: PolyT,
    c: _ToArray2D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> PolyT: ...
@overload  # ?d ?, ?d ?, ?d ?  (fallback)
def leggrid2d(
    x: _ToCoefND,
    y: _ToCoefND,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 0d T, 0d T, ?d ~O
def leggrid2d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[_SupportsCoefOps[Any]]],
) -> CoefT: ...

# keep in sync with `polynomial.*grid3d`
@overload  # ?d +f64, Nd +f64, Nd +f64, 3d +f64  (workaround)
def leggrid3d(
    x: _ArrayJustND[_ToFloat64],
    y: _ToFloat64_ND,
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, ?d +f64, Nd +f64, 3d +f64  (workaround)
def leggrid3d(
    x: _ToFloat64_ND,
    y: _ArrayJustND[_ToFloat64],
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # Nd +f64, Nd +f64, ?d +f64, 3d +f64  (workaround)
def leggrid3d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    z: _ArrayJustND[_ToFloat64],
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ?, Nd ?, Nd ?, 3d ?  (workaround)
def leggrid3d(
    x: _ArrayJustND[_PolyScalar],
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, ?d ?, Nd ?, 3d ?  (workaround)
def leggrid3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayJustND[_PolyScalar],
    z: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # Nd ?, Nd ?, ?d ?, 3d ?  (workaround)
def leggrid3d(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    y: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    z: _ArrayJustND[_PolyScalar],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[Any]: ...
@overload  # 0d +f64, 0d +f64, 0d +f64, 3d +f64
def leggrid3d(
    x: float | _ToFloat64,
    y: float | _ToFloat64,
    z: float | _ToFloat64,
    c: _ToArray3D[_AsFloat64, float],
) -> np.float64: ...
@overload  # 0d +c128, 0d +c128, 0d +c128, 3d ~c128
def leggrid3d(
    x: complex | np.complex64 | _ToFloat64,
    y: complex | np.complex64 | _ToFloat64,
    z: complex | np.complex64 | _ToFloat64,
    c: Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64, 1d +f64, 3d +f64
def leggrid3d(
    x: _ToArray1D[_ToFloat64, float],
    y: _ToArray1D[_ToFloat64, float],
    z: _ToArray1D[_ToFloat64, float],
    c: _ToArray3D[_AsFloat64, float],
) -> Array3D[np.float64]: ...
@overload  # 1d +c128, 1d +c128, 1d +c128, 3d ~c128
def leggrid3d(
    x: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    z: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> Array3D[np.complex128]: ...
@overload  # 1d ~c128, 1d +c128, 1d +c128, 3d +c128
def leggrid3d(
    x: Array1D[np.complex128] | list[complex],
    y: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    z: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> Array3D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, 3d +O
def leggrid3d(
    x: Array1D[np.object_],
    y: Array1D[np.object_],
    z: Array1D[np.object_],
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> Array3D[np.object_]: ...
@overload  # 1d ?, 1d ?, 1d ?, 3d ?  (fallback)
def leggrid3d(
    x: _ToArray1D[_PolyScalar, _NumberLike_co],
    y: _ToArray1D[_PolyScalar, _NumberLike_co],
    z: _ToArray1D[_PolyScalar, _NumberLike_co],
    c: _ToArray3D[_PolyScalar, _NumberLike_co],
) -> Array3D[Any]: ...
@overload  # ?d +f64, ?d +f64, ?d +f64, 3d +f64
def leggrid3d(
    x: _ToFloat64_ND,
    y: _ToFloat64_ND,
    z: _ToFloat64_ND,
    c: _ToArray3D[_AsFloat64, float],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d +c128, ?d +c128, ?d +c128, 3d ~c128
def leggrid3d(
    x: _ToComplex128_ND,
    y: _ToComplex128_ND,
    z: _ToComplex128_ND,
    c: Array3D[np.complex128] | Sequence[Sequence[list[complex]]],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~c128, ?d +c128, ?d +c128, 3d +c128
def leggrid3d(
    x: np.ndarray[Any, np.dtype[np.complex128]] | _NestedSequence[list[complex]] | list[complex],
    y: _ToComplex128_ND,
    z: _ToComplex128_ND,
    c: _ToArray3D[np.complex128 | np.complex64 | _ToFloat64, complex],
) -> npt.NDArray[np.complex128]: ...
@overload  # ?d ~O, ?d ~O, ?d ~O, 3d +O
def leggrid3d(
    x: _ArrayLikeObject_co,
    y: _ArrayLikeObject_co,
    z: _ArrayLikeObject_co,
    c: _ToArray3D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
) -> npt.NDArray[np.object_]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O, ?d ~O
def leggrid3d(
    x: Sequence[_SupportsCoefOps[Any]],
    y: Sequence[_SupportsCoefOps[Any]],
    z: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> Array3D[np.object_]: ...
@overload  # ?d ?, ?d ?, ?d ?, ?d ?  (fallback)
def leggrid3d(
    x: _ToCoefND,
    y: _ToCoefND,
    z: _ToCoefND,
    c: _ToCoefND,
) -> npt.NDArray[Any] | Any: ...
@overload  # 0d T, 0d T, 0d T, ?d ~O
def leggrid3d[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    y: CoefT,
    z: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[Sequence[Sequence[_SupportsCoefOps[Any]]]],
) -> CoefT: ...

# keep in sync with `polynomial.*vander`
@overload  # ?d T  (workaround)
def legvander[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    deg: SupportsIndex,
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64  (workaround)
def legvander(
    x: _ArrayJustND[np.integer | np.bool],
    deg: SupportsIndex,
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O  (workaround)
def legvander(
    x: _ArrayJustND[np.object_],
    deg: SupportsIndex,
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T
def legvander[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: SupportsIndex,
) -> Array2D[ScalarT]: ...
@overload  # <=1d +f64
def legvander(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[np.integer | np.bool]]] | Sequence[float] | float,
    deg: SupportsIndex,
) -> Array2D[np.float64]: ...
@overload  # <=1d ~c128
def legvander(
    x: list[complex],
    deg: SupportsIndex,
) -> Array2D[np.complex128]: ...
@overload  # <=1d ~O
def legvander(
    x: np.ndarray[tuple[()] | tuple[int], np.dtype[np.object_]],
    deg: SupportsIndex,
) -> Array2D[np.object_]: ...
@overload  # 2d T
def legvander[ScalarT: np.inexact](
    x: Array2D[ScalarT],
    deg: SupportsIndex,
) -> Array3D[ScalarT]: ...
@overload  # 2d +f64
def legvander(
    x: _ToArray2D[np.integer | np.bool, float],
    deg: SupportsIndex,
) -> Array3D[np.float64]: ...
@overload  # 2d ~c128
def legvander(
    x: Sequence[list[complex]],
    deg: SupportsIndex,
) -> Array3D[np.complex128]: ...
@overload  # 2d ~O
def legvander(
    x: Array2D[np.object_],
    deg: SupportsIndex,
) -> Array3D[np.object_]: ...
@overload  # ?d  (fallback)
def legvander(
    x: _ToCoefND | _SupportsCoefOps[Any],
    deg: SupportsIndex,
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*vander2d`
@overload  # ?d T, ?d T  (workaround)
def legvander2d[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    y: _ArrayJustND[ScalarT],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64, ?d +f64  (workaround)
def legvander2d(
    x: _ArrayJustND[_AsFloat64],
    y: _ArrayJustND[_AsFloat64],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O, ?d ~O  (workaround)
def legvander2d(
    x: _ArrayJustND[np.object_],
    y: _ArrayJustND[np.object_],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T, <=1d T
def legvander2d[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: Sequence[SupportsIndex],
) -> Array2D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64
def legvander2d(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    deg: Sequence[SupportsIndex],
) -> Array2D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128
def legvander2d(
    x: list[complex],
    y: Sequence[complex] | complex,
    deg: Sequence[SupportsIndex],
) -> Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O
def legvander2d(
    x: Array1D[np.object_],
    y: Array1D[np.object_],
    deg: Sequence[SupportsIndex],
) -> Array2D[np.object_]: ...
@overload  # 2d T, 2d T
def legvander2d[ScalarT: np.inexact](
    x: Array2D[ScalarT],
    y: Array2D[ScalarT],
    deg: Sequence[SupportsIndex],
) -> Array3D[ScalarT]: ...
@overload  # 2d +f64, 2d +f64
def legvander2d(
    x: _ToArray2D[_AsFloat64, float],
    y: _ToArray2D[_AsFloat64, float],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.float64]: ...
@overload  # 2d ~c128, 2d +c128
def legvander2d(
    x: Sequence[list[complex]],
    y: Sequence[Sequence[complex]],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.complex128]: ...
@overload  # 2d ~O, 2d ~O
def legvander2d(
    x: Array2D[np.object_],
    y: Array2D[np.object_],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.object_]: ...
@overload  # ?d, ?d  (fallback)
def legvander2d(
    x: _ToCoefND | _SupportsCoefOps[Any],
    y: _ToCoefND | _SupportsCoefOps[Any],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*vander3d`
@overload  # ?d T, ?d T, ?d T  (workaround)
def legvander3d[ScalarT: np.inexact](
    x: _ArrayJustND[ScalarT],
    y: _ArrayJustND[ScalarT],
    z: _ArrayJustND[ScalarT],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[ScalarT]: ...
@overload  # ?d +f64, ?d +f64, ?d +f64  (workaround)
def legvander3d(
    x: _ArrayJustND[_AsFloat64],
    y: _ArrayJustND[_AsFloat64],
    z: _ArrayJustND[_AsFloat64],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.float64]: ...
@overload  # ?d ~O, ?d ~O, ?d ~O  (workaround)
def legvander3d(
    x: _ArrayJustND[np.object_],
    y: _ArrayJustND[np.object_],
    z: _ArrayJustND[np.object_],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[np.object_]: ...
@overload  # <=1d T, <=1d T, <=1d T
def legvander3d[ScalarT: np.inexact](
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    z: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[ScalarT]]],
    deg: Sequence[SupportsIndex],
) -> Array2D[ScalarT]: ...
@overload  # <=1d +f64, <=1d +f64, <=1d +f64
def legvander3d(
    x: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    y: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    z: _CanArray[np.ndarray[tuple[()] | tuple[int], np.dtype[_AsFloat64]]] | Sequence[float] | float,
    deg: Sequence[SupportsIndex],
) -> Array2D[np.float64]: ...
@overload  # <=1d ~c128, <=1d +c128, <=1d +c128
def legvander3d(
    x: list[complex],
    y: Sequence[complex] | complex,
    z: Sequence[complex] | complex,
    deg: Sequence[SupportsIndex],
) -> Array2D[np.complex128]: ...
@overload  # 1d ~O, 1d ~O, 1d ~O
def legvander3d(
    x: Array1D[np.object_],
    y: Array1D[np.object_],
    z: Array1D[np.object_],
    deg: Sequence[SupportsIndex],
) -> Array2D[np.object_]: ...
@overload  # 2d T, 2d T, 2d T
def legvander3d[ScalarT: np.inexact](
    x: Array2D[ScalarT],
    y: Array2D[ScalarT],
    z: Array2D[ScalarT],
    deg: Sequence[SupportsIndex],
) -> Array3D[ScalarT]: ...
@overload  # 2d +f64, 2d +f64, 2d +f64
def legvander3d(
    x: _ToArray2D[_AsFloat64, float],
    y: _ToArray2D[_AsFloat64, float],
    z: _ToArray2D[_AsFloat64, float],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.float64]: ...
@overload  # 2d ~c128, 2d +c128, 2d +c128
def legvander3d(
    x: Sequence[list[complex]],
    y: Sequence[Sequence[complex]],
    z: Sequence[Sequence[complex]],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.complex128]: ...
@overload  # 2d ~O, 2d ~O, 2d ~O
def legvander3d(
    x: Array2D[np.object_],
    y: Array2D[np.object_],
    z: Array2D[np.object_],
    deg: Sequence[SupportsIndex],
) -> Array3D[np.object_]: ...
@overload  # ?d, ?d, ?d  (fallback)
def legvander3d(
    x: _ToCoefND | _SupportsCoefOps[Any],
    y: _ToCoefND | _SupportsCoefOps[Any],
    z: _ToCoefND | _SupportsCoefOps[Any],
    deg: Sequence[SupportsIndex],
) -> npt.NDArray[Any]: ...

# keep in sync with `polynomial.*fit`
@overload  # Nd +f64
def legfit[ShapeT: _Shape](
    x: _ToArray1D[_ToFloat64, float],
    y: np.ndarray[ShapeT, np.dtype[_AsFloat64]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f64, full=True
def legfit[ShapeT: _Shape](
    x: _ToArray1D[_ToFloat64, float],
    y: np.ndarray[ShapeT, np.dtype[_AsFloat64]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[np.ndarray[ShapeT, np.dtype[np.float64]], list[Any]]: ...
@overload  # 1d +f64
def legfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> Array1D[np.float64]: ...
@overload  # 1d +f64, full=True
def legfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[Array1D[np.float64], list[Any]]: ...
@overload  # 2d +f64
def legfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> Array2D[np.float64]: ...
@overload  # 2d +f64, full=True
def legfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[Array2D[np.float64], list[Any]]: ...
@overload  # Nd
def legfit[ShapeT: _Shape](
    x: _ToComplex128_1D,
    y: np.ndarray[ShapeT, np.dtype[np.number | np.bool]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToComplex128_1D | None = None,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # Nd, full=True
def legfit[ShapeT: _Shape](
    x: _ToComplex128_1D,
    y: np.ndarray[ShapeT, np.dtype[np.number | np.bool]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToComplex128_1D | None = None,
) -> tuple[np.ndarray[ShapeT, np.dtype[Any]], list[Any]]: ...
@overload  # ?d  (fallback)
def legfit(
    x: _ToComplex128_1D,
    y: _ArrayLikeNumber_co,
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToComplex128_1D | None = None,
) -> npt.NDArray[Any]: ...
@overload  # ?d, full=True
def legfit(
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
def legcompanion[ScalarT: np.inexact](c: _CanArray[Array1D[ScalarT]]) -> Array2D[ScalarT]: ...
@overload  # 1d +f64
def legcompanion(c: _CanArray[Array1D[np.integer]] | Sequence[float]) -> Array2D[np.float64]: ...
@overload  # 1d ~c128
def legcompanion(c: list[complex]) -> Array2D[np.complex128]: ...
@overload  # 1d  (fallback)
def legcompanion(c: _CanArray[Array1D[_PolyScalar]] | Sequence[_NumberLike_co | _SupportsCoefOps[Any]]) -> Array2D[Any]: ...

# keep in sync with `polynomial.*roots`
@overload  # 1d T
def legroots[ScalarT: np.complexfloating](c: _CanArray[Array1D[ScalarT]] | Sequence[ScalarT]) -> Array1D[ScalarT]: ...
@overload  # 1d ~f32
def legroots(c: _CanArray[Array1D[np.float32]] | Sequence[np.float32]) -> Array1D[np.float32 | np.complex64]: ...
@overload  # 1d +f64
def legroots(c: _CanArray[Array1D[np.float64 | np.integer]] | Sequence[float]) -> Array1D[np.float64 | np.complex128]: ...
@overload  # 1d ~c128
def legroots(c: list[complex]) -> Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def legroots(c: _CanArray[Array1D[_PolyScalar]] | Sequence[_NumberLike_co | _SupportsCoefOps[Any]]) -> Array1D[Any]: ...

#
def leggauss(deg: SupportsIndex) -> tuple[Array1D[np.float64], Array1D[np.float64]]: ...

@overload  # Nd T
def legweight[ShapeT: _Shape, ScalarT: np.inexact](
    x: np.ndarray[ShapeT, np.dtype[ScalarT]],
) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload  # Nd +f64
def legweight[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.integer | np.bool]],
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd ~O
def legweight[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # 0d T
def legweight[ScalarT: np.inexact](x: ScalarT) -> ScalarT: ...
@overload  # 0d +f64
def legweight(x: np.integer | np.bool) -> np.float64: ...
@overload  # 0d ~float
def legweight(x: float) -> float: ...
@overload  # 0d ~complex
def legweight(x: complex) -> complex: ...

class Legendre(ABCPolyBase[L["P"]]):
    basis_name: ClassVar[L["P"]] = "P"  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]
    domain: Array1D[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
    window: Array1D[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
