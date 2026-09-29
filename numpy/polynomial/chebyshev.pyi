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
    _Array1,
    _Array2,
    _CanArray,
    _CoefSeries,
    _FuncBinOp,
    _FuncCompanion,
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
    _FuncVander2D,
    _FuncVander3D,
    _PolyScalar,
    _Series,
    _SeriesLikeCoef_co,
    _SupportsCoefOps,
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

type _AsFloat64 = np.float64 | np.integer | np.bool
type _ToFloat64 = np.float64 | np.float32 | np.float16 | np.integer | np.bool

type _ToComplex128_1D = _SupportsArray[np.dtype[np.number | np.bool]] | Sequence[_NumberLike_co]
type _ToInt_1D = _SupportsArray[np.dtype[np.integer]] | Sequence[SupportsIndex]

###

def _cseries_to_zseries[ScalarT: np.number | np.object_](c: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...
def _zseries_to_cseries[ScalarT: np.number | np.object_](zs: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...
def _zseries_mul[ScalarT: np.number | np.object_](z1: npt.NDArray[ScalarT], z2: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...
def _zseries_div[ScalarT: np.number | np.object_](z1: npt.NDArray[ScalarT], z2: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...
def _zseries_der[ScalarT: np.number | np.object_](zs: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...
def _zseries_int[ScalarT: np.number | np.object_](zs: npt.NDArray[ScalarT]) -> _Series[ScalarT]: ...

poly2cheb: Final[_FuncPoly2Ortho] = ...
cheb2poly: Final[_FuncUnOp] = ...

chebdomain: Final[_Array2[np.float64]] = ...
chebzero: Final[_Array1[np.int_]] = ...
chebone: Final[_Array1[np.int_]] = ...
chebx: Final[_Array2[np.int_]] = ...

chebline: Final[_FuncLine] = ...
chebfromroots: Final[_FuncFromRoots] = ...
chebadd: Final[_FuncBinOp] = ...
chebsub: Final[_FuncBinOp] = ...
chebmulx: Final[_FuncUnOp] = ...
chebmul: Final[_FuncBinOp] = ...
chebdiv: Final[_FuncBinOp] = ...
chebpow: Final[_FuncPow] = ...

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
    c: _Array2D[np.integer | np.bool] | Sequence[Sequence[float]],
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
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _SupportsCoefOps[Any] | _NestedSequence[_SupportsCoefOps[Any]],
    m: SupportsIndex = 1,
    scl: _NumberLike_co | _SupportsCoefOps[Any] = 1,
    axis: SupportsIndex = 0,
) -> npt.NDArray[Any]: ...

chebint: Final[_FuncInteg] = ...
chebval2d: Final[_FuncVal2D] = ...
chebval3d: Final[_FuncVal3D] = ...
chebvalnd: Final[_FuncValND] = ...

# keep in sync with `polynomial.*val`
@overload  # Nd +f64, 1d +f64
def chebval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f64, 1d ~c128
def chebval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
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
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
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

chebgrid2d: Final[_FuncVal2D] = ...
chebgrid3d: Final[_FuncVal3D] = ...

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
    x: _Array2D[np.integer | np.bool] | Sequence[Sequence[float]],
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
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co | _SupportsCoefOps[Any] | _NestedSequence[_SupportsCoefOps[Any]],
    deg: SupportsIndex,
) -> npt.NDArray[Any]: ...

chebvander2d: Final[_FuncVander2D] = ...
chebvander3d: Final[_FuncVander3D] = ...

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

chebcompanion: Final[_FuncCompanion] = ...
chebroots: Final[_FuncRoots] = ...
chebgauss: Final[_FuncGauss] = ...

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
    domain: _Array2[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
    window: _Array2[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]

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
