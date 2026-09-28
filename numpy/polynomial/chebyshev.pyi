from _typeshed import ConvertibleToInt
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, Concatenate, Final, Literal as L, Self, overload

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
    _CoefSeries,
    _FuncBinOp,
    _FuncCompanion,
    _FuncDer,
    _FuncFit,
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
    _FuncWeight,
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

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _ToArray1D[ScalarT: np.generic, T] = _Array1D[ScalarT] | Sequence[T]

type _AsFloat64 = np.float64 | np.integer | np.bool
type _ToFloat64 = np.float64 | np.float32 | np.float16 | np.integer | np.bool

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
chebder: Final[_FuncDer] = ...
chebint: Final[_FuncInteg] = ...
chebval2d: Final[_FuncVal2D] = ...
chebval3d: Final[_FuncVal3D] = ...
chebvalnd: Final[_FuncValND] = ...

# keep in sync with `.polynomial.polyval`, `.legendre.legval`
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
chebvander: Final[_FuncVander] = ...
chebvander2d: Final[_FuncVander2D] = ...
chebvander3d: Final[_FuncVander3D] = ...
chebfit: Final[_FuncFit] = ...
chebcompanion: Final[_FuncCompanion] = ...
chebroots: Final[_FuncRoots] = ...
chebgauss: Final[_FuncGauss] = ...
chebweight: Final[_FuncWeight] = ...
def chebpts1(npts: ConvertibleToInt) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...
def chebpts2(npts: ConvertibleToInt) -> np.ndarray[tuple[int], np.dtype[np.float64]]: ...

# keep in sync with `Chebyshev.interpolate` (minus `domain` parameter)
@overload
def chebinterpolate(
    func: np.ufunc,
    deg: _IntLike_co,
    args: tuple[()] = (),
) -> npt.NDArray[np.float64 | np.complex128 | np.object_]: ...
@overload
def chebinterpolate[CoefScalarT: np.number | np.bool | np.object_](
    func: Callable[[npt.NDArray[np.float64]], CoefScalarT],
    deg: _IntLike_co,
    args: tuple[()] = (),
) -> npt.NDArray[CoefScalarT]: ...
@overload
def chebinterpolate[CoefScalarT: np.number | np.bool | np.object_](
    func: Callable[Concatenate[npt.NDArray[np.float64], ...], CoefScalarT],
    deg: _IntLike_co,
    args: Iterable[Any],
) -> npt.NDArray[CoefScalarT]: ...

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
