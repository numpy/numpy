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
    _FuncWeight,
    _PolyScalar,
    _SupportsCoefOps,
)
from .polyutils import trimcoef as hermtrim

__all__ = [
    "hermzero",
    "hermone",
    "hermx",
    "hermdomain",
    "hermline",
    "hermadd",
    "hermsub",
    "hermmulx",
    "hermmul",
    "hermdiv",
    "hermpow",
    "hermval",
    "hermder",
    "hermint",
    "herm2poly",
    "poly2herm",
    "hermfromroots",
    "hermvander",
    "hermfit",
    "hermtrim",
    "hermroots",
    "Hermite",
    "hermval2d",
    "hermval3d",
    "hermvalnd",
    "hermgrid2d",
    "hermgrid3d",
    "hermvander2d",
    "hermvander3d",
    "hermcompanion",
    "hermgauss",
    "hermweight",
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

poly2herm: Final[_FuncPoly2Ortho] = ...
herm2poly: Final[_FuncUnOp] = ...

hermdomain: Final[_Array2[np.float64]] = ...
hermzero: Final[_Array1[np.int_]] = ...
hermone: Final[_Array1[np.int_]] = ...
hermx: Final[_Array2[np.int_]] = ...

hermline: Final[_FuncLine] = ...
hermfromroots: Final[_FuncFromRoots] = ...
hermadd: Final[_FuncBinOp] = ...
hermsub: Final[_FuncBinOp] = ...
hermmulx: Final[_FuncUnOp] = ...
hermmul: Final[_FuncBinOp] = ...
hermdiv: Final[_FuncBinOp] = ...
hermpow: Final[_FuncPow] = ...
hermder: Final[_FuncDer] = ...
hermint: Final[_FuncInteg] = ...
hermval2d: Final[_FuncVal2D] = ...
hermval3d: Final[_FuncVal3D] = ...
hermvalnd: Final[_FuncValND] = ...

# keep in sync with `polynomial.*val`
@overload  # Nd +f64, 1d +f64
def hermval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f64, 1d ~c128
def hermval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_ToFloat64]],
    c: _Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~c128, 1d +c128
def hermval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.complex128]],
    c: _ToArray1D[np.complex128 | np.complex64 | _ToFloat64, complex],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.complex128]]: ...
@overload  # Nd ~O, 1d +O
def hermval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    c: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
@overload  # Nd ?, 1d ? (fallback)
def hermval[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[_PolyScalar]],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # 0d +f64, 1d +f64
def hermval(
    x: float | _ToFloat64,
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> np.float64: ...
@overload  # 0d +c128, 1d ~c128
def hermval(
    x: complex | np.complex64 | _ToFloat64,
    c: _Array1D[np.complex128] | list[complex],
    tensor: bool = True,
) -> np.complex128: ...
@overload  # 1d +f64, 1d +f64
def hermval(
    x: Sequence[float],
    c: _ToArray1D[_AsFloat64, float],
    tensor: bool = True,
) -> _Array1D[np.float64]: ...
@overload  # 1d ~c128, 1d +c128
def hermval(
    x: list[complex],
    c: _ToArray1D[np.complex128 | _AsFloat64, complex],
    tensor: bool = True,
) -> _Array1D[np.complex128]: ...
@overload  # 1d ?, 1d ?  (fallback)
def hermval(
    x: Sequence[_NumberLike_co],
    c: _ToArray1D[_PolyScalar, _NumberLike_co],
    tensor: bool = True,
) -> _Array1D[Any]: ...
@overload  # ?d ?, ?d ?  (fallback)
def hermval(
    x: _ArrayLikeNumber_co | _ArrayLikeObject_co,
    c: _ArrayLikeNumber_co | _ArrayLikeObject_co | _NestedSequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> npt.NDArray[Any] | Any: ...
@overload  # 1d ~O, ?d ~O
def hermval(
    x: Sequence[_SupportsCoefOps[Any]],
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> _Array1D[np.object_]: ...
@overload  # poly, 1d ?
def hermval[PolyT: ABCPolyBase](
    x: PolyT,
    c: _ToArray1D[_PolyScalar, _NumberLike_co | _SupportsCoefOps[Any]],
    tensor: bool = True,
) -> PolyT: ...
@overload  # 0d T, ?d ~O
def hermval[CoefT: _SupportsCoefOps[Any]](
    x: CoefT,
    c: _SupportsArray[np.dtype[np.object_]] | Sequence[_SupportsCoefOps[Any]],
    tensor: bool = True,
) -> CoefT: ...

hermgrid2d: Final[_FuncVal2D] = ...
hermgrid3d: Final[_FuncVal3D] = ...
hermvander: Final[_FuncVander] = ...
hermvander2d: Final[_FuncVander2D] = ...
hermvander3d: Final[_FuncVander3D] = ...

# keep in sync with `polynomial.*fit`
@overload  # Nd +f64
def hermfit[ShapeT: _Shape](
    x: _ToArray1D[_ToFloat64, float],
    y: np.ndarray[ShapeT, np.dtype[_AsFloat64]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
@overload  # Nd +f64, full=True
def hermfit[ShapeT: _Shape](
    x: _ToArray1D[_ToFloat64, float],
    y: np.ndarray[ShapeT, np.dtype[_AsFloat64]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[np.ndarray[ShapeT, np.dtype[np.float64]], list[Any]]: ...
@overload  # 1d +f64
def hermfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> _Array1D[np.float64]: ...
@overload  # 1d +f64, full=True
def hermfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[float],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[_Array1D[np.float64], list[Any]]: ...
@overload  # 2d +f64
def hermfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> _Array2D[np.float64]: ...
@overload  # 2d +f64, full=True
def hermfit(
    x: _ToArray1D[_ToFloat64, float],
    y: Sequence[Sequence[float]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToArray1D[_ToFloat64, float] | None = None,
) -> tuple[_Array2D[np.float64], list[Any]]: ...
@overload  # Nd
def hermfit[ShapeT: _Shape](
    x: _ToComplex128_1D,
    y: np.ndarray[ShapeT, np.dtype[np.number | np.bool]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToComplex128_1D | None = None,
) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
@overload  # Nd, full=True
def hermfit[ShapeT: _Shape](
    x: _ToComplex128_1D,
    y: np.ndarray[ShapeT, np.dtype[np.number | np.bool]],
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToComplex128_1D | None = None,
) -> tuple[np.ndarray[ShapeT, np.dtype[Any]], list[Any]]: ...
@overload  # ?d  (fallback)
def hermfit(
    x: _ToComplex128_1D,
    y: _ArrayLikeNumber_co,
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ToComplex128_1D | None = None,
) -> npt.NDArray[Any]: ...
@overload  # ?d, full=True
def hermfit(
    x: _ToComplex128_1D,
    y: _ArrayLikeNumber_co,
    deg: SupportsIndex | _ToInt_1D,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ToComplex128_1D | None = None,
) -> tuple[npt.NDArray[Any], list[Any]]: ...

hermcompanion: Final[_FuncCompanion] = ...
hermroots: Final[_FuncRoots] = ...

def _normed_hermite_n[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.float64]],
    n: int,
) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...

hermgauss: Final[_FuncGauss] = ...
hermweight: Final[_FuncWeight] = ...

class Hermite(ABCPolyBase[L["H"]]):
    basis_name: ClassVar[L["H"]] = "H"  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]
    domain: _Array2[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
    window: _Array2[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
