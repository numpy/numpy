from collections.abc import Sequence
from typing import Any, ClassVar, Final, overload

import numpy as np
import numpy.typing as npt
from numpy._typing import (
    _ArrayLikeFloat_co,
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _FloatLike_co,
    _NestedSequence,
    _NumberLike_co,
    _Shape,
    _SupportsArray,
)

from ._polybase import ABCPolyBase
from ._polytypes import (
    _Array1,
    _Array2,
    _ArrayLikeCoef_co,
    _FuncBinOp,
    _FuncCompanion,
    _FuncDer,
    _FuncFit,
    _FuncFromRoots,
    _FuncInteg,
    _FuncLine,
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
type _ToArray1D[ScalarT: np.generic, T] = _Array1D[ScalarT] | Sequence[T]

type _AsFloat64 = np.float64 | np.integer | np.bool
type _ToFloat64 = np.float64 | np.float32 | np.float16 | np.integer | np.bool

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

#
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

#
@overload
def polyvalfromroots(x: _FloatLike_co, r: _FloatLike_co, tensor: bool = True) -> np.float64 | Any: ...
@overload
def polyvalfromroots(x: _NumberLike_co, r: _NumberLike_co, tensor: bool = True) -> np.complex128 | Any: ...
@overload
def polyvalfromroots(x: _ArrayLikeFloat_co, r: _ArrayLikeFloat_co, tensor: bool = True) -> npt.NDArray[np.float64 | Any]: ...
@overload
def polyvalfromroots(x: _ArrayLikeNumber_co, r: _ArrayLikeNumber_co, tensor: bool = True) -> npt.NDArray[np.complex128 | Any]: ...
@overload
def polyvalfromroots(x: _ArrayLikeCoef_co, r: _ArrayLikeCoef_co, tensor: bool = True) -> npt.NDArray[np.object_ | Any]: ...

polygrid2d: Final[_FuncVal2D] = ...
polygrid3d: Final[_FuncVal3D] = ...
polyvander: Final[_FuncVander] = ...
polyvander2d: Final[_FuncVander2D] = ...
polyvander3d: Final[_FuncVander3D] = ...
polyfit: Final[_FuncFit] = ...
polycompanion: Final[_FuncCompanion] = ...
polyroots: Final[_FuncRoots] = ...

class Polynomial(ABCPolyBase[None]):
    basis_name: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride] # pyrefly: ignore[bad-override]
    domain: _Array2[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
    window: _Array2[np.float64 | Any] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
