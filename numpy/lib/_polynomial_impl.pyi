from _typeshed import ConvertibleToInt, Incomplete
from collections.abc import Iterator, Sequence
from typing import (
    Any,
    ClassVar,
    Literal as L,
    Never,
    NoReturn,
    Self,
    SupportsIndex,
    SupportsInt,
    TypeVar,
    overload,
    override,
)

import numpy as np
from numpy import complex128, float64, floating, int32
from numpy._typing import (
    ArrayLike,
    NDArray,
    _AnyShape,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex128_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat64_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ComplexLike_co,
    _FloatLike_co,
    _IntLike_co,
    _NestedSequence,
    _ScalarLike_co,
)

type _Int_co = np.integer | np.bool
type _Float_co = np.floating | np.integer | np.bool
type _Number_co = np.number | np.bool

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]

# workaround for mypy and pyright not following the typing spec for overloads
type _ArrayJustND[ScalarT: np.generic] = np.ndarray[tuple[Never, Never, Never, Never], np.dtype[ScalarT]]

type _2Tup[T] = tuple[T, T]
type _5Tup[T] = tuple[T, _Array1D[float64], int32, _Array1D[float64], float | floating]

_AnyNumberT = TypeVar(
    "_AnyNumberT",
    np.bool,
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64, np.longdouble,
    np.complex64, np.complex128, np.clongdouble,
    np.object_,
)
_ShapeT = TypeVar("_ShapeT", bound=_AnyShape)

###

__all__ = [
    "poly",
    "roots",
    "polyint",
    "polyder",
    "polyadd",
    "polysub",
    "polymul",
    "polydiv",
    "polyval",
    "poly1d",
    "polyfit",
]

class poly1d:
    __module__: L["numpy"] = "numpy"  # pyrefly: ignore[bad-override]

    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    def variable(self) -> str: ...
    @property
    def order(self) -> int: ...
    @property
    def o(self) -> int: ...
    @property
    def roots(self) -> NDArray[Incomplete]: ...
    @property
    def r(self) -> NDArray[Incomplete]: ...

    #
    @property
    def coeffs(self) -> NDArray[Incomplete]: ...
    @coeffs.setter
    def coeffs(self, value: NDArray[Incomplete], /) -> None: ...

    #
    @property
    def c(self) -> NDArray[Any]: ...
    @c.setter
    def c(self, value: NDArray[Incomplete], /) -> None: ...

    #
    @property
    def coef(self) -> NDArray[Incomplete]: ...
    @coef.setter
    def coef(self, value: NDArray[Incomplete], /) -> None: ...

    #
    @property
    def coefficients(self) -> NDArray[Incomplete]: ...
    @coefficients.setter
    def coefficients(self, value: NDArray[Incomplete], /) -> None: ...

    #
    def __init__(self, /, c_or_r: ArrayLike, r: bool = False, variable: str | None = None) -> None: ...

    #
    @overload
    def __array__(self, /, t: None = None, copy: bool | None = None) -> np.ndarray[tuple[int], np.dtype[Incomplete]]: ...
    @overload
    def __array__[DTypeT: np.dtype](self, /, t: DTypeT, copy: bool | None = None) -> np.ndarray[tuple[int], DTypeT]: ...

    #
    @overload
    def __call__(self, /, val: _ScalarLike_co) -> Incomplete: ...
    @overload
    def __call__(self, /, val: poly1d) -> Self: ...
    @overload
    def __call__(self, /, val: NDArray[Incomplete] | _NestedSequence[_ScalarLike_co]) -> NDArray[Incomplete]: ...

    #
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Incomplete]: ...

    #
    def __getitem__(self, val: int, /) -> Incomplete: ...
    def __setitem__(self, key: int, val: Incomplete, /) -> None: ...

    def __neg__(self) -> Self: ...
    def __pos__(self) -> Self: ...

    #
    def __add__(self, other: ArrayLike, /) -> Self: ...
    def __radd__(self, other: ArrayLike, /) -> Self: ...

    #
    def __sub__(self, other: ArrayLike, /) -> Self: ...
    def __rsub__(self, other: ArrayLike, /) -> Self: ...

    #
    def __mul__(self, other: ArrayLike, /) -> Self: ...
    def __rmul__(self, other: ArrayLike, /) -> Self: ...

    #
    def __pow__(self, val: _FloatLike_co, /) -> Self: ...  # Integral floats are accepted

    #
    def __truediv__(self, other: ArrayLike, /) -> Self: ...
    def __rtruediv__(self, other: ArrayLike, /) -> Self: ...

    #
    @override
    def __eq__(self, other: poly1d, /) -> bool: ...  # type:ignore[override]
    @override
    def __ne__(self, other: poly1d, /) -> bool: ...  # type:ignore[override]

    #
    def deriv(self, /, m: ConvertibleToInt = 1) -> Self: ...
    def integ(self, /, m: ConvertibleToInt = 1, k: _ArrayLikeComplex_co | _ArrayLikeObject_co | None = 0) -> poly1d: ...

#
@overload  # <=2d Any  (workaround)
def poly(seq_of_zeros: NDArray[np.inexact[Never]] | poly1d) -> _Array1D[Any]: ...
@overload  # <=2d ~f32
def poly(seq_of_zeros: NDArray[np.float32] | Sequence[np.float32]) -> _Array1D[np.float32]: ...
@overload  # <=2d ~c64
def poly(seq_of_zeros: NDArray[np.complex64] | Sequence[np.complex64]) -> _Array1D[np.float32 | np.complex64]: ...
@overload  # <=2d +f64
def poly(seq_of_zeros: NDArray[np.float64 | _Int_co] | Sequence[float | _Int_co]) -> _Array1D[np.float64]: ...
@overload  # <=2d +c128
def poly(seq_of_zeros: NDArray[np.complex128] | Sequence[complex | np.complex128]) -> _Array1D[np.float64 | np.complex128]: ...
@overload  # 1d ~object_
def poly(seq_of_zeros: _Array1D[np.object_]) -> _Array1D[np.object_]: ...
@overload  # <=2d  (fallback)
def poly(seq_of_zeros: _ArrayLikeComplex_co) -> _Array1D[Any]: ...

# Returns either a float or complex array for real input depending on the input values.
@overload  # 1d Any  (workaround)
def roots(p: NDArray[np.inexact[Never]] | poly1d) -> _Array1D[Any]: ...
@overload  # 1d T
def roots[ScalarT: np.complexfloating](p: _Array1D[ScalarT] | Sequence[ScalarT]) -> _Array1D[ScalarT]: ...
@overload  # 1d ~f32
def roots(p: _Array1D[np.float32] | Sequence[np.float32]) -> _Array1D[np.float32 | np.complex64]: ...
@overload  # 1d +f64
def roots(
    p: _Array1D[np.float64 | _Int_co | np.object_] | Sequence[float | _Int_co | np.object_],
) -> _Array1D[np.float64 | np.complex128]: ...
@overload  # 1d ~complex
def roots(p: list[complex]) -> _Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def roots(p: _ArrayLikeComplex_co) -> _Array1D[Any]: ...

# keep in sync with `polyder`
@overload  # poly1d
def polyint(
    p: poly1d,
    m: SupportsIndex = 1,
    k: _ArrayLikeComplex_co | _ArrayLikeObject_co | None = None,
) -> poly1d: ...
@overload  # 1d T
def polyint[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble | np.object_](
    p: _Array1D[ScalarT] | Sequence[ScalarT],
    m: SupportsIndex = 1,
    k: _ArrayLikeFloat64_co | None = None,
) -> _Array1D[ScalarT]: ...
@overload  # 1d +f64
def polyint(
    p: _Array1D[np.float16 | np.float32 | _Int_co] | list[float],
    m: SupportsIndex = 1,
    k: _ArrayLikeFloat64_co | None = None,
) -> _Array1D[np.float64]: ...
@overload  # 1d +c128
def polyint(
    p: _Array1D[np.complex64] | list[complex],
    m: SupportsIndex = 1,
    k: _ArrayLikeComplex128_co | None = None,
) -> _Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def polyint(
    p: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    m: SupportsIndex = 1,
    k: _ArrayLikeComplex_co | _ArrayLikeObject_co | None = None,
) -> _Array1D[Any]: ...

# keep in sync with `polyint`
@overload  # poly1d
def polyder(p: poly1d, m: SupportsIndex = 1) -> poly1d: ...
@overload  # 1d T
def polyder[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble | np.object_](
    p: _Array1D[ScalarT] | Sequence[ScalarT],
    m: SupportsIndex = 1,
) -> _Array1D[ScalarT]: ...
@overload  # 1d +int
def polyder(
    p: _Array1D[np.bool | np.signedinteger | np.uint8 | np.uint16 | np.uint32] | list[int],
    m: SupportsIndex = 1,
) -> _Array1D[np.int_]: ...
@overload  # 1d +f64
def polyder(p: _Array1D[np.float16 | np.float32 | np.uint64] | list[float], m: SupportsIndex = 1) -> _Array1D[np.float64]: ...
@overload  # 1d +c128
def polyder(p: _Array1D[np.complex64] | list[complex], m: SupportsIndex = 1) -> _Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def polyder(p: _ArrayLikeComplex_co | _ArrayLikeObject_co, m: SupportsIndex = 1) -> _Array1D[Any]: ...

#
@overload  # ?d +f64, ?d +f64  (workaround)
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayJustND[_Float_co],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False,
) -> NDArray[float64]: ...
@overload  # ?d +f64, 1d +f64
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _Array1D[_Float_co] | Sequence[float],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False,
) -> _Array1D[float64]: ...
@overload  # ?d +f64, ?d +f64  (fallback)
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False,
) -> NDArray[float64]: ...
@overload  # ?d +f64, ?d +f64, cov=<given>  (workaround)
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayJustND[_Float_co],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> _2Tup[NDArray[float64]]: ...
@overload  # ?d +f64, 1d +f64, cov=<given>
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _Array1D[_Float_co] | Sequence[float],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> tuple[_Array1D[float64], _Array2D[float64]]: ...
@overload  # ?d +f64, ?d +f64, cov=<given>  (fallback)
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> _2Tup[NDArray[float64]]: ...
@overload  # ?d +f64, ?d +f64, full=True  (positional)
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None,
    full: L[True],
    w: _ArrayLikeFloat_co | None = None,
    cov: bool | L["unscaled"] = False,
) -> _5Tup[NDArray[float64]]: ...
@overload  # ?d +f64, ?d +f64, full=True  (keyword)
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ArrayLikeFloat_co | None = None,
    cov: bool | L["unscaled"] = False,
) -> _5Tup[NDArray[float64]]: ...
@overload  # ?d ~c128, ?d ~c128  (workaround)
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayJustND[_Number_co],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False,
) -> NDArray[complex128 | Any]: ...
@overload  # ?d ~c128, 1d ~c128
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _Array1D[_Number_co] | Sequence[complex],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False,
) -> _Array1D[complex128 | Any]: ...
@overload  # ?d ~c128, ?d ~c128  (fallback)
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False,
) -> NDArray[complex128 | Any]: ...
@overload  # ?d ~c128, ?d ~c128, cov=<given>  (workaround)
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayJustND[_Number_co],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> tuple[NDArray[complex128 | Any], NDArray[Any]]: ...
@overload  # ?d ~c128, 1d ~c128, cov=<given>
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _Array1D[_Number_co] | Sequence[complex],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> tuple[_Array1D[complex128 | Any], _Array2D[Any]]: ...
@overload  # ?d ~c128, ?d ~c128, cov=<given>  (fallback)
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> tuple[NDArray[complex128 | Any], NDArray[Any]]: ...
@overload  # ?d ~c128, ?d ~c128, full=True  (positional)
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None,
    full: L[True],
    w: _ArrayLikeFloat_co | None = None,
    cov: bool | L["unscaled"] = False,
) -> _5Tup[NDArray[complex128 | Any]]: ...
@overload  # ?d ~c128, ?d ~c128, full=True  (keyword)
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ArrayLikeFloat_co | None = None,
    cov: bool | L["unscaled"] = False,
) -> _5Tup[NDArray[complex128 | Any]]: ...

#
@overload  # 1d, poly1d
def polyval(p: _ArrayLikeComplex_co | _ArrayLikeObject_co, x: poly1d) -> poly1d: ...
@overload  # 1d T, 0d T
def polyval(p: _ArrayLike[_AnyNumberT], x: _AnyNumberT) -> _AnyNumberT: ...  # noqa: UP047
@overload  # 1d T, Nd T
def polyval(  # noqa: UP047
    p: _ArrayLike[_AnyNumberT],
    x: np.ndarray[_ShapeT, np.dtype[_AnyNumberT]],
) -> np.ndarray[_ShapeT, np.dtype[_AnyNumberT]]: ...
@overload  # 1d +int, 0d +int
def polyval(
    p: _ArrayLikeInt_co,
    x: _IntLike_co,
) -> np.int_ | Any: ...
@overload  # 1d +int, Nd +int
def polyval[ShapeT: _AnyShape](
    p: _ArrayLikeInt_co,
    x: np.ndarray[ShapeT, np.dtype[_Int_co]],
) -> np.ndarray[ShapeT, np.dtype[np.int_ | Any]]: ...
@overload  # 1d +int, ?d +int
def polyval(
    p: _ArrayLikeInt_co,
    x: _NestedSequence[NDArray[_Int_co] | int],
) -> NDArray[np.int_ | Any]: ...
@overload  # 1d +f64, 0d +f64
def polyval(
    p: _ArrayLikeFloat_co,
    x: _FloatLike_co,
) -> float64 | Any: ...
@overload  # 1d +f64, Nd +f64
def polyval[ShapeT: _AnyShape](
    p: _ArrayLikeFloat_co,
    x: np.ndarray[ShapeT, np.dtype[_Float_co]],
) -> np.ndarray[ShapeT, np.dtype[float64 | Any]]: ...
@overload  # 1d +f64, ?d +f64
def polyval(
    p: _ArrayLikeFloat_co,
    x: _NestedSequence[NDArray[_Float_co] | float],
) -> NDArray[float64 | Any]: ...
@overload  # 1d +c128, 0d +c128
def polyval(
    p: _ArrayLikeComplex_co,
    x: _ComplexLike_co,
) -> complex128 | Any: ...
@overload  # 1d +c128, Nd +c128
def polyval[ShapeT: _AnyShape](
    p: _ArrayLikeComplex_co,
    x: np.ndarray[ShapeT, np.dtype[_Number_co]],
) -> np.ndarray[ShapeT, np.dtype[complex128 | Any]]: ...
@overload  # 1d +c128, ?d +c128
def polyval(
    p: _ArrayLikeComplex_co,
    x: _NestedSequence[NDArray[_Number_co] | complex],
) -> NDArray[complex128 | Any]: ...
@overload  # fallback
def polyval(
    p: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    x: _ArrayLikeComplex_co | _ArrayLikeObject_co,
) -> NDArray[Any] | Any: ...

# keep in sync with `polysub` and `polymul`
@overload  # poly1d, <=1d
def polyadd(a1: poly1d, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co | poly1d) -> poly1d: ...
@overload  # <=1d, poly1d
def polyadd(a1: _ArrayLikeComplex_co | _ArrayLikeObject_co, a2: poly1d) -> poly1d: ...
@overload  # <=1d, <=1d T
def polyadd[ScalarT: np.number](a1: _ArrayLike[ScalarT], a2: _ArrayLike[ScalarT]) -> _Array1D[ScalarT]: ...
@overload  # <=1d, <=1d bool
def polyadd(a1: _ArrayLikeBool_co, a2: _ArrayLikeBool_co) -> _Array1D[np.bool]: ...
@overload  # <=1d, <=1d +int
def polyadd(a1: _ArrayLikeInt_co, a2: _ArrayLikeInt_co) -> _Array1D[np.int_ | Any]: ...
@overload  # <=1d, <=1d +f64
def polyadd(a1: _ArrayLikeFloat_co, a2: _ArrayLikeFloat_co) -> _Array1D[np.float64 | Any]: ...
@overload  # <=1d, <=1d +c128
def polyadd(a1: _ArrayLikeComplex_co, a2: _ArrayLikeComplex_co) -> _Array1D[np.complex128 | Any]: ...
@overload  # <=1d ~object_, <=1d
def polyadd(a1: _ArrayLikeObject_co, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co) -> _Array1D[np.object_]: ...
@overload  # <=1d, <=1d ~object_
def polyadd(a1: _ArrayLikeComplex_co, a2: _ArrayLikeObject_co) -> _Array1D[np.object_]: ...

# keep in sync with `polyadd` and `polymul`
@overload  # poly1d, <=1d
def polysub(a1: poly1d, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co | poly1d) -> poly1d: ...
@overload  # <=1d, poly1d
def polysub(a1: _ArrayLikeComplex_co | _ArrayLikeObject_co, a2: poly1d) -> poly1d: ...
@overload  # <=1d, <=1d T
def polysub[ScalarT: np.number](a1: _ArrayLike[ScalarT], a2: _ArrayLike[ScalarT]) -> _Array1D[ScalarT]: ...
@overload  # <=1d, <=1d bool
def polysub(a1: _ArrayLikeBool_co, a2: _ArrayLikeBool_co) -> NoReturn: ...
@overload  # <=1d, <=1d +int
def polysub(a1: _ArrayLikeInt_co, a2: _ArrayLikeInt_co) -> _Array1D[np.int_ | Any]: ...
@overload  # <=1d, <=1d +f64
def polysub(a1: _ArrayLikeFloat_co, a2: _ArrayLikeFloat_co) -> _Array1D[np.float64 | Any]: ...
@overload  # <=1d, <=1d +c128
def polysub(a1: _ArrayLikeComplex_co, a2: _ArrayLikeComplex_co) -> _Array1D[np.complex128 | Any]: ...
@overload  # <=1d ~object_, <=1d
def polysub(a1: _ArrayLikeObject_co, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co) -> _Array1D[np.object_]: ...
@overload  # <=1d, <=1d ~object_
def polysub(a1: _ArrayLikeComplex_co, a2: _ArrayLikeObject_co) -> _Array1D[np.object_]: ...

# keep in sync with `polyadd` and `polysub`
@overload  # poly1d, <=1d
def polymul(a1: poly1d, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co | poly1d) -> poly1d: ...
@overload  # <=1d, poly1d
def polymul(a1: _ArrayLikeComplex_co | _ArrayLikeObject_co, a2: poly1d) -> poly1d: ...
@overload  # <=1d, <=1d T
def polymul[ScalarT: np.number](a1: _ArrayLike[ScalarT], a2: _ArrayLike[ScalarT]) -> _Array1D[ScalarT]: ...
@overload  # <=1d, <=1d bool
def polymul(a1: _ArrayLikeBool_co, a2: _ArrayLikeBool_co) -> _Array1D[np.bool]: ...
@overload  # <=1d, <=1d +int
def polymul(a1: _ArrayLikeInt_co, a2: _ArrayLikeInt_co) -> _Array1D[np.int_ | Any]: ...
@overload  # <=1d, <=1d +f64
def polymul(a1: _ArrayLikeFloat_co, a2: _ArrayLikeFloat_co) -> _Array1D[np.float64 | Any]: ...
@overload  # <=1d, <=1d +c128
def polymul(a1: _ArrayLikeComplex_co, a2: _ArrayLikeComplex_co) -> _Array1D[np.complex128 | Any]: ...
@overload  # <=1d ~object_, <=1d
def polymul(a1: _ArrayLikeObject_co, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co) -> _Array1D[np.object_]: ...
@overload  # <=1d, <=1d ~object_
def polymul(a1: _ArrayLikeComplex_co, a2: _ArrayLikeObject_co) -> _Array1D[np.object_]: ...

#
@overload  # poly1d, 1d
def polydiv(u: poly1d, v: _ArrayLikeComplex_co | _ArrayLikeObject_co | poly1d) -> _2Tup[poly1d]: ...
@overload  # 1d, poly1d
def polydiv(u: _ArrayLikeComplex_co | _ArrayLikeObject_co, v: poly1d) -> _2Tup[poly1d]: ...
@overload  # 1d T, 1d T
def polydiv[ScalarT: np.inexact](u: _ArrayLike[ScalarT], v: _ArrayLike[ScalarT]) -> _2Tup[_Array1D[ScalarT]]: ...
@overload  # 1d +f64, 1d +f64
def polydiv(u: _ArrayLikeFloat_co, v: _ArrayLikeFloat_co) -> _2Tup[_Array1D[np.float64 | Any]]: ...
@overload  # 1d +c128, 1d +c128
def polydiv(u: _ArrayLikeComplex_co, v: _ArrayLikeComplex_co) -> _2Tup[_Array1D[np.complex128 | Any]]: ...
