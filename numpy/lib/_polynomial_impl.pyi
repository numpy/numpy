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
from numpy import (
    complex128,
    complexfloating,
    float64,
    floating,
    int32,
    object_,
    signedinteger,
    unsignedinteger,
)
from numpy._typing import (
    ArrayLike,
    NDArray,
    _AnyShape,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeUInt_co,
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
def poly(seq_of_zeros: ArrayLike) -> NDArray[floating]: ...

# Returns either a float or complex array depending on the input values.
# See `np.linalg.eigvals`.
def roots(p: ArrayLike) -> NDArray[complexfloating] | NDArray[floating]: ...

@overload
def polyint(
    p: poly1d,
    m: SupportsInt | SupportsIndex = 1,
    k: _ArrayLikeComplex_co | _ArrayLikeObject_co | None = None,
) -> poly1d: ...
@overload
def polyint(
    p: _ArrayLikeFloat_co,
    m: SupportsInt | SupportsIndex = 1,
    k: _ArrayLikeFloat_co | None = None,
) -> NDArray[floating]: ...
@overload
def polyint(
    p: _ArrayLikeComplex_co,
    m: SupportsInt | SupportsIndex = 1,
    k: _ArrayLikeComplex_co | None = None,
) -> NDArray[complexfloating]: ...
@overload
def polyint(
    p: _ArrayLikeObject_co,
    m: SupportsInt | SupportsIndex = 1,
    k: _ArrayLikeObject_co | None = None,
) -> NDArray[object_]: ...

@overload
def polyder(
    p: poly1d,
    m: SupportsInt | SupportsIndex = 1,
) -> poly1d: ...
@overload
def polyder(
    p: _ArrayLikeFloat_co,
    m: SupportsInt | SupportsIndex = 1,
) -> NDArray[floating]: ...
@overload
def polyder(
    p: _ArrayLikeComplex_co,
    m: SupportsInt | SupportsIndex = 1,
) -> NDArray[complexfloating]: ...
@overload
def polyder(
    p: _ArrayLikeObject_co,
    m: SupportsInt | SupportsIndex = 1,
) -> NDArray[object_]: ...

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

#
@overload
def polyadd(
    a1: poly1d,
    a2: _ArrayLikeComplex_co | _ArrayLikeObject_co,
) -> poly1d: ...
@overload
def polyadd(
    a1: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    a2: poly1d,
) -> poly1d: ...
@overload
def polyadd(
    a1: _ArrayLikeBool_co,
    a2: _ArrayLikeBool_co,
) -> NDArray[np.bool]: ...
@overload
def polyadd(
    a1: _ArrayLikeUInt_co,
    a2: _ArrayLikeUInt_co,
) -> NDArray[unsignedinteger]: ...
@overload
def polyadd(
    a1: _ArrayLikeInt_co,
    a2: _ArrayLikeInt_co,
) -> NDArray[signedinteger]: ...
@overload
def polyadd(
    a1: _ArrayLikeFloat_co,
    a2: _ArrayLikeFloat_co,
) -> NDArray[floating]: ...
@overload
def polyadd(
    a1: _ArrayLikeComplex_co,
    a2: _ArrayLikeComplex_co,
) -> NDArray[complexfloating]: ...
@overload
def polyadd(
    a1: _ArrayLikeObject_co,
    a2: _ArrayLikeObject_co,
) -> NDArray[object_]: ...

@overload
def polysub(
    a1: poly1d,
    a2: _ArrayLikeComplex_co | _ArrayLikeObject_co,
) -> poly1d: ...
@overload
def polysub(
    a1: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    a2: poly1d,
) -> poly1d: ...
@overload
def polysub(
    a1: _ArrayLikeBool_co,
    a2: _ArrayLikeBool_co,
) -> NoReturn: ...
@overload
def polysub(
    a1: _ArrayLikeUInt_co,
    a2: _ArrayLikeUInt_co,
) -> NDArray[unsignedinteger]: ...
@overload
def polysub(
    a1: _ArrayLikeInt_co,
    a2: _ArrayLikeInt_co,
) -> NDArray[signedinteger]: ...
@overload
def polysub(
    a1: _ArrayLikeFloat_co,
    a2: _ArrayLikeFloat_co,
) -> NDArray[floating]: ...
@overload
def polysub(
    a1: _ArrayLikeComplex_co,
    a2: _ArrayLikeComplex_co,
) -> NDArray[complexfloating]: ...
@overload
def polysub(
    a1: _ArrayLikeObject_co,
    a2: _ArrayLikeObject_co,
) -> NDArray[object_]: ...

# NOTE: Not an alias, but they do have the same signature (that we can reuse)
polymul = polyadd

@overload
def polydiv(
    u: poly1d,
    v: _ArrayLikeComplex_co | _ArrayLikeObject_co,
) -> _2Tup[poly1d]: ...
@overload
def polydiv(
    u: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    v: poly1d,
) -> _2Tup[poly1d]: ...
@overload
def polydiv(
    u: _ArrayLikeFloat_co,
    v: _ArrayLikeFloat_co,
) -> _2Tup[NDArray[floating]]: ...
@overload
def polydiv(
    u: _ArrayLikeComplex_co,
    v: _ArrayLikeComplex_co,
) -> _2Tup[NDArray[complexfloating]]: ...
@overload
def polydiv(
    u: _ArrayLikeObject_co,
    v: _ArrayLikeObject_co,
) -> _2Tup[NDArray[Any]]: ...
