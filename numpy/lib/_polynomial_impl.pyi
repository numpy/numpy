from _typeshed import ConvertibleToInt, Incomplete
from collections.abc import Iterator
from typing import (
    Any,
    ClassVar,
    Literal as L,
    NoReturn,
    Self,
    SupportsIndex,
    SupportsInt,
    overload,
)

import numpy as np
from numpy import (
    complex128,
    complexfloating,
    float64,
    floating,
    int32,
    int64,
    object_,
    signedinteger,
    unsignedinteger,
)
from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeUInt_co,
    _FloatLike_co,
    _NestedSequence,
    _ScalarLike_co,
)

type _2Tup[T] = tuple[T, T]
type _5Tup[T] = tuple[T, NDArray[float64], NDArray[int32], NDArray[float64], NDArray[float64]]

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
    __module__: L["numpy"] = "numpy"

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

@overload
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False,
) -> NDArray[float64]: ...
@overload
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False,
) -> NDArray[complex128]: ...
@overload
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
@overload
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> _2Tup[NDArray[complex128]]: ...
@overload
def polyfit(
    x: _ArrayLikeFloat_co,
    y: _ArrayLikeFloat_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None,
    full: L[True],
    w: _ArrayLikeFloat_co | None = None,
    cov: bool | L["unscaled"] = False,
) -> _5Tup[NDArray[float64]]: ...
@overload
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
@overload
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None,
    full: L[True],
    w: _ArrayLikeFloat_co | None = None,
    cov: bool | L["unscaled"] = False,
) -> _5Tup[NDArray[complex128]]: ...
@overload
def polyfit(
    x: _ArrayLikeComplex_co,
    y: _ArrayLikeComplex_co,
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    *,
    full: L[True],
    w: _ArrayLikeFloat_co | None = None,
    cov: bool | L["unscaled"] = False,
) -> _5Tup[NDArray[complex128]]: ...

@overload
def polyval(
    p: _ArrayLikeBool_co,
    x: _ArrayLikeBool_co,
) -> NDArray[int64]: ...
@overload
def polyval(
    p: _ArrayLikeUInt_co,
    x: _ArrayLikeUInt_co,
) -> NDArray[unsignedinteger]: ...
@overload
def polyval(
    p: _ArrayLikeInt_co,
    x: _ArrayLikeInt_co,
) -> NDArray[signedinteger]: ...
@overload
def polyval(
    p: _ArrayLikeFloat_co,
    x: _ArrayLikeFloat_co,
) -> NDArray[floating]: ...
@overload
def polyval(
    p: _ArrayLikeComplex_co,
    x: _ArrayLikeComplex_co,
) -> NDArray[complexfloating]: ...
@overload
def polyval(
    p: _ArrayLikeObject_co,
    x: _ArrayLikeObject_co,
) -> NDArray[object_]: ...

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
