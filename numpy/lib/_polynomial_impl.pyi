from _typeshed import Incomplete
from collections.abc import Iterator, Sequence
from typing import (
    Any,
    ClassVar,
    Generic,
    Literal as L,
    Never,
    NoReturn,
    Protocol,
    Self,
    SupportsIndex,
    SupportsInt,
    overload,
    override,
    type_check_only,
)
from typing_extensions import TypeVar

import numpy as np
from numpy import complex128, float64
from numpy._typing import (
    Array1D,
    Array2D,
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
    _Shape,
)

type _Int_co = np.integer | np.bool
type _Float_co = np.floating | np.integer | np.bool
type _Number_co = np.number | np.bool

# workaround for mypy and pyright not following the typing spec for overloads
type _ArrayJustND[ScalarT: np.generic] = np.ndarray[tuple[Never, Never, Never, Never], np.dtype[ScalarT]]

type _2Tup[T] = tuple[T, T]
type _5Tup[T] = tuple[
    T,
    Array1D[np.float64],
    np.int32,
    Array1D[np.float64],
    float | np.floating,
]

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
_ScalarT_co = TypeVar("_ScalarT_co", bound=_Number_co | np.object_, default=Any, covariant=True)

@type_check_only
class _CanNeg[T](Protocol):
    def __neg__(self, /) -> T: ...

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

class poly1d(Generic[_ScalarT_co]):
    __module__: L["numpy"] = "numpy"  # pyrefly: ignore[bad-override]

    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]

    @property
    def variable(self) -> str: ...
    @property
    def order(self) -> int: ...
    @property
    def o(self) -> int: ...
    @property
    def roots(self) -> Array1D[Incomplete]: ...
    @property
    def r(self) -> Array1D[Incomplete]: ...

    # NOTE: setting coefficients is type-unsafe, so we disallow this (using `Never`)

    #
    @property
    def coeffs(self) -> Array1D[_ScalarT_co]: ...
    @coeffs.setter
    def coeffs(self, value: Never, /) -> None: ...

    #
    @property
    def c(self) -> Array1D[_ScalarT_co]: ...
    @c.setter
    def c(self, value: Never, /) -> None: ...

    #
    @property
    def coef(self) -> Array1D[_ScalarT_co]: ...
    @coef.setter
    def coef(self, value: Never, /) -> None: ...

    #
    @property
    def coefficients(self) -> Array1D[_ScalarT_co]: ...
    @coefficients.setter
    def coefficients(self, value: Never, /) -> None: ...

    #
    @overload  # T
    def __init__[ScalarT: _Number_co | np.object_](
        self: poly1d[ScalarT],
        /,
        c_or_r: _ArrayLike[ScalarT],
        r: L[False] = False,
        variable: str | None = None,
    ) -> None: ...
    @overload  # ~bool
    def __init__(
        self: poly1d[np.bool],
        /,
        c_or_r: list[bool],
        r: L[False] = False,
        variable: str | None = None,
    ) -> None: ...
    @overload  # ~int
    def __init__(
        self: poly1d[np.int_],
        /,
        c_or_r: list[int],
        r: L[False] = False,
        variable: str | None = None,
    ) -> None: ...
    @overload  # ~float
    def __init__(
        self: poly1d[np.float64],
        /,
        c_or_r: list[float],
        r: L[False] = False,
        variable: str | None = None,
    ) -> None: ...
    @overload  # ~complex
    def __init__(
        self: poly1d[np.complex128],
        /,
        c_or_r: list[complex],
        r: L[False] = False,
        variable: str | None = None,
    ) -> None: ...
    @overload  # fallback
    def __init__(
        self: poly1d[Any],
        /,
        c_or_r: ArrayLike,
        r: bool = False,
        variable: str | None = None,
    ) -> None: ...

    #
    @overload
    def __array__(self, /, t: None = None, copy: bool | None = None) -> Array1D[_ScalarT_co]: ...
    @overload
    def __array__[DTypeT: np.dtype](self, /, t: DTypeT, copy: bool | None = None) -> np.ndarray[tuple[int], DTypeT]: ...

    #
    @overload  # T, poly1d T
    def __call__[ScalarT: np.number](self: poly1d[ScalarT], /, val: poly1d[ScalarT]) -> poly1d[ScalarT]: ...
    @overload  # T, Nd T
    def __call__[ScalarT: _Number_co, ShapeT: _Shape](
        self: poly1d[ScalarT],
        /,
        val: np.ndarray[ShapeT, np.dtype[ScalarT]],
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # T, 0d T
    def __call__[ScalarT: _Number_co](self: poly1d[ScalarT], /, val: ScalarT) -> ScalarT: ...
    @overload  # T, 1d T
    def __call__[ScalarT: _Number_co](self: poly1d[ScalarT], /, val: Sequence[ScalarT]) -> Array1D[ScalarT]: ...
    @overload  # T, 2d T
    def __call__[ScalarT: _Number_co](self: poly1d[ScalarT], /, val: Sequence[Sequence[ScalarT]]) -> Array2D[ScalarT]: ...
    @overload  # ~f64, 0d ~f64
    def __call__(self: poly1d[np.float64], /, val: float) -> np.float64: ...
    @overload  # ~f64, 1d ~f64
    def __call__(self: poly1d[np.float64], /, val: Sequence[float]) -> Array1D[np.float64]: ...
    @overload  # ~f64, 2d ~f64
    def __call__(self: poly1d[np.float64], /, val: Sequence[Sequence[float]]) -> Array2D[np.float64]: ...
    @overload  # ~c128, 0d ~c128
    def __call__(self: poly1d[np.complex128], /, val: complex) -> np.complex128: ...
    @overload  # ~c128, 1d ~c128
    def __call__(self: poly1d[np.complex128], /, val: Sequence[complex]) -> Array1D[np.complex128]: ...
    @overload  # ~c128, 2d ~c128
    def __call__(self: poly1d[np.complex128], /, val: Sequence[Sequence[complex]]) -> Array2D[np.complex128]: ...
    @overload  # poly1d
    def __call__(self, /, val: poly1d) -> poly1d: ...
    @overload  # Nd
    def __call__[ShapeT: _Shape](
        self,
        /,
        val: np.ndarray[ShapeT, np.dtype[_Number_co | np.object_]],
    ) -> np.ndarray[ShapeT, np.dtype[Any]]: ...
    @overload  # 0d
    def __call__(self, /, val: _ComplexLike_co) -> Any: ...
    @overload  # 1d
    def __call__(self, /, val: Sequence[_ComplexLike_co]) -> Array1D[Any]: ...
    @overload  # 2d
    def __call__(self, /, val: Sequence[Sequence[_ComplexLike_co]]) -> Array2D[Any]: ...
    @overload  # ?d  (fallback)
    def __call__(self, /, val: _ArrayLikeComplex_co | _ArrayLikeObject_co) -> NDArray[Any]: ...

    #
    def __len__(self) -> int: ...

    #
    @overload  # ~object_
    def __iter__[T](self: poly1d[np.object_[T]]) -> Iterator[T]: ...
    @overload  # T
    def __iter__[ScalarT: _Number_co](self: poly1d[ScalarT]) -> Iterator[ScalarT]: ...

    #
    @overload  # ~object_
    def __getitem__[T](self: poly1d[np.object_[T]], val: int, /) -> T: ...
    @overload  # T
    def __getitem__[ScalarT: _Number_co](self: poly1d[ScalarT], val: int, /) -> ScalarT: ...

    #
    def __setitem__(self, key: int, val: _ComplexLike_co, /) -> None: ...

    #
    @overload  # T
    def __neg__[ScalarT: np.number](self: poly1d[ScalarT], /) -> poly1d[ScalarT]: ...
    @overload  # ~object_
    def __neg__[T](self: poly1d[np.object_[_CanNeg[T]]], /) -> poly1d[np.object_[T]]: ...

    #
    def __pos__(self) -> Self: ...

    #
    @overload  # T, <=1d T
    def __add__[ScalarT: _Number_co](self: poly1d[ScalarT], other: _ArrayLike[ScalarT], /) -> poly1d[ScalarT]: ...
    @overload  # ~f64, <=1d ~f64
    def __add__(self: poly1d[np.float64], other: float | Sequence[float], /) -> poly1d[np.float64]: ...
    @overload  # ~c128, <=1d ~c128
    def __add__(self: poly1d[np.complex128], other: complex | Sequence[complex], /) -> poly1d[np.complex128]: ...
    @overload  # <=1d  (fallback)
    def __add__(self, other: ArrayLike, /) -> poly1d: ...

    #
    @overload  # T, <=1d T
    def __radd__[ScalarT: _Number_co](self: poly1d[ScalarT], other: _ArrayLike[ScalarT], /) -> poly1d[ScalarT]: ...
    @overload  # ~f64, <=1d ~f64
    def __radd__(self: poly1d[np.float64], other: float | Sequence[float], /) -> poly1d[np.float64]: ...
    @overload  # ~c128, <=1d ~c128
    def __radd__(self: poly1d[np.complex128], other: complex | Sequence[complex], /) -> poly1d[np.complex128]: ...
    @overload  # <=1d  (fallback)
    def __radd__(self, other: ArrayLike, /) -> poly1d: ...

    #
    @overload  # T, <=1d T
    def __sub__[ScalarT: np.number](self: poly1d[ScalarT], other: _ArrayLike[ScalarT], /) -> poly1d[ScalarT]: ...
    @overload  # ~f64, <=1d ~f64
    def __sub__(self: poly1d[np.float64], other: float | Sequence[float], /) -> poly1d[np.float64]: ...
    @overload  # ~c128, <=1d ~c128
    def __sub__(self: poly1d[np.complex128], other: complex | Sequence[complex], /) -> poly1d[np.complex128]: ...
    @overload  # <=1d  (fallback)
    def __sub__(self, other: ArrayLike, /) -> poly1d: ...

    #
    @overload  # T, <=1d T
    def __rsub__[ScalarT: np.number](self: poly1d[ScalarT], other: _ArrayLike[ScalarT], /) -> poly1d[ScalarT]: ...
    @overload  # ~f64, <=1d ~f64
    def __rsub__(self: poly1d[np.float64], other: float | Sequence[float], /) -> poly1d[np.float64]: ...
    @overload  # ~c128, <=1d ~c128
    def __rsub__(self: poly1d[np.complex128], other: complex | Sequence[complex], /) -> poly1d[np.complex128]: ...
    @overload  # <=1d  (fallback)
    def __rsub__(self, other: ArrayLike, /) -> poly1d: ...

    #
    @overload  # T, <=1d T
    def __mul__[ScalarT: _Number_co](self: poly1d[ScalarT], other: _ArrayLike[ScalarT], /) -> poly1d[ScalarT]: ...
    @overload  # 0d
    def __mul__(self, other: np.number | np.bool, /) -> poly1d: ...
    @overload  # T, 0d ~i8
    def __mul__[ScalarT: np.number](self: poly1d[ScalarT], other: int, /) -> poly1d[ScalarT]: ...
    @overload  # T, 0d ~f64
    def __mul__[ScalarT: np.inexact](self: poly1d[ScalarT], other: float, /) -> poly1d[ScalarT]: ...
    @overload  # T, 0d ~c128
    def __mul__[ScalarT: np.complexfloating](self: poly1d[ScalarT], other: complex, /) -> poly1d[ScalarT]: ...
    @overload  # ~f64, 1d ~f64
    def __mul__(self: poly1d[np.float64], other: Sequence[float], /) -> poly1d[np.float64]: ...
    @overload  # ~c128, 1d ~c128
    def __mul__(self: poly1d[np.complex128], other: Sequence[complex], /) -> poly1d[np.complex128]: ...
    @overload  # <=1d  (fallback)
    def __mul__(self, other: ArrayLike, /) -> poly1d: ...

    #
    @overload  # T, <=1d T
    def __rmul__[ScalarT: _Number_co](self: poly1d[ScalarT], other: _ArrayLike[ScalarT], /) -> poly1d[ScalarT]: ...
    @overload  # 0d
    def __rmul__(self, other: np.number | np.bool, /) -> poly1d: ...
    @overload  # T, 0d ~i8
    def __rmul__[ScalarT: np.number](self: poly1d[ScalarT], other: int, /) -> poly1d[ScalarT]: ...
    @overload  # T, 0d ~f64
    def __rmul__[ScalarT: np.inexact](self: poly1d[ScalarT], other: float, /) -> poly1d[ScalarT]: ...
    @overload  # T, 0d ~c128
    def __rmul__[ScalarT: np.complexfloating](self: poly1d[ScalarT], other: complex, /) -> poly1d[ScalarT]: ...
    @overload  # ~f64, 1d ~f64
    def __rmul__(self: poly1d[np.float64], other: Sequence[float], /) -> poly1d[np.float64]: ...
    @overload  # ~c128, 1d ~c128
    def __rmul__(self: poly1d[np.complex128], other: Sequence[complex], /) -> poly1d[np.complex128]: ...
    @overload  # <=1d  (fallback)
    def __rmul__(self, other: ArrayLike, /) -> poly1d: ...

    #
    @overload  # T
    def __pow__[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self: poly1d[ScalarT],
        val: _IntLike_co,
        /,
    ) -> poly1d[ScalarT]: ...
    @overload  # +int
    def __pow__(
        self: poly1d[np.signedinteger | np.uint32 | np.uint16 | np.uint8 | np.bool],
        val: _IntLike_co,
        /,
    ) -> poly1d[np.int_]: ...
    @overload  # +f64
    def __pow__(self: poly1d[np.float32 | np.float16 | np.uint64], val: _IntLike_co, /) -> poly1d[np.float64]: ...
    @overload  # +c128
    def __pow__(self: poly1d[np.complex64], val: _IntLike_co, /) -> poly1d[np.complex128]: ...
    @overload  # fallback
    def __pow__(self, val: _IntLike_co, /) -> poly1d: ...

    #
    @overload  # T, 0d T
    def __truediv__[ScalarT: np.inexact](
        self: poly1d[ScalarT],
        other: ScalarT,
        /,
    ) -> poly1d[ScalarT]: ...
    @overload  # 0d
    def __truediv__(
        self,
        other: np.number | np.bool,
        /,
    ) -> poly1d: ...
    @overload  # T, 0d ~f64
    def __truediv__[ScalarT: np.inexact](
        self: poly1d[ScalarT],
        other: float,
        /,
    ) -> poly1d[ScalarT]: ...
    @overload  # +int, 0d ~f64
    def __truediv__(
        self: poly1d[_Int_co],
        other: float,
        /,
    ) -> poly1d[np.float64]: ...
    @overload  # T, 0d ~c128
    def __truediv__[ScalarT: np.complexfloating](
        self: poly1d[ScalarT],
        other: complex,
        /,
    ) -> poly1d[ScalarT]: ...
    @overload  # 0d  (fallback)
    def __truediv__(
        self,
        other: complex,
        /,
    ) -> poly1d: ...
    @overload  # T, <=1d T
    def __truediv__[ScalarT: np.inexact](
        self: poly1d[ScalarT],
        other: NDArray[ScalarT] | poly1d[ScalarT],
        /,
    ) -> _2Tup[poly1d[ScalarT]]: ...
    @overload  # +f64, 1d ~f64
    def __truediv__(
        self: poly1d[np.float64 | np.float32 | np.float16 | _Int_co],
        other: Sequence[float],
        /,
    ) -> _2Tup[poly1d[np.float64]]: ...
    @overload  # +c128, 1d ~c128
    def __truediv__(
        self: poly1d[np.complex128 | np.complex64],
        other: Sequence[complex],
        /,
    ) -> _2Tup[poly1d[np.complex128]]: ...
    @overload  # <=1d  (fallback)
    def __truediv__(
        self,
        other: NDArray[_Number_co | np.object_] | poly1d | Sequence[_ComplexLike_co],
        /,
    ) -> _2Tup[poly1d]: ...

    #
    @overload  # T, 0d T
    def __rtruediv__[ScalarT: np.inexact](
        self: poly1d[ScalarT],
        other: ScalarT,
        /,
    ) -> poly1d[ScalarT]: ...
    @overload  # 0d
    def __rtruediv__(
        self,
        other: np.number | np.bool,
        /,
    ) -> poly1d: ...
    @overload  # T, 0d ~f64
    def __rtruediv__[ScalarT: np.inexact](
        self: poly1d[ScalarT],
        other: float,
        /,
    ) -> poly1d[ScalarT]: ...
    @overload  # +int, 0d ~f64
    def __rtruediv__(
        self: poly1d[_Int_co],
        other: float,
        /,
    ) -> poly1d[np.float64]: ...
    @overload  # T, 0d ~c128
    def __rtruediv__[ScalarT: np.complexfloating](
        self: poly1d[ScalarT],
        other: complex,
        /,
    ) -> poly1d[ScalarT]: ...
    @overload  # 0d  (fallback)
    def __rtruediv__(
        self,
        other: complex,
        /,
    ) -> poly1d: ...
    @overload  # T, <=1d T
    def __rtruediv__[ScalarT: np.inexact](
        self: poly1d[ScalarT],
        other: poly1d[ScalarT],
        /,
    ) -> _2Tup[poly1d[ScalarT]]: ...
    @overload  # +f64, 1d ~f64
    def __rtruediv__(
        self: poly1d[np.float64 | np.float32 | np.float16 | _Int_co],
        other: Sequence[float],
        /,
    ) -> _2Tup[poly1d[np.float64]]: ...
    @overload  # +c128, 1d ~c128
    def __rtruediv__(
        self: poly1d[np.complex128 | np.complex64],
        other: Sequence[complex],
        /,
    ) -> _2Tup[poly1d[np.complex128]]: ...
    @overload  # <=1d  (fallback)
    def __rtruediv__(
        self,
        other: NDArray[_Number_co | np.object_] | poly1d | Sequence[_ComplexLike_co],
        /,
    ) -> _2Tup[poly1d]: ...

    #
    @override
    def __eq__(self, other: poly1d, /) -> bool: ...  # type:ignore[override]
    @override
    def __ne__(self, other: poly1d, /) -> bool: ...  # type:ignore[override]

    #
    @overload  # T
    def deriv[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self: poly1d[ScalarT],
        /,
        m: SupportsIndex = 1,
    ) -> poly1d[ScalarT]: ...
    @overload  # +f64
    def deriv(
        self: poly1d[np.float32 | np.float16 | np.uint64],
        /,
        m: SupportsIndex = 1,
    ) -> poly1d[np.float64]: ...
    @overload  # +int
    def deriv(
        self: poly1d[np.signedinteger | np.uint32 | np.uint16 | np.uint8 | np.bool],
        /,
        m: SupportsIndex = 1,
    ) -> poly1d[np.int_]: ...
    @overload  # +c128
    def deriv(
        self: poly1d[np.complex64],
        /,
        m: SupportsIndex = 1,
    ) -> poly1d[np.complex128]: ...
    @overload  # fallback
    def deriv(
        self,
        /,
        m: SupportsIndex = 1,
    ) -> poly1d: ...

    #
    @overload  # T
    def integ[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self: poly1d[ScalarT],
        /,
        m: SupportsIndex = 1,
        k: _ArrayLikeFloat64_co | None = 0,
    ) -> poly1d[ScalarT]: ...
    @overload  # +f64
    def integ(
        self: poly1d[np.float32 | np.float16 | _Int_co],
        /,
        m: SupportsIndex = 1,
        k: _ArrayLikeFloat64_co | None = 0,
    ) -> poly1d[np.float64]: ...
    @overload  # +c128
    def integ(
        self: poly1d[np.complex64],
        /,
        m: SupportsIndex = 1,
        k: _ArrayLikeComplex128_co | None = 0,
    ) -> poly1d[np.complex128]: ...
    @overload  # fallback
    def integ(
        self,
        /,
        m: SupportsIndex = 1,
        k: _ArrayLikeComplex_co | _ArrayLikeObject_co | None = 0,
    ) -> poly1d: ...

#
@overload  # <=2d Any  (workaround)
def poly(seq_of_zeros: NDArray[np.inexact[Never]] | poly1d) -> Array1D[Any]: ...
@overload  # <=2d ~f32
def poly(seq_of_zeros: NDArray[np.float32] | Sequence[np.float32]) -> Array1D[np.float32]: ...
@overload  # <=2d ~c64
def poly(seq_of_zeros: NDArray[np.complex64] | Sequence[np.complex64]) -> Array1D[np.float32 | np.complex64]: ...
@overload  # <=2d +f64
def poly(seq_of_zeros: NDArray[np.float64 | _Int_co] | Sequence[float | _Int_co]) -> Array1D[np.float64]: ...
@overload  # <=2d +c128
def poly(seq_of_zeros: NDArray[np.complex128] | Sequence[complex | np.complex128]) -> Array1D[np.float64 | np.complex128]: ...
@overload  # 1d ~object_
def poly(seq_of_zeros: Array1D[np.object_]) -> Array1D[np.object_]: ...
@overload  # <=2d  (fallback)
def poly(seq_of_zeros: _ArrayLikeComplex_co) -> Array1D[Any]: ...

# Returns either a float or complex array for real input depending on the input values.
@overload  # 1d Any  (workaround)
def roots(p: NDArray[np.inexact[Never]] | poly1d) -> Array1D[Any]: ...
@overload  # 1d T
def roots[ScalarT: np.complexfloating](p: Array1D[ScalarT] | Sequence[ScalarT]) -> Array1D[ScalarT]: ...
@overload  # 1d ~f32
def roots(p: Array1D[np.float32] | Sequence[np.float32]) -> Array1D[np.float32 | np.complex64]: ...
@overload  # 1d +f64
def roots(
    p: Array1D[np.float64 | _Int_co | np.object_] | Sequence[float | _Int_co | np.object_],
) -> Array1D[np.float64 | np.complex128]: ...
@overload  # 1d ~complex
def roots(p: list[complex]) -> Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def roots(p: _ArrayLikeComplex_co) -> Array1D[Any]: ...

# keep in sync with `polyder`
@overload  # poly1d
def polyint(
    p: poly1d,
    m: SupportsIndex = 1,
    k: _ArrayLikeComplex_co | _ArrayLikeObject_co | None = None,
) -> poly1d: ...
@overload  # 1d T
def polyint[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble | np.object_](
    p: Array1D[ScalarT] | Sequence[ScalarT],
    m: SupportsIndex = 1,
    k: _ArrayLikeFloat64_co | None = None,
) -> Array1D[ScalarT]: ...
@overload  # 1d +f64
def polyint(
    p: Array1D[np.float16 | np.float32 | _Int_co] | list[float],
    m: SupportsIndex = 1,
    k: _ArrayLikeFloat64_co | None = None,
) -> Array1D[np.float64]: ...
@overload  # 1d +c128
def polyint(
    p: Array1D[np.complex64] | list[complex],
    m: SupportsIndex = 1,
    k: _ArrayLikeComplex128_co | None = None,
) -> Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def polyint(
    p: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    m: SupportsIndex = 1,
    k: _ArrayLikeComplex_co | _ArrayLikeObject_co | None = None,
) -> Array1D[Any]: ...

# keep in sync with `polyint`
@overload  # poly1d
def polyder(p: poly1d, m: SupportsIndex = 1) -> poly1d: ...
@overload  # 1d T
def polyder[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble | np.object_](
    p: Array1D[ScalarT] | Sequence[ScalarT],
    m: SupportsIndex = 1,
) -> Array1D[ScalarT]: ...
@overload  # 1d +int
def polyder(
    p: Array1D[np.bool | np.signedinteger | np.uint8 | np.uint16 | np.uint32] | list[int],
    m: SupportsIndex = 1,
) -> Array1D[np.int_]: ...
@overload  # 1d +f64
def polyder(p: Array1D[np.float16 | np.float32 | np.uint64] | list[float], m: SupportsIndex = 1) -> Array1D[np.float64]: ...
@overload  # 1d +c128
def polyder(p: Array1D[np.complex64] | list[complex], m: SupportsIndex = 1) -> Array1D[np.complex128]: ...
@overload  # 1d  (fallback)
def polyder(p: _ArrayLikeComplex_co | _ArrayLikeObject_co, m: SupportsIndex = 1) -> Array1D[Any]: ...

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
    y: Array1D[_Float_co] | Sequence[float],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False,
) -> Array1D[float64]: ...
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
    y: Array1D[_Float_co] | Sequence[float],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> tuple[Array1D[float64], Array2D[float64]]: ...
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
    y: Array1D[_Number_co] | Sequence[complex],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    cov: L[False] = False,
) -> Array1D[complex128 | Any]: ...
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
    y: Array1D[_Number_co] | Sequence[complex],
    deg: SupportsIndex | SupportsInt,
    rcond: float | None = None,
    full: L[False] = False,
    w: _ArrayLikeFloat_co | None = None,
    *,
    cov: L[True, "unscaled"],
) -> tuple[Array1D[complex128 | Any], Array2D[Any]]: ...
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
def polyadd[ScalarT: np.number](a1: _ArrayLike[ScalarT], a2: _ArrayLike[ScalarT]) -> Array1D[ScalarT]: ...
@overload  # <=1d, <=1d bool
def polyadd(a1: _ArrayLikeBool_co, a2: _ArrayLikeBool_co) -> Array1D[np.bool]: ...
@overload  # <=1d, <=1d +int
def polyadd(a1: _ArrayLikeInt_co, a2: _ArrayLikeInt_co) -> Array1D[np.int_ | Any]: ...
@overload  # <=1d, <=1d +f64
def polyadd(a1: _ArrayLikeFloat_co, a2: _ArrayLikeFloat_co) -> Array1D[np.float64 | Any]: ...
@overload  # <=1d, <=1d +c128
def polyadd(a1: _ArrayLikeComplex_co, a2: _ArrayLikeComplex_co) -> Array1D[np.complex128 | Any]: ...
@overload  # <=1d ~object_, <=1d
def polyadd(a1: _ArrayLikeObject_co, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d ~object_
def polyadd(a1: _ArrayLikeComplex_co, a2: _ArrayLikeObject_co) -> Array1D[np.object_]: ...

# keep in sync with `polyadd` and `polymul`
@overload  # poly1d, <=1d
def polysub(a1: poly1d, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co | poly1d) -> poly1d: ...
@overload  # <=1d, poly1d
def polysub(a1: _ArrayLikeComplex_co | _ArrayLikeObject_co, a2: poly1d) -> poly1d: ...
@overload  # <=1d, <=1d T
def polysub[ScalarT: np.number](a1: _ArrayLike[ScalarT], a2: _ArrayLike[ScalarT]) -> Array1D[ScalarT]: ...
@overload  # <=1d, <=1d bool
def polysub(a1: _ArrayLikeBool_co, a2: _ArrayLikeBool_co) -> NoReturn: ...
@overload  # <=1d, <=1d +int
def polysub(a1: _ArrayLikeInt_co, a2: _ArrayLikeInt_co) -> Array1D[np.int_ | Any]: ...
@overload  # <=1d, <=1d +f64
def polysub(a1: _ArrayLikeFloat_co, a2: _ArrayLikeFloat_co) -> Array1D[np.float64 | Any]: ...
@overload  # <=1d, <=1d +c128
def polysub(a1: _ArrayLikeComplex_co, a2: _ArrayLikeComplex_co) -> Array1D[np.complex128 | Any]: ...
@overload  # <=1d ~object_, <=1d
def polysub(a1: _ArrayLikeObject_co, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d ~object_
def polysub(a1: _ArrayLikeComplex_co, a2: _ArrayLikeObject_co) -> Array1D[np.object_]: ...

# keep in sync with `polyadd` and `polysub`
@overload  # poly1d, <=1d
def polymul(a1: poly1d, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co | poly1d) -> poly1d: ...
@overload  # <=1d, poly1d
def polymul(a1: _ArrayLikeComplex_co | _ArrayLikeObject_co, a2: poly1d) -> poly1d: ...
@overload  # <=1d, <=1d T
def polymul[ScalarT: np.number](a1: _ArrayLike[ScalarT], a2: _ArrayLike[ScalarT]) -> Array1D[ScalarT]: ...
@overload  # <=1d, <=1d bool
def polymul(a1: _ArrayLikeBool_co, a2: _ArrayLikeBool_co) -> Array1D[np.bool]: ...
@overload  # <=1d, <=1d +int
def polymul(a1: _ArrayLikeInt_co, a2: _ArrayLikeInt_co) -> Array1D[np.int_ | Any]: ...
@overload  # <=1d, <=1d +f64
def polymul(a1: _ArrayLikeFloat_co, a2: _ArrayLikeFloat_co) -> Array1D[np.float64 | Any]: ...
@overload  # <=1d, <=1d +c128
def polymul(a1: _ArrayLikeComplex_co, a2: _ArrayLikeComplex_co) -> Array1D[np.complex128 | Any]: ...
@overload  # <=1d ~object_, <=1d
def polymul(a1: _ArrayLikeObject_co, a2: _ArrayLikeComplex_co | _ArrayLikeObject_co) -> Array1D[np.object_]: ...
@overload  # <=1d, <=1d ~object_
def polymul(a1: _ArrayLikeComplex_co, a2: _ArrayLikeObject_co) -> Array1D[np.object_]: ...

#
@overload  # poly1d, 1d
def polydiv(u: poly1d, v: _ArrayLikeComplex_co | _ArrayLikeObject_co | poly1d) -> _2Tup[poly1d]: ...
@overload  # 1d, poly1d
def polydiv(u: _ArrayLikeComplex_co | _ArrayLikeObject_co, v: poly1d) -> _2Tup[poly1d]: ...
@overload  # 1d T, 1d T
def polydiv[ScalarT: np.inexact](u: _ArrayLike[ScalarT], v: _ArrayLike[ScalarT]) -> _2Tup[Array1D[ScalarT]]: ...
@overload  # 1d +f64, 1d +f64
def polydiv(u: _ArrayLikeFloat_co, v: _ArrayLikeFloat_co) -> _2Tup[Array1D[np.float64 | Any]]: ...
@overload  # 1d +c128, 1d +c128
def polydiv(u: _ArrayLikeComplex_co, v: _ArrayLikeComplex_co) -> _2Tup[Array1D[np.complex128 | Any]]: ...
