import contextvars
from _typeshed import SupportsWrite
from collections.abc import Callable, Sequence
from types import EllipsisType
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    Never,
    Protocol,
    Self,
    TypedDict,
    Unpack,
    overload,
    override,
    type_check_only,
)
from typing_extensions import CapsuleType, TypeVar

import numpy as np
import numpy.typing as npt
from numpy import (
    _CastingKind,
    _OrderKACF,
    add,
    arctan2,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    conj,
    copysign,
    divide,
    divmod,
    e,
    equal,
    euler_gamma,
    float_power,
    floor_divide,
    fmax,
    fmin,
    fmod,
    frexp,
    frompyfunc,
    gcd,
    greater,
    greater_equal,
    heaviside,
    hypot,
    lcm,
    ldexp,
    left_shift,
    less,
    less_equal,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_or,
    logical_xor,
    matmul,
    matvec,
    maximum,
    minimum,
    mod,
    modf,
    multiply,
    nextafter,
    not_equal,
    pi,
    power,
    remainder,
    right_shift,
    subtract,
    true_divide,
    vecdot,
    vecmat,
)
from numpy._typing import (
    _ArrayLike,
    _ArrayLikeAnyString_co,
    _ArrayLikeBool_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt,
    _ArrayLikeInt_co,
    _ArrayLikeNumber_co,
    _ArrayLikeStr_co,
    _ArrayLikeString_co,
    _CharLike_co,
    _DTypeLike,
    _DTypeLikeBool,
    _FloatLike_co,
    _IntLike_co,
    _NestedSequence,
    _NumberLike_co,
    _Shape,
)
from numpy._typing._array_like import _DualArrayLike

__all__ = [
    "absolute",
    "add",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "bitwise_and",
    "bitwise_count",
    "bitwise_or",
    "bitwise_xor",
    "cbrt",
    "ceil",
    "conj",
    "conjugate",
    "copysign",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "divide",
    "divmod",
    "e",
    "equal",
    "euler_gamma",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "float_power",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "frompyfunc",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "invert",
    "isfinite",
    "isinf",
    "isnan",
    "isnat",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "matmul",
    "matvec",
    "maximum",
    "minimum",
    "mod",
    "modf",
    "multiply",
    "negative",
    "nextafter",
    "not_equal",
    "pi",
    "positive",
    "power",
    "rad2deg",
    "radians",
    "reciprocal",
    "remainder",
    "right_shift",
    "rint",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spacing",
    "sqrt",
    "square",
    "subtract",
    "tan",
    "tanh",
    "true_divide",
    "trunc",
    "vecdot",
    "vecmat",
]

###

type _Array0D[ScalarT: np.generic] = np.ndarray[tuple[()], np.dtype[ScalarT]]
type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]

type _ErrKind = Literal["ignore", "warn", "raise", "call", "print", "log"]
type _ErrCall = Callable[[str, int], Any] | SupportsWrite[str]

type _to_integer = np.integer | np.bool
type _to_floating = np.floating | _to_integer
type _to_number = np.number | np.bool
type _to_numeric = _to_number | np.timedelta64
type _numeric = np.number | np.timedelta64
type _time = np.datetime64 | np.timedelta64

type _ArrayLikeNumericObj = _DualArrayLike[np.dtype[_numeric | np.object_], complex]
type _ArrayLikeNumericObj_co = _DualArrayLike[np.dtype[_to_numeric | np.object_], complex]

@type_check_only
class _ExtOjbDict(TypedDict, total=False):
    divide: _ErrKind
    over: _ErrKind
    under: _ErrKind
    invalid: _ErrKind
    call: _ErrCall | None
    bufsize: int

type _Signature1 = tuple[str | None, str | None] | str

@type_check_only
class _CanUfuncCall1[OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["__call__"], /, *args: Any, **kwargs: Any) -> OutT: ...

@type_check_only
class _CanUfuncAt1[IxT, OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["at"], a: Self, indices: IxT, /) -> OutT: ...

###
# ufunc specializations

# NOTE: ignoring LSP errors here is harmless, because ufuncs are final at runtime
# mypy: disable-error-code=override
# pyright: reportIncompatibleMethodOverride=false

@type_check_only
class _Kwargs11(TypedDict, total=False):
    where: _ArrayLikeBool_co  # = True
    casting: _CastingKind  # = "same_kind"
    order: _OrderKACF  # = "K",
    subok: bool  # = True,
    signature: _Signature1

_IdT_co = TypeVar("_IdT_co", covariant=True, default=None)

@type_check_only
class _ufunc_11(np.ufunc, Generic[_IdT_co]):  # type: ignore[misc]
    @property
    @override
    def identity(self) -> _IdT_co: ...
    @property
    @override
    def nin(self) -> Literal[1]: ...
    @property
    @override
    def nout(self) -> Literal[1]: ...
    @property
    @override
    def nargs(self) -> Literal[2]: ...
    @property
    @override
    def signature(self) -> None: ...

    #
    @override
    def accumulate(self, array: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]
    @override
    def reduce(self, array: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]
    @override
    def reduceat(self, array: Never, /, indices: Never) -> Never: ...  # pyrefly:ignore[bad-override]
    @override
    def outer(self, A: Never, B: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]

# Mm => ?
@type_check_only
class _ufunc_11_m_b(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_time]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
    @overload  # scalar
    def __call__(
        self,
        x: _time,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.bool: ...
    @overload  # scalar, out=...
    def __call__(
        self,
        x: _time,
        /,
        *,
        out: EllipsisType,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array0D[np.bool]: ...
    @overload  # 1d
    def __call__(
        self,
        x: Sequence[_time],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.bool]: ...
    @overload  # 2d
    def __call__(
        self,
        x: Sequence[Sequence[_time]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.bool]: ...
    @overload  # 3d
    def __call__(
        self,
        x: Sequence[Sequence[Sequence[_time]]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array3D[np.bool]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLike[_time],
        /,
        out: OutT,
        *,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: np.ndarray[_Shape, np.dtype[_time]], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# efdg => ?
@type_check_only
class _ufunc_11_f_b(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_floating]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
    @overload  # scalar
    def __call__(
        self,
        x: float | _to_floating,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.bool: ...
    @overload  # scalar, out=...
    def __call__(
        self,
        x: float | _to_floating,
        /,
        *,
        out: EllipsisType,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array0D[np.bool]: ...
    @overload  # 1d
    def __call__(
        self,
        x: Sequence[float | _to_floating],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.bool]: ...
    @overload  # 2d
    def __call__(
        self,
        x: Sequence[Sequence[float | _to_floating]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.bool]: ...
    @overload  # 3d
    def __call__(
        self,
        x: Sequence[Sequence[Sequence[float | _to_floating]]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array3D[np.bool]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeFloat_co,
        /,
        out: OutT,
        *,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: np.ndarray[_Shape, np.dtype[np.floating]], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# ?bBhHiIlLqQefdgFDGmM => ?
@type_check_only
class _ufunc_11_bifgcm_b(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_number | _time]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
    @overload  # scalar
    def __call__(
        self,
        x: complex | _to_number | _time,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.bool: ...
    @overload  # scalar, out=...
    def __call__(
        self,
        x: complex | _to_number | _time,
        /,
        *,
        out: EllipsisType,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array0D[np.bool]: ...
    @overload  # 1d
    def __call__(
        self,
        x: Sequence[complex | _to_number | _time],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.bool]: ...
    @overload  # 2d
    def __call__(
        self,
        x: Sequence[Sequence[complex | _to_number | _time]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.bool]: ...
    @overload  # 3d
    def __call__(
        self,
        x: Sequence[Sequence[Sequence[complex | _to_number | _time]]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array3D[np.bool]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeNumber_co | _ArrayLike[_time],
        /,
        out: OutT,
        *,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: np.ndarray[_Shape, np.dtype[_to_number]], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# ?bBhHiIlLqQefdgFDGO => ?O, where ?bBhHiIlLqQefdgFDG => ? and O => O (builtins.bool)
@type_check_only
class _ufunc_11_bifgco_bo(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_number]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
    @overload  # Nd object, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
    @overload  # scalar
    def __call__(
        self,
        x: complex | _to_number,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.bool: ...
    @overload  # scalar, out=...
    def __call__(
        self,
        x: complex | _to_number,
        /,
        *,
        out: EllipsisType,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array0D[np.bool]: ...
    @overload  # 1d
    def __call__(
        self,
        x: Sequence[complex | _to_number],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.bool]: ...
    @overload  # 2d
    def __call__(
        self,
        x: Sequence[Sequence[complex | _to_number]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.bool]: ...
    @overload  # 3d
    def __call__(
        self,
        x: Sequence[Sequence[Sequence[complex | _to_number]]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array3D[np.bool]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeNumber_co,
        /,
        out: OutT,
        *,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: np.ndarray[_Shape, np.dtype[_to_number | np.object_]], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# bBhHiIlLqQO => BO, where bBhHiIlLqQ => B and O => O
@type_check_only
class _ufunc_11_io(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_integer]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.uint8]]: ...
    @overload  # Nd object, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.object_]]: ...
    @overload  # scalar
    def __call__(
        self,
        x: int | _to_integer,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.uint8: ...
    @overload  # scalar, out=...
    def __call__(
        self,
        x: int | _to_integer,
        /,
        *,
        out: EllipsisType,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array0D[np.uint8]: ...
    @overload  # 1d
    def __call__(
        self,
        x: Sequence[int | _to_integer],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.uint8]: ...
    @overload  # 2d
    def __call__(
        self,
        x: Sequence[Sequence[int | _to_integer]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.uint8]: ...
    @overload  # 3d
    def __call__(
        self,
        x: Sequence[Sequence[Sequence[int | _to_integer]]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array3D[np.uint8]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeInt_co,
        /,
        out: OutT,
        *,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: np.ndarray[_Shape, np.dtype[_to_integer]], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# efdg => efdg
@type_check_only
class _ufunc_11_f(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # known shape, known scalar/array
    def __call__[T: np.floating | npt.NDArray[np.floating]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> T: ...
    @overload  # Nd, +f64
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_integer]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
    @overload  # scalar, float | +f64
    def __call__(
        self,
        x: float | _to_integer,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64: ...
    @overload  # 1d, +float
    def __call__(
        self,
        x: Sequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.float64]: ...
    @overload  # 2d, +float
    def __call__(
        self,
        x: Sequence[Sequence[float]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.float64]: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: np.floating](
        self,
        x: _FloatLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> ScalarT: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: np.floating](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_floating]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_floating]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: np.floating](
        self,
        x: _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: np.floating](
        self,
        x: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT] | Any: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> Any: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeFloat_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # out=<given>
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[np.bool | np.number], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# efdgO => efdgO
@type_check_only
class _ufunc_11_fo(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # known shape, known scalar/array
    def __call__[T: np.floating | npt.NDArray[np.floating | np.object_]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> T: ...
    @overload  # Nd, +f64
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.integer | np.bool]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
    @overload  # scalar, float | +f64
    def __call__(
        self,
        x: float | np.integer | np.bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64: ...
    @overload  # 1d, +float
    def __call__(
        self,
        x: Sequence[float],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.float64]: ...
    @overload  # 2d, +float
    def __call__(
        self,
        x: Sequence[Sequence[float]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.float64]: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: np.floating](
        self,
        x: _FloatLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> ScalarT: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: np.floating](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_floating | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_floating | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: np.floating | np.object_](
        self,
        x: _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: np.floating | np.object_](
        self,
        x: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT] | Any: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> Any: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeFloat_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # out=<given>
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[np.floating | np.object_], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# efdgFDGO => efdgFDGO
@type_check_only
class _ufunc_11_fco(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # known shape, known scalar/array
    def __call__[T: np.inexact | npt.NDArray[np.inexact | np.object_]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> T: ...
    @overload  # Nd, +f64
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.integer | np.bool]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
    @overload  # scalar, float | +f64
    def __call__(
        self,
        x: float | np.integer | np.bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64: ...
    @overload  # 1d, +float
    def __call__(
        self,
        x: Sequence[float],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.float64]: ...
    @overload  # 1d, ~complex
    def __call__(
        self,
        x: list[complex],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.complex128]: ...
    @overload  # 2d, +float
    def __call__(
        self,
        x: Sequence[Sequence[float]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.float64]: ...
    @overload  # 2d, ~complex
    def __call__(
        self,
        x: Sequence[list[complex]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.complex128]: ...
    @overload  # scalar, +complex  (overlaps with float)
    def __call__(
        self,
        x: complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.complex128 | Any: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: np.inexact](
        self,
        x: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> ScalarT: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: np.inexact](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: np.inexact | np.object_](
        self,
        x: _NestedSequence[complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: np.inexact | np.object_](
        self,
        x: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT] | Any: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> Any: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeNumber_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # out=<given>
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[np.inexact | np.object_], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# bBhHiIlLqQefdgFDGO => bBhHiIlLqQefdgFDGO
@type_check_only
class _ufunc_11_ifco(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # known shape, known scalar/array
    def __call__[T: np.number | npt.NDArray[np.number | np.object_]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> T: ...
    @overload  # Nd, bool->i8
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.bool]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.int8]]: ...
    @overload  # scalar, bool->i8
    def __call__(
        self,
        x: bool | np.bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.int8: ...
    @overload  # scalar, int  (overlaps with bool)
    def __call__(
        self,
        x: int,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.int_ | Any: ...
    @overload  # scalar, float  (overlaps with int)
    def __call__(
        self,
        x: float,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64 | Any: ...
    @overload  # scalar, complex  (overlaps with float)
    def __call__(
        self,
        x: complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.complex128 | Any: ...
    @overload  # 1d, bool
    def __call__(
        self,
        x: Sequence[bool],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.int8]: ...
    @overload  # 1d, ~int
    def __call__(
        self,
        x: list[int],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.int_]: ...
    @overload  # 1d, ~float
    def __call__(
        self,
        x: list[float],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.float64]: ...
    @overload  # 1d, ~complex
    def __call__(
        self,
        x: list[complex],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.complex128]: ...
    @overload  # 2d, bool
    def __call__(
        self,
        x: Sequence[Sequence[bool]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.int8]: ...
    @overload  # 2d, ~int
    def __call__(
        self,
        x: Sequence[list[int]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.int_]: ...
    @overload  # 2d, ~float
    def __call__(
        self,
        x: Sequence[list[float]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.float64]: ...
    @overload  # 2d, ~complex
    def __call__(
        self,
        x: Sequence[list[complex]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.complex128]: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: np.number](
        self,
        x: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> ScalarT: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: np.number | np.object_](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_number | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_number | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: np.number | np.object_](
        self,
        x: _NestedSequence[complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: np.number | np.object_](
        self,
        x: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT] | Any: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> Any: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeNumber_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[np.number | np.object_], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# bBhHiIlLqQefdgmFDGO => bBhHiIlLqQefdgFDGO, where m => d
@type_check_only
class _ufunc_11_ifcmo_ifco(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # known shape, known scalar/array
    def __call__[T: np.number | npt.NDArray[np.number | np.object_]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> T: ...
    @overload  # Nd, timedelta64
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.timedelta64]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
    @overload  # scalar, int
    def __call__(
        self,
        x: int,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.int_: ...
    @overload  # scalar, float  (overlaps with int)
    def __call__(
        self,
        x: float,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64 | Any: ...
    @overload  # scalar, complex  (overlaps with float)
    def __call__(
        self,
        x: complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.complex128 | Any: ...
    @overload  # scalar, timedelta64
    def __call__(
        self,
        x: np.timedelta64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64: ...
    @overload  # 1d, int
    def __call__(
        self,
        x: Sequence[int],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.int_]: ...
    @overload  # 1d, ~float | m
    def __call__(
        self,
        x: list[float] | Sequence[np.timedelta64],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.float64]: ...
    @overload  # 1d, ~complex
    def __call__(
        self,
        x: list[complex],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.complex128]: ...
    @overload  # 2d, int
    def __call__(
        self,
        x: Sequence[Sequence[int]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.int_]: ...
    @overload  # 2d, ~float | m
    def __call__(
        self,
        x: Sequence[list[float]] | Sequence[Sequence[np.timedelta64]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.float64]: ...
    @overload  # 2d, ~complex
    def __call__(
        self,
        x: Sequence[list[complex]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.complex128]: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: np.number](
        self,
        x: complex | _numeric,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> ScalarT: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: np.number | np.object_](
        self,
        x: np.ndarray[ShapeT, np.dtype[_numeric | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_numeric | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: np.number | np.object_](
        self,
        x: _NestedSequence[complex | _numeric],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[complex | _numeric],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: np.number | np.object_](
        self,
        x: _ArrayLikeNumericObj,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT] | Any: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeNumericObj,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> Any: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeNumericObj,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[np.number | np.object_], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# bBhHiIlLqQefdgmFDGO => bBhHiIlLqQefdgmFDGO
@type_check_only
class _ufunc_11_ifcmo(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # known shape, known scalar/array
    def __call__[T: _numeric | npt.NDArray[_numeric | np.object_]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> T: ...
    @overload  # scalar, int
    def __call__(
        self,
        x: int,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.int_: ...
    @overload  # scalar, float  (overlaps with int)
    def __call__(
        self,
        x: float,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64 | Any: ...
    @overload  # scalar, complex  (overlaps with float)
    def __call__(
        self,
        x: complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.complex128 | Any: ...
    @overload  # 1d, ~int
    def __call__(
        self,
        x: Sequence[int],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.int_]: ...
    @overload  # 1d, ~float
    def __call__(
        self,
        x: list[float],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.float64]: ...
    @overload  # 1d, ~complex
    def __call__(
        self,
        x: list[complex],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.complex128]: ...
    @overload  # 2d, ~int
    def __call__(
        self,
        x: Sequence[Sequence[int]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.int_]: ...
    @overload  # 2d, ~float
    def __call__(
        self,
        x: Sequence[list[float]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.float64]: ...
    @overload  # 2d, ~complex
    def __call__(
        self,
        x: Sequence[list[complex]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.complex128]: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: np.number](
        self,
        x: complex | _numeric,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> ScalarT: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: _numeric | np.object_](
        self,
        x: np.ndarray[ShapeT, np.dtype[_numeric | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_numeric | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: _numeric | np.object_](
        self,
        x: _NestedSequence[complex | _numeric],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[complex | _numeric],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: _numeric | np.object_](
        self,
        x: _ArrayLikeNumericObj,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT] | Any: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeNumericObj,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> Any: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeNumericObj,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[_numeric | np.object_], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# ?bBhHiIlLqQO => ?bBhHiIlLqQO
@type_check_only
class _ufunc_11_bio(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # known shape, known scalar/array
    def __call__[T: _to_integer | npt.NDArray[_to_integer | np.object_]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> T: ...
    @overload  # scalar, bool
    def __call__(
        self,
        x: bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.bool: ...
    @overload  # scalar, int (overlaps with bool)
    def __call__(
        self,
        x: int,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.int_ | Any: ...
    @overload  # 1d, bool
    def __call__(
        self,
        x: Sequence[bool],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.bool]: ...
    @overload  # 1d, ~int
    def __call__(
        self,
        x: list[int],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.int_]: ...
    @overload  # 1d, +int
    def __call__(
        self,
        x: Sequence[int],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.int_ | Any]: ...
    @overload  # 2d, bool
    def __call__(
        self,
        x: Sequence[Sequence[bool]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.bool]: ...
    @overload  # 2d, ~int
    def __call__(
        self,
        x: Sequence[list[int]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.int_]: ...
    @overload  # 2d, +int
    def __call__(
        self,
        x: Sequence[Sequence[int]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.int_ | Any]: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: _to_integer](
        self,
        x: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> ScalarT: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: _to_integer | np.object_](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_integer | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_integer | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: _to_integer | np.object_](
        self,
        x: _NestedSequence[int],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[int],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: _to_integer | np.object_](
        self,
        x: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT] | Any: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> Any: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeInt_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # out=<given>
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[_to_integer | np.object_], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# ?bBhHiIlLqQefdgO => ?bBhHiIlLqQefdgO
@type_check_only
class _ufunc_11_bifo(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # known shape, known scalar/array
    def __call__[T: _to_floating | npt.NDArray[_to_floating | np.object_]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> T: ...
    @overload  # scalar, bool
    def __call__(
        self,
        x: bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.bool: ...
    @overload  # scalar, int (overlaps with bool)
    def __call__(
        self,
        x: int,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.int_ | Any: ...
    @overload  # scalar, float (overlaps with int)
    def __call__(
        self,
        x: float,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64 | Any: ...
    @overload  # 1d, bool
    def __call__(
        self,
        x: Sequence[bool],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.bool]: ...
    @overload  # 1d, ~int
    def __call__(
        self,
        x: list[int],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.int_]: ...
    @overload  # 1d, ~float
    def __call__(
        self,
        x: list[float],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.float64]: ...
    @overload  # 1d, +float
    def __call__(
        self,
        x: Sequence[float],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.float64 | Any]: ...
    @overload  # 2d, bool
    def __call__(
        self,
        x: Sequence[Sequence[bool]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.bool]: ...
    @overload  # 2d, ~int
    def __call__(
        self,
        x: Sequence[list[int]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.int_]: ...
    @overload  # 2d, ~float
    def __call__(
        self,
        x: Sequence[list[float]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.float64]: ...
    @overload  # 2d, +float
    def __call__(
        self,
        x: Sequence[Sequence[float]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.float64 | Any]: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: _to_floating](
        self,
        x: _FloatLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> ScalarT: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: _to_floating | np.object_](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_floating | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_floating | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: _to_floating | np.object_](
        self,
        x: _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: _to_floating | np.object_](
        self,
        x: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT] | Any: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> Any: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeFloat_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # out=<given>
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[_to_floating | np.object_], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# ?bBhHiIlLqQefdgmFDGO => ?bBhHiIlLqQefdgmO, where F => f, D => d, G => g
@type_check_only
class _ufunc_11_bifcmo(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # known shape, known scalar/array
    def __call__[T: _to_floating | np.timedelta64 | npt.NDArray[_to_floating | np.timedelta64 | np.object_]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> T: ...
    @overload  # Nd, c128 -> f64
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.complex128]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
    @overload  # Nd, c64 -> f32
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.complex64]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.float32]]: ...
    @overload  # Nd, c160 -> f80
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.clongdouble]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.longdouble]]: ...
    @overload  # scalar, c128 -> f64
    def __call__(
        self,
        x: np.complex128,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64: ...
    @overload  # scalar, c64 -> f32
    def __call__(
        self,
        x: np.complex64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float32: ...
    @overload  # scalar, c160 -> f80
    def __call__(
        self,
        x: np.clongdouble,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.longdouble: ...
    @overload  # scalar, bool
    def __call__(
        self,
        x: bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.bool: ...
    @overload  # scalar, int (overlaps with bool)
    def __call__(
        self,
        x: int,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.int_ | Any: ...
    @overload  # scalar, float | complex (overlaps with int)
    def __call__(
        self,
        x: float | complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64 | Any: ...
    @overload  # 1d, bool
    def __call__(
        self,
        x: Sequence[bool],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.bool]: ...
    @overload  # 1d, ~int
    def __call__(
        self,
        x: list[int],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.int_]: ...
    @overload  # 1d, ~float | ~complex
    def __call__(
        self,
        x: list[float] | list[complex],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.float64]: ...
    @overload  # 1d, +complex
    def __call__(
        self,
        x: Sequence[complex],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.float64 | Any]: ...
    @overload  # 2d, bool
    def __call__(
        self,
        x: Sequence[Sequence[bool]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.bool]: ...
    @overload  # 2d, ~int
    def __call__(
        self,
        x: Sequence[list[int]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.int_]: ...
    @overload  # 2d, ~float | ~complex
    def __call__(
        self,
        x: Sequence[list[float]] | Sequence[list[complex]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.float64]: ...
    @overload  # 2d, +complex
    def __call__(
        self,
        x: Sequence[Sequence[complex]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.float64 | Any]: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: _to_floating | np.timedelta64](
        self,
        x: complex | _to_numeric,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> ScalarT: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: _to_floating | np.timedelta64 | np.object_](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_numeric]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[_to_numeric | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: _to_floating | np.object_](
        self,
        x: _NestedSequence[complex | _to_numeric],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[complex | _to_numeric],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: _to_floating | np.object_](
        self,
        x: _ArrayLikeNumericObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> npt.NDArray[ScalarT] | Any: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeNumericObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> Any: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeNumericObj_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # out=<given>
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[_to_floating | np.timedelta64 | np.object_], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# UT => ?;  identity=False
@type_check_only
class _ufunc_11_ut_b(_ufunc_11[Literal[False]]):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.str_] | np.dtypes.StringDType],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
    @overload  # scalar
    def __call__(
        self,
        x: str,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.bool: ...
    @overload  # scalar, out=...
    def __call__(
        self,
        x: str,
        /,
        *,
        out: EllipsisType,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array0D[np.bool]: ...
    @overload  # 1d  (`list` because `Sequence[str] :> str` would cause overlap)
    def __call__(
        self,
        x: list[str],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.bool]: ...
    @overload  # 2d
    def __call__(
        self,
        x: Sequence[list[str]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.bool]: ...
    @overload  # 3d
    def __call__(
        self,
        x: Sequence[Sequence[list[str]]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array3D[np.bool]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeStr_co | _ArrayLikeString_co,
        /,
        out: OutT,
        *,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: np.ndarray[_Shape, np.dtype[np.str_] | np.dtypes.StringDType], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# SUT => ?;  identity=False
@type_check_only
class _ufunc_11_sut_b(_ufunc_11[Literal[False]]):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.character] | np.dtypes.StringDType],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
    @overload  # scalar
    def __call__(
        self,
        x: _CharLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.bool: ...
    @overload  # scalar, out=...
    def __call__(
        self,
        x: _CharLike_co,
        /,
        *,
        out: EllipsisType,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array0D[np.bool]: ...
    @overload  # 1d
    def __call__(
        self,
        x: Sequence[_CharLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.bool]: ...
    @overload  # 2d
    def __call__(
        self,
        x: Sequence[Sequence[_CharLike_co]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.bool]: ...
    @overload  # 3d
    def __call__(
        self,
        x: Sequence[Sequence[Sequence[_CharLike_co]]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array3D[np.bool]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeAnyString_co,
        /,
        out: OutT,
        *,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: np.ndarray[_Shape, np.dtype[np.character] | np.dtypes.StringDType], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# SUT => n;  identity=0
@type_check_only
class _ufunc_11_sut_i(_ufunc_11[Literal[0]]):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.character] | np.dtypes.StringDType],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.ndarray[ShapeT, np.dtype[np.int_]]: ...
    @overload  # scalar
    def __call__(
        self,
        x: _CharLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.int_: ...
    @overload  # scalar, out=...
    def __call__(
        self,
        x: _CharLike_co,
        /,
        *,
        out: EllipsisType,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array0D[np.int_]: ...
    @overload  # 1d
    def __call__(
        self,
        x: Sequence[_CharLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array1D[np.int_]: ...
    @overload  # 2d
    def __call__(
        self,
        x: Sequence[Sequence[_CharLike_co]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array2D[np.int_]: ...
    @overload  # 3d
    def __call__(
        self,
        x: Sequence[Sequence[Sequence[_CharLike_co]]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array3D[np.int_]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeAnyString_co,
        /,
        out: OutT,
        *,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...
    @overload  # x.__array_ufunc__(...) -> OutT
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: np.ndarray[_Shape, np.dtype[np.character] | np.dtypes.StringDType], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

isnat: Final[_ufunc_11_m_b] = ...

signbit: Final[_ufunc_11_f_b] = ...

isfinite: Final[_ufunc_11_bifgcm_b] = ...
isinf: Final[_ufunc_11_bifgcm_b] = ...
isnan: Final[_ufunc_11_bifgcm_b] = ...  # TODO: StringDType[?] support

logical_not: Final[_ufunc_11_bifgco_bo] = ...

bitwise_count: Final[_ufunc_11_io] = ...

spacing: Final[_ufunc_11_f] = ...

cbrt: Final[_ufunc_11_fo] = ...
deg2rad: Final[_ufunc_11_fo] = ...
degrees: Final[_ufunc_11_fo] = ...
fabs: Final[_ufunc_11_fo] = ...
rad2deg: Final[_ufunc_11_fo] = ...
radians: Final[_ufunc_11_fo] = ...

arccos: Final[_ufunc_11_fco] = ...
arccosh: Final[_ufunc_11_fco] = ...
arcsin: Final[_ufunc_11_fco] = ...
arcsinh: Final[_ufunc_11_fco] = ...
arctan: Final[_ufunc_11_fco] = ...
arctanh: Final[_ufunc_11_fco] = ...
cos: Final[_ufunc_11_fco] = ...
cosh: Final[_ufunc_11_fco] = ...
exp: Final[_ufunc_11_fco] = ...
exp2: Final[_ufunc_11_fco] = ...
expm1: Final[_ufunc_11_fco] = ...
log: Final[_ufunc_11_fco] = ...
log10: Final[_ufunc_11_fco] = ...
log1p: Final[_ufunc_11_fco] = ...
log2: Final[_ufunc_11_fco] = ...
rint: Final[_ufunc_11_fco] = ...
sin: Final[_ufunc_11_fco] = ...
sinh: Final[_ufunc_11_fco] = ...
sqrt: Final[_ufunc_11_fco] = ...
tan: Final[_ufunc_11_fco] = ...
tanh: Final[_ufunc_11_fco] = ...

conjugate: Final[_ufunc_11_ifco] = ...
reciprocal: Final[_ufunc_11_ifco] = ...
square: Final[_ufunc_11_ifco] = ...

sign: Final[_ufunc_11_ifcmo_ifco] = ...

positive: Final[_ufunc_11_ifcmo] = ...
negative: Final[_ufunc_11_ifcmo] = ...

invert: Final[_ufunc_11_bio] = ...

ceil: Final[_ufunc_11_bifo] = ...
floor: Final[_ufunc_11_bifo] = ...
trunc: Final[_ufunc_11_bifo] = ...

absolute: Final[_ufunc_11_bifcmo] = ...

isdecimal: Final[_ufunc_11_ut_b] = ...
isnumeric: Final[_ufunc_11_ut_b] = ...

isalnum: Final[_ufunc_11_sut_b] = ...
isalpha: Final[_ufunc_11_sut_b] = ...
isdigit: Final[_ufunc_11_sut_b] = ...
islower: Final[_ufunc_11_sut_b] = ...
isspace: Final[_ufunc_11_sut_b] = ...
istitle: Final[_ufunc_11_sut_b] = ...
isupper: Final[_ufunc_11_sut_b] = ...

str_len: Final[_ufunc_11_sut_i] = ...

###
# re-exports from `_core._multiarray_umath` that are used by `_core._ufunc_config`

NAN: Final[float] = float("nan")
PINF: Final[float] = float("+inf")
NINF: Final[float] = float("-inf")
PZERO: Final[float] = +0.0
NZERO: Final[float] = -0.0
_UFUNC_API: Final[CapsuleType] = ...
_extobj_contextvar: Final[contextvars.ContextVar[CapsuleType]] = ...

def _get_extobj_dict() -> _ExtOjbDict: ...
def _make_extobj(*, all: _ErrKind = ..., **kwargs: Unpack[_ExtOjbDict]) -> CapsuleType: ...
