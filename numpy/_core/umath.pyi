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
    final,
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
    conj,
    divmod,
    e,
    euler_gamma,
    floor_divide,
    fmax,
    fmin,
    fmod,
    frompyfunc,
    matmul,
    matvec,
    maximum,
    minimum,
    mod,
    multiply,
    pi,
    power,
    remainder,
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
    _DTypeLikeFloat,
    _DTypeLikeObject,
    _FloatLike_co,
    _IntLike_co,
    _NestedSequence,
    _NumberLike_co,
    _ScalarLike_co,
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

_IdT_co = TypeVar("_IdT_co", covariant=True, default=None)
_ScalarT_contra = TypeVar("_ScalarT_contra", bound=np.generic, contravariant=True)

type _Array[ShapeT: _Shape, ScalarT: np.generic] = np.ndarray[ShapeT, np.dtype[ScalarT]]
type _Array0D[ScalarT: np.generic] = _Array[tuple[()], ScalarT]
type _Array1D[ScalarT: np.generic] = _Array[tuple[int], ScalarT]
type _Array2D[ScalarT: np.generic] = _Array[tuple[int, int], ScalarT]
type _Array3D[ScalarT: np.generic] = _Array[tuple[int, int, int], ScalarT]

type _tuple2[T] = tuple[T, T]

type _ErrKind = Literal["ignore", "warn", "raise", "call", "print", "log"]
type _ErrCall = Callable[[str, int], Any] | SupportsWrite[str]

type _to_integer = np.integer | np.bool
type _to_floating = np.floating | _to_integer
type _to_number = np.number | np.bool
type _to_numeric = _to_number | np.timedelta64
type _numeric = np.number | np.timedelta64
type _time = np.datetime64 | np.timedelta64
type _non_object = _to_number | _time | np.flexible  # np.generic - np.object_

type _as_f64 = np.int64 | np.uint64 | np.int32 | np.uint32
type _as_f32 = np.int16 | np.uint16
type _as_f16 = np.int8 | np.uint8 | np.bool

type _to_u8 = np.uint8 | np.bool
type _to_u16 = np.uint16 | _to_u8
type _to_u32 = np.uint32 | _to_u16
type _to_u64 = np.unsignedinteger | np.bool
type _to_i8 = np.int8 | np.bool
type _to_i16 = np.int16 | np.uint8 | _to_i8
type _to_i32 = np.int32 | np.uint16 | _to_i16
type _to_i64 = np.signedinteger | _to_u32  # exclude i64 * u64 -> f64
type _to_f32 = np.float32 | np.float16 | _as_f32 | _as_f16
type _to_f64 = np.float64 | np.float32 | np.float16 | _to_integer
type _to_c128 = np.complex128 | np.complex64 | _to_f64

type _ArrayLikeIntObj_co = _DualArrayLike[np.dtype[_to_integer | np.object_], int]
type _ArrayLikeNumberObj_co = _DualArrayLike[np.dtype[_to_number | np.object_], complex]
type _ArrayLikeNumericObj = _DualArrayLike[np.dtype[_numeric | np.object_], complex]
type _ArrayLikeNumericObj_co = _DualArrayLike[np.dtype[_to_numeric | np.object_], complex]

type _ArrayUnlikeObject = _DualArrayLike[np.dtype[_non_object] | np.dtypes.StringDType, complex | bytes | str]
type _ScalarUnlikeObject = complex | bytes | str | np.generic  # bare `np.object_` don't exist

@type_check_only
class _ExtOjbDict(TypedDict, total=False):
    divide: _ErrKind
    over: _ErrKind
    under: _ErrKind
    invalid: _ErrKind
    call: _ErrCall | None
    bufsize: int

type _Signature1 = tuple[str | None, str | None] | str
type _Signature2 = tuple[str | None, str | None, str | None] | str

# TODO(@jorenham): make these `__array_ufunc__` protocols generic on `ufunc` so that
# implementing types can structurally determine the signature of the calling `ufunc`.

@type_check_only
class _CanUfuncCall1[OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["__call__"], x: Self, /) -> OutT: ...

@type_check_only
class _CanUfuncAt1[IxT, OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["at"], a: Self, indices: IxT, /) -> OutT: ...

@type_check_only
class _CanUfuncCall2L[OtherT, OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["__call__"], lhs: Self, rhs: OtherT, /) -> OutT: ...

@type_check_only
class _CanUfuncCall2R[OtherT, OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["__call__"], lhs: OtherT, rhs: Self, /) -> OutT: ...

@type_check_only
class _CanUfuncOuterL[OtherT, OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["outer"], lhs: Self, rhs: OtherT, /) -> OutT: ...

@type_check_only
class _CanUfuncOuterR[OtherT, OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["outer"], lhs: OtherT, rhs: Self, /) -> OutT: ...

@type_check_only
class _CanUfuncAt2L[OtherT, IxT, OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["at"], a: Self, indices: IxT, b: OtherT, /) -> OutT: ...

@type_check_only
class _CanUfuncAt2R[OtherT, IxT, OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["at"], a: OtherT, indices: IxT, b: Self, /) -> OutT: ...

@type_check_only
class _CanUfuncAccumulate[OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["accumulate"], a: Self, /) -> OutT: ...

@type_check_only
class _CanUfuncReduce[OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["reduce"], a: Self, /) -> OutT: ...

@type_check_only
class _CanUfuncReduceAt[IxT, OutT](Protocol):
    def __array_ufunc__(self, ufunc: np.ufunc, method: Literal["reduceat"], a: Self, indices: IxT, /) -> OutT: ...

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

@type_check_only
class _Kwargs12(TypedDict, total=False):
    where: _ArrayLikeBool_co  # = True
    casting: _CastingKind  # = "same_kind"
    order: _OrderKACF  # = "K",
    subok: bool  # = True,
    signature: _Signature2

@type_check_only
class _Kwargs21(TypedDict, total=False):
    where: _ArrayLikeBool_co  # = True
    casting: _CastingKind  # = "same_kind"
    order: _OrderKACF  # = "K",
    subok: bool  # = True,
    signature: _Signature2

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

@type_check_only
class _ufunc_12(np.ufunc):  # type: ignore[misc]
    @property
    @override
    def identity(self) -> None: ...
    @property
    @override
    def nin(self) -> Literal[1]: ...
    @property
    @override
    def nout(self) -> Literal[2]: ...
    @property
    @override
    def nargs(self) -> Literal[3]: ...
    @property
    @override
    def signature(self) -> None: ...

    #
    @final
    @override
    def at(self, a: Never, indices: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]
    @final
    @override
    def accumulate(self, array: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]
    @final
    @override
    def reduce(self, array: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]
    @final
    @override
    def reduceat(self, array: Never, /, indices: Never) -> Never: ...  # pyrefly:ignore[bad-override]
    @final
    @override
    def outer(self, A: Never, B: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]

@type_check_only
class _ufunc_21(np.ufunc, Generic[_IdT_co]):  # type: ignore[misc]
    @property
    @override
    def identity(self) -> _IdT_co: ...
    @property
    @override
    def nin(self) -> Literal[2]: ...
    @property
    @override
    def nout(self) -> Literal[1]: ...
    @property
    @override
    def nargs(self) -> Literal[3]: ...
    @property
    @override
    def signature(self) -> None: ...

# Mm => ?
@type_check_only
class _ufunc_11_m_b(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _time],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.bool]: ...
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
    def at(self, a: npt.NDArray[_time], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# efdg => ?
@type_check_only
class _ufunc_11_f_b(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_floating],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.bool]: ...
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
    def at(self, a: npt.NDArray[np.floating], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# ?bBhHiIlLqQefdgFDGmM => ?
@type_check_only
class _ufunc_11_bifgcm_b(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_number | _time],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.bool]: ...
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
    def at(self, a: npt.NDArray[_to_number], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# ?bBhHiIlLqQefdgFDGO => ?O, where ?bBhHiIlLqQefdgFDG => ? and O => O (builtins.bool)
@type_check_only
class _ufunc_11_bifgco_bo(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_number],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.bool]: ...
    @overload  # Nd object, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.object_]: ...
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
    def at(self, a: npt.NDArray[_to_number | np.object_], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

# bBhHiIlLqQO => BO, where bBhHiIlLqQ => B and O => O
@type_check_only
class _ufunc_11_io(_ufunc_11):  # type: ignore[misc]
    @override
    @overload  # Nd, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_integer],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.uint8]: ...
    @overload  # Nd object, known shape
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.object_]: ...
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
    def at(self, a: npt.NDArray[_to_integer], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
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
        x: _Array[ShapeT, _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float64]: ...
    @overload  # Nd, +f32
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float32]: ...
    @overload  # Nd, +f16
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f16],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float16]: ...
    @overload  # scalar, float | +f64
    def __call__(
        self,
        x: float | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64: ...
    @overload  # scalar, +f32
    def __call__(
        self,
        x: _as_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float32: ...
    @overload  # scalar, +f16
    def __call__(
        self,
        x: _as_f16,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float16: ...
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
        x: _Array[ShapeT, _to_floating],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_floating],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, Any]: ...
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
        x: _Array[ShapeT, _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float64]: ...
    @overload  # Nd, +f32
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float32]: ...
    @overload  # Nd, +f16
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f16],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float16]: ...
    @overload  # scalar, float | +f64
    def __call__(
        self,
        x: float | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64: ...
    @overload  # scalar, +f32
    def __call__(
        self,
        x: _as_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float32: ...
    @overload  # scalar, +f16
    def __call__(
        self,
        x: _as_f16,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float16: ...
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
        x: _Array[ShapeT, _to_floating | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_floating | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, Any]: ...
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
    ) -> npt.NDArray[Any]: ...
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
        x: _Array[ShapeT, _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float64]: ...
    @overload  # Nd, +f32
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float32]: ...
    @overload  # Nd, +f16
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f16],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float16]: ...
    @overload  # scalar, float | +f64
    def __call__(
        self,
        x: float | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float64: ...
    @overload  # scalar, +f32
    def __call__(
        self,
        x: _as_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float32: ...
    @overload  # scalar, +f16
    def __call__(
        self,
        x: _as_f16,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> np.float16: ...
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
        x: _Array[ShapeT, np.number | np.bool | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, np.number | np.bool | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, Any]: ...
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
        x: _Array[ShapeT, np.bool],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.int8]: ...
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
        x: _Array[ShapeT, _to_number | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_number | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, Any]: ...
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
        x: _Array[ShapeT, np.timedelta64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float64]: ...
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
        x: _Array[ShapeT, _numeric | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _numeric | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, Any]: ...
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
        x: _Array[ShapeT, _numeric | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _numeric | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, Any]: ...
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
        x: _Array[ShapeT, _to_integer | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_integer | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, Any]: ...
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
        x: _Array[ShapeT, _to_floating | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_floating | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, Any]: ...
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
        x: _Array[ShapeT, np.complex128],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float64]: ...
    @overload  # Nd, c64 -> f32
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, np.complex64],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.float32]: ...
    @overload  # Nd, c160 -> f80
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, np.clongdouble],
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, np.longdouble]: ...
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
        x: _Array[ShapeT, _to_numeric],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_numeric | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs11],
    ) -> _Array[ShapeT, Any]: ...
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
    ) -> _Array[ShapeT, np.bool]: ...
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
    ) -> _Array[ShapeT, np.bool]: ...
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
    ) -> _Array[ShapeT, np.int_]: ...
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

# efdg => (efdg, i)
@type_check_only
class _ufunc_12_frexp(_ufunc_12):  # type: ignore[misc]
    @override
    @overload  # Nd, known dtype
    def __call__[ShapeT: _Shape, DTypeT: np.dtype[np.floating]](
        self,
        x: np.ndarray[ShapeT, DTypeT],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[np.ndarray[ShapeT, DTypeT], _Array[ShapeT, np.int32]]: ...
    @overload  # Nd, +f64
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[_Array[ShapeT, np.float64], _Array[ShapeT, np.int32]]: ...
    @overload  # Nd, +f32
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[_Array[ShapeT, np.float32], _Array[ShapeT, np.int32]]: ...
    @overload  # Nd, +f16
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f16],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[_Array[ShapeT, np.float16], _Array[ShapeT, np.int32]]: ...
    @overload  # scalar, known dtype
    def __call__[ScalarT: np.floating](
        self,
        x: ScalarT,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[ScalarT, np.int32]: ...
    @overload  # scalar, float | +f64
    def __call__(
        self,
        x: float | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[np.float64, np.int32]: ...
    @overload  # scalar, +f32
    def __call__(
        self,
        x: _as_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[np.float32, np.int32]: ...
    @overload  # scalar, +f16
    def __call__(
        self,
        x: _as_f16,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[np.float16, np.int32]: ...
    @overload  # 1d, +float
    def __call__(
        self,
        x: Sequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[_Array1D[np.float64], _Array1D[np.int32]]: ...
    @overload  # 2d, +float
    def __call__(
        self,
        x: Sequence[Sequence[float]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[_Array2D[np.float64], _Array2D[np.int32]]: ...
    @overload  # ?d, unknown dtype
    def __call__(
        self,
        x: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[Any]: ...
    @overload  # out=<given>
    def __call__[OutT1: np.ndarray, OutT2: np.ndarray](
        self,
        x: _ArrayLikeFloat_co,
        /,
        out: tuple[OutT1, OutT2],
        *,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[OutT1, OutT2]: ...
    @overload  # out=<given>
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> OutT: ...

# efdg => (efdg, efdg)
@type_check_only
class _ufunc_12_modf(_ufunc_12):  # type: ignore[misc]
    @override
    @overload  # known shape, known scalar/array
    def __call__[T: np.floating | npt.NDArray[np.floating]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[T]: ...
    @overload  # Nd, +f64
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[_Array[ShapeT, np.float64]]: ...
    @overload  # Nd, +f32
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[_Array[ShapeT, np.float32]]: ...
    @overload  # Nd, +f16
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _as_f16],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[_Array[ShapeT, np.float16]]: ...
    @overload  # scalar, float | +f64
    def __call__(
        self,
        x: float | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[np.float64]: ...
    @overload  # scalar, +f32
    def __call__(
        self,
        x: _as_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[np.float32]: ...
    @overload  # scalar, +f16
    def __call__(
        self,
        x: _as_f16,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[np.float16]: ...
    @overload  # 1d, +float
    def __call__(
        self,
        x: Sequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[_Array1D[np.float64]]: ...
    @overload  # 2d, +float
    def __call__(
        self,
        x: Sequence[Sequence[float]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[_Array2D[np.float64]]: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: np.floating](
        self,
        x: _FloatLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs11],
    ) -> _tuple2[ScalarT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: np.floating](
        self,
        x: _Array[ShapeT, _to_floating],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[_Array[ShapeT, ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: _Array[ShapeT, _to_floating],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[_Array[ShapeT, Any]]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: np.floating](
        self,
        x: _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[npt.NDArray[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[np.ndarray]: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: np.floating](
        self,
        x: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[npt.NDArray[ScalarT] | Any]: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> _tuple2[Any]: ...
    @overload  # out=<given>
    def __call__[OutT1: np.ndarray, OutT2: np.ndarray](
        self,
        x: _ArrayLikeFloat_co,
        /,
        out: tuple[OutT1, OutT2],
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> tuple[OutT1, OutT2]: ...
    @overload  # out=<given>
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs12],
    ) -> OutT: ...

frexp: Final[_ufunc_12_frexp] = ...
modf: Final[_ufunc_12_modf] = ...

# ?bBhHiIlLqQefdgFDGOSUVT, ?bBhHiIlLqQefdgFDGOSUVT => ?O;  (also supports `mM` input)
@type_check_only
class _ufunc_21_logical[IdT: bool](_ufunc_21[IdT]):  # type: ignore[misc]
    # TODO(@jorenham): shape-typing

    @override
    @overload  # 0d, 0d
    def __call__(
        self,
        x1: _ScalarUnlikeObject,
        x2: _ScalarUnlikeObject,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.bool: ...
    @overload  # >0d, >=0d
    def __call__(
        self,
        x1: np.ndarray[Any, np.dtype[_non_object] | np.dtypes.StringDType] | _NestedSequence[_ScalarUnlikeObject],
        x2: _ArrayUnlikeObject,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # >=0d, >0d
    def __call__(
        self,
        x1: _ArrayUnlikeObject,
        x2: np.ndarray[Any, np.dtype[_non_object] | np.dtypes.StringDType] | _NestedSequence[_ScalarUnlikeObject],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # >0d object_, >=0d
    def __call__(
        self,
        x1: npt.NDArray[np.object_],
        x2: npt.ArrayLike,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeObject | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # >=0d, >0d object_
    def __call__(
        self,
        x1: npt.ArrayLike,
        x2: npt.NDArray[np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeObject | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # >=0d, >=0d, out=...
    def __call__(
        self,
        x1: _ArrayUnlikeObject,
        x2: _ArrayUnlikeObject,
        /,
        *,
        out: EllipsisType,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x1: npt.ArrayLike,
        x2: npt.ArrayLike,
        /,
        out: OutT,
        *,
        dtype: _DTypeLikeBool | _DTypeLikeObject | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x1.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: _CanUfuncCall2L[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncCall2R[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # +object, +object (overlaps with, well, everything...)
    def __call__(
        self,
        x1: object,
        x2: object,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeObject | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> Any: ...

    #
    @override
    @overload  # 0d, 0d
    def outer(
        self,
        x1: _ScalarUnlikeObject,
        x2: _ScalarUnlikeObject,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.bool: ...
    @overload  # >0d, >=0d
    def outer(
        self,
        x1: np.ndarray[Any, np.dtype[_non_object] | np.dtypes.StringDType] | _NestedSequence[_ScalarUnlikeObject],
        x2: _ArrayUnlikeObject,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # >=0d, >0d
    def outer(
        self,
        x1: _ArrayUnlikeObject,
        x2: np.ndarray[Any, np.dtype[_non_object] | np.dtypes.StringDType] | _NestedSequence[_ScalarUnlikeObject],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # >0d object, >=0d
    def outer(
        self,
        x1: npt.NDArray[np.object_],
        x2: npt.ArrayLike,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeObject | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # >=0d, >0d object
    def outer(
        self,
        x1: npt.ArrayLike,
        x2: npt.NDArray[np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeObject | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # >=0d, >=0d, out=<given>
    def outer[OutT: np.ndarray](
        self,
        x1: npt.ArrayLike,
        x2: npt.ArrayLike,
        /,
        *,
        out: OutT,
        dtype: _DTypeLikeBool | _DTypeLikeObject | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # >=0d, >=0d  (fallback)
    def outer(
        self,
        x1: _ArrayUnlikeObject,
        x2: _ArrayUnlikeObject,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool] | Any: ...
    @overload  # x1.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: _CanUfuncOuterL[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncOuterR[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # +object, +object (overlaps with everything...)
    def outer(
        self,
        x1: object,
        x2: object,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeObject | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> Any: ...

    #
    @override
    @overload
    def at(self, a: np.ndarray, indices: _ArrayLikeInt, b: npt.ArrayLike, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[OtherT, IxT, OutT](self, a: _CanUfuncAt2L[OtherT, IxT, OutT], indices: IxT, b: OtherT, /) -> OutT: ...
    @overload
    def at[OtherT, IxT, OutT](self, a: OtherT, indices: IxT, b: _CanUfuncAt2R[OtherT, IxT, OutT], /) -> OutT: ...

    #
    @override
    @overload  # unknown shape, not object_
    def reduce(  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayUnlikeObject,
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.bool] | Any: ...
    @overload  # unknown shape, not object_, axis=None
    def reduce(
        self,
        array: _ArrayUnlikeObject,
        /,
        *,
        axis: None,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.bool: ...
    @overload  # unknown shape, not object_ keepdims=True
    def reduce(
        self,
        array: _ArrayUnlikeObject,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
        keepdims: Literal[True],
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # unknown shape, not object_, out=...
    def reduce(
        self,
        array: _ArrayUnlikeObject,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLikeBool | None = None,
        out: EllipsisType,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # unknown shape, object_
    def reduce(
        self,
        array: npt.NDArray[np.object_],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: _DTypeLikeObject | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.object_] | Any: ...
    @overload  # unknown shape, object_, axis=None
    def reduce(
        self,
        array: npt.NDArray[np.object_],
        /,
        *,
        axis: None,
        dtype: _DTypeLikeObject | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> Any: ...
    @overload  # unknown shape, not object_ keepdims=True
    def reduce(
        self,
        array: npt.NDArray[np.object_],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLikeObject | None = None,
        out: None = None,
        keepdims: Literal[True],
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.object_]: ...
    @overload  # unknown shape, object_, out=...
    def reduce(
        self,
        array: npt.NDArray[np.object_],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLikeBool | None = None,
        out: EllipsisType,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.object_]: ...
    @overload  # out=<given>
    def reduce[OutT: np.ndarray](
        self,
        array: npt.ArrayLike,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLikeBool | _DTypeLikeObject | None = None,
        out: OutT,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduceat", array, ...)
    def reduce[OutT](
        self,
        array: _CanUfuncReduce[OutT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | EllipsisType | None = None,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...

    #
    @override
    @overload  # known shape
    def reduceat[ShapeT: _Shape](  # pyrefly:ignore[bad-override]
        self,
        array: np.ndarray[ShapeT, np.dtype[_non_object] | np.dtypes.StringDType],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
    ) -> _Array[ShapeT, np.bool]: ...
    @overload  # known shape, object_
    def reduceat[ShapeT: _Shape](  # pyrefly:ignore[bad-override]
        self,
        array: _Array[ShapeT, np.object_],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLikeObject | None = None,
        out: None = None,
    ) -> _Array[ShapeT, np.object_]: ...
    @overload  # unknown shape
    def reduceat(
        self,
        array: _ArrayUnlikeObject,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # out=<given>
    def reduceat[OutT: np.ndarray](
        self,
        array: npt.ArrayLike,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | _DTypeLikeObject | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduceat", array, ...)
    def reduceat[IxT, OutT](
        self,
        array: _CanUfuncReduceAt[IxT, OutT],
        /,
        indices: IxT,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

    # TODO
    @override
    @overload  # known shape
    def accumulate[ShapeT: _Shape](  # pyrefly:ignore[bad-override]
        self,
        array: np.ndarray[ShapeT, np.dtype[_non_object] | np.dtypes.StringDType],
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
    ) -> _Array[ShapeT, np.bool]: ...
    @overload  # known shape, OBJECT_
    def accumulate[ShapeT: _Shape](
        self,
        array: _Array[ShapeT, np.object_],
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLikeObject | None = None,
        out: None = None,
    ) -> _Array[ShapeT, np.object_]: ...
    @overload  # unknown shape
    def accumulate(
        self,
        array: _ArrayUnlikeObject,
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # out=<given>
    def accumulate[OutT: np.ndarray](
        self,
        array: npt.ArrayLike,
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | _DTypeLikeObject | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "accumulate", array, ...)
    def accumulate[OutT](
        self,
        array: _CanUfuncAccumulate[OutT],
        /,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

# ?bBhHiIlLqQefdgFDGMmOSUT => ?O
# NOTE: These theoretically also supports O => O but that's an unlikely usecase because
# that'd require something incompatible with `bool` or `np.bool` to be returned from the
# respective "rich" comparison dunder methods of the underlying python objects.
# So for the sake of simplicity we assume that all `np.generic` are accepted and that
# the output dtype is always `np.bool`.
@type_check_only
class _ufunc_21_cmp(_ufunc_21[None]):  # type: ignore[misc]
    # TODO(@jorenham): shape-typing

    @override
    @overload  # 0d, 0d
    def __call__(
        self,
        x1: _ScalarLike_co,
        x2: _ScalarLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.bool: ...
    @overload  # >0d, >=0d
    def __call__(
        self,
        x1: np.ndarray | _NestedSequence[_ScalarLike_co],
        x2: npt.ArrayLike,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # >=0d, >0d
    def __call__(
        self,
        x1: npt.ArrayLike,
        x2: np.ndarray | _NestedSequence[_ScalarLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # >=0d, >=0d, out=...
    def __call__(
        self,
        x1: npt.ArrayLike,
        x2: npt.ArrayLike,
        /,
        *,
        out: EllipsisType,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x1: npt.ArrayLike,
        x2: npt.ArrayLike,
        /,
        out: OutT,
        *,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x1.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: _CanUfuncCall2L[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncCall2R[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    #
    @override
    @overload  # 0d, 0d
    def outer(
        self,
        x1: _ScalarLike_co,
        x2: _ScalarLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.bool: ...
    @overload  # >0d, >=0d
    def outer(
        self,
        x1: np.ndarray | _NestedSequence[_ScalarLike_co],
        x2: npt.ArrayLike,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # >=0d, >0d
    def outer(
        self,
        x1: npt.ArrayLike,
        x2: np.ndarray | _NestedSequence[_ScalarLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # >=0d, >=0d, out=<given>
    def outer[OutT: np.ndarray](
        self,
        x1: npt.ArrayLike,
        x2: npt.ArrayLike,
        /,
        *,
        out: OutT,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # >=0d, >=0d  (fallback)
    def outer(
        self,
        x1: npt.ArrayLike,
        x2: npt.ArrayLike,
        /,
        *,
        out: None = None,
        dtype: _DTypeLikeBool | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool] | Any: ...
    @overload  # x1.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: _CanUfuncOuterL[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncOuterR[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: np.ndarray, indices: _ArrayLikeInt, b: npt.ArrayLike, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[OtherT, IxT, OutT](self, a: _CanUfuncAt2L[OtherT, IxT, OutT], indices: IxT, b: OtherT, /) -> OutT: ...
    @overload
    def at[OtherT, IxT, OutT](self, a: OtherT, indices: IxT, b: _CanUfuncAt2R[OtherT, IxT, OutT], /) -> OutT: ...

    #
    @override
    @overload  # unknown shape
    def reduce(  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.bool] | np.bool: ...
    @overload  # unknown shape, axis=None
    def reduce(
        self,
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: None,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.bool: ...
    @overload  # unknown shape, keepdims=True
    def reduce(
        self,
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
        keepdims: Literal[True],
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # unknown shape, out=...
    def reduce(
        self,
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLikeBool | None = None,
        out: EllipsisType,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # out=<given>
    def reduce[OutT: np.ndarray](
        self,
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLikeBool | None = None,
        out: OutT,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduceat", array, ...)
    def reduce[OutT](
        self,
        array: _CanUfuncReduce[OutT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | EllipsisType | None = None,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...

    #
    @override
    @overload  # known shape
    def reduceat[ShapeT: _Shape](  # pyrefly:ignore[bad-override]
        self,
        array: _Array[ShapeT, np.bool],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
    ) -> _Array[ShapeT, np.bool]: ...
    @overload  # unknown shape
    def reduceat(
        self,
        array: _ArrayLikeBool_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # out=<given>
    def reduceat[OutT: np.ndarray](
        self,
        array: _ArrayLikeBool_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduceat", array, ...)
    def reduceat[IxT, OutT](
        self,
        array: _CanUfuncReduceAt[IxT, OutT],
        /,
        indices: IxT,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

    #
    @override
    @overload  # known shape
    def accumulate[ShapeT: _Shape](  # pyrefly:ignore[bad-override]
        self,
        array: _Array[ShapeT, np.bool],
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
    ) -> _Array[ShapeT, np.bool]: ...
    @overload  # unknown shape
    def accumulate(
        self,
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | None = None,
        out: None = None,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # out=<given>
    def accumulate[OutT: np.ndarray](
        self,
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLikeBool | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "accumulate", array, ...)
    def accumulate[OutT](
        self,
        array: _CanUfuncAccumulate[OutT],
        /,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

# efdg, il => efdg
@type_check_only
class _ufunc_21_ldexp(_ufunc_21[None]):  # type: ignore[misc]
    # TODO(@jorenham): shape-typing

    @override
    @overload  # 0d T@floating, 0d
    def __call__[ScalarT: np.floating](
        self,
        x1: ScalarT,
        x2: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d float | +f64, 0d
    def __call__(
        self,
        x1: float | _as_f64,
        x2: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d +f32, 0d
    def __call__(
        self,
        x1: _as_f32,
        x2: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float32: ...
    @overload  # 0d +f16, 0d
    def __call__(
        self,
        x1: _as_f16,
        x2: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float16: ...
    @overload  # >0d T@floating, >=0d
    def __call__[ScalarT: np.floating](
        self,
        x1: npt.NDArray[ScalarT] | _NestedSequence[ScalarT],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # >0d +f64, >=0d
    def __call__(
        self,
        x1: npt.NDArray[_as_f64] | _NestedSequence[float | _as_f64],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # >0d +f32, >=0d
    def __call__(
        self,
        x1: npt.NDArray[_as_f32] | _NestedSequence[_as_f32],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # >0d +f16, >=0d
    def __call__(
        self,
        x1: npt.NDArray[_as_f16] | _NestedSequence[_as_f16],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float16]: ...
    @overload  # >=0d T@floating, >0d
    def __call__[ScalarT: np.floating](
        self,
        x1: _ArrayLike[ScalarT],
        x2: npt.NDArray[_to_integer] | _NestedSequence[_IntLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # >=0d +f64, >0d
    def __call__(
        self,
        x1: _DualArrayLike[np.dtype[_as_f64], float],
        x2: npt.NDArray[_to_integer] | _NestedSequence[_IntLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # >=0d +f32, >0d
    def __call__(
        self,
        x1: _ArrayLike[_as_f32],
        x2: npt.NDArray[_to_integer] | _NestedSequence[_IntLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # >=0d +f32, >0d
    def __call__(
        self,
        x1: _ArrayLike[_as_f16],
        x2: npt.NDArray[_to_integer] | _NestedSequence[_IntLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float16]: ...
    @overload  # >=0d T@floating, >=0d, out=...
    def __call__[ScalarT: np.floating](
        self,
        x1: _ArrayLike[ScalarT],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # >=0d +f64, >=0d, out=...
    def __call__(
        self,
        x1: _DualArrayLike[np.dtype[_as_f64], float],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # >=0d +f32, >=0d, out=...
    def __call__(
        self,
        x1: _ArrayLike[_as_f32],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # >=0d +f16, >=0d, out=...
    def __call__(
        self,
        x1: _ArrayLike[_as_f16],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float16]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x1: _ArrayLikeFloat_co,
        x2: _ArrayLikeInt_co,
        /,
        out: OutT,
        *,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x1.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: _CanUfuncCall2L[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncCall2R[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    #
    @override
    @overload  # 0d T@floating, 0d
    def outer[ScalarT: np.floating](  # pyrefly:ignore[bad-override]
        self,
        x1: ScalarT,
        x2: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d float | +f64, 0d
    def outer(
        self,
        x1: float | _as_f64,
        x2: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d +f32, 0d
    def outer(
        self,
        x1: _as_f32,
        x2: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float32: ...
    @overload  # 0d +f16, 0d
    def outer(
        self,
        x1: _as_f16,
        x2: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float16: ...
    @overload  # >0d T@floating, >=0d
    def outer[ScalarT: np.floating](
        self,
        x1: npt.NDArray[ScalarT] | _NestedSequence[ScalarT],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # >0d +f64, >=0d
    def outer(
        self,
        x1: npt.NDArray[_as_f64] | _NestedSequence[float | _as_f64],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # >0d +f32, >=0d
    def outer(
        self,
        x1: npt.NDArray[_as_f32] | _NestedSequence[_as_f32],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # >0d +f16, >=0d
    def outer(
        self,
        x1: npt.NDArray[_as_f16] | _NestedSequence[_as_f16],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float16]: ...
    @overload  # >=0d T@floating, >0d
    def outer[ScalarT: np.floating](
        self,
        x1: _ArrayLike[ScalarT],
        x2: npt.NDArray[_to_integer] | _NestedSequence[_IntLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # >=0d +f64, >0d
    def outer(
        self,
        x1: _DualArrayLike[np.dtype[_as_f64], float],
        x2: npt.NDArray[_to_integer] | _NestedSequence[_IntLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # >=0d +f32, >0d
    def outer(
        self,
        x1: _ArrayLike[_as_f32],
        x2: npt.NDArray[_to_integer] | _NestedSequence[_IntLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # >=0d +f16, >0d
    def outer(
        self,
        x1: _ArrayLike[_as_f16],
        x2: npt.NDArray[_to_integer] | _NestedSequence[_IntLike_co],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float16]: ...
    @overload  # >=0d T@floating, >=0d, out=...
    def outer[ScalarT: np.floating](
        self,
        x1: _ArrayLike[ScalarT],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # >=0d +f64, >=0d, out=...
    def outer(
        self,
        x1: _DualArrayLike[np.dtype[_as_f64], float],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # >=0d +f32, >=0d, out=...
    def outer(
        self,
        x1: _ArrayLike[_as_f32],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # >=0d +f16, >=0d, out=...
    def outer(
        self,
        x1: _ArrayLike[_as_f16],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float16]: ...
    @overload  # out=<given>
    def outer[OutT: np.ndarray](
        self,
        x1: _ArrayLikeFloat_co,
        x2: _ArrayLikeInt_co,
        /,
        out: OutT,
        *,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x1.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: _CanUfuncCall2L[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncCall2R[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[_to_floating], indices: _ArrayLikeInt, b: _ArrayLikeInt_co, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[OtherT, IxT, OutT](self, a: _CanUfuncAt2L[OtherT, IxT, OutT], indices: IxT, b: OtherT, /) -> OutT: ...
    @overload
    def at[OtherT, IxT, OutT](self, a: OtherT, indices: IxT, b: _CanUfuncAt2R[OtherT, IxT, OutT], /) -> OutT: ...

    #
    @override
    @overload  # unknown shape, +f64
    def reduce(  # pyrefly:ignore[bad-override]
        self,
        array: _DualArrayLike[np.dtype[_as_f64], float],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64] | np.float64: ...
    @overload  # unknown shape, +f64, axis=None
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_as_f64], float],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.float64: ...
    @overload  # unknown shape, +f64, keepdims=True
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_as_f64], float],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[True],
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # unknown shape, +f64, out=...
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_as_f64], float],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # unknown shape, +f32
    def reduce(
        self,
        array: _ArrayLike[_as_f32],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float32] | np.float32: ...
    @overload  # unknown shape, +f32, axis=None
    def reduce(
        self,
        array: _ArrayLike[_as_f32],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.float32: ...
    @overload  # unknown shape, +f32, keepdims=True
    def reduce(
        self,
        array: _ArrayLike[_as_f32],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[True],
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float32]: ...
    @overload  # unknown shape, +f32, out=...
    def reduce(
        self,
        array: _ArrayLike[_as_f32],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float32]: ...
    @overload  # unknown shape, +f16
    def reduce(
        self,
        array: _ArrayLike[_as_f16],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float16] | np.float16: ...
    @overload  # unknown shape, +f16, axis=None
    def reduce(
        self,
        array: _ArrayLike[_as_f16],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.float16: ...
    @overload  # unknown shape, +f16, keepdims=True
    def reduce(
        self,
        array: _ArrayLike[_as_f16],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[True],
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float16]: ...
    @overload  # unknown shape, +f16, out=...
    def reduce(
        self,
        array: _ArrayLike[_as_f16],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float16]: ...
    @overload  # out=<given>
    def reduce[OutT: np.ndarray](
        self,
        array: _ArrayLikeInt,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: OutT,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduce", array, ...)
    def reduce[OutT](
        self,
        array: _CanUfuncReduce[OutT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | EllipsisType | None = None,
        keepdims: bool = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...

    @override
    def reduceat(self, array: Never, /, indices: Never) -> Never: ...  # pyrefly:ignore[bad-override]
    @override
    def accumulate(self, array: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]

# dgDG, dgDG => dgDG
@type_check_only
class _ufunc_21_float_power(_ufunc_21[None]):  # type: ignore[misc]
    @override
    @overload  # 0d +f64, 0d +f64
    def __call__(
        self,
        x1: float | _to_f64,
        x2: float | _to_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~f80, 0d +f80
    def __call__(
        self,
        x1: np.longdouble,
        x2: float | _to_floating,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.longdouble: ...
    @overload  # 0d +f80, 0d ~f80
    def __call__(
        self,
        x1: float | _to_floating,
        x2: np.longdouble,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.longdouble: ...
    @overload  # 0d ~c128, 0d +c128
    def __call__(
        self,
        x1: np.complex128 | np.complex64,
        x2: complex | _to_c128,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d +c128, 0d ~c128
    def __call__(
        self,
        x1: complex | _to_c128,
        x2: np.complex128 | np.complex64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d ~c160, 0d +c160
    def __call__(
        self,
        x1: np.clongdouble,
        x2: complex | _to_number,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.clongdouble: ...
    @overload  # 0d +c160, 0d ~c160
    def __call__(
        self,
        x1: complex | _to_number,
        x2: np.clongdouble,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.clongdouble: ...
    @overload  # 0d ~f80, 0d ~c
    def __call__(
        self,
        x1: np.longdouble,
        x2: np.complexfloating,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.clongdouble: ...
    @overload  # 0d ~c, 0d ~f80
    def __call__(
        self,
        x1: np.complexfloating,
        x2: np.longdouble,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.clongdouble: ...
    @overload  # 0d +complex, 0d +complex  (unavoidable overlap with first overload)
    def __call__(
        self,
        x1: complex,
        x2: complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128 | Any: ...
    @overload  # 0d _, 0d _, dtype=<known>
    def __call__[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self,
        x1: _NumberLike_co,
        x2: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d ?, 0d ?  (fallback)
    def __call__(
        self,
        x1: _NumberLike_co,
        x2: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: type[complex] | str | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> Any: ...
    @overload  # ?d +f64, ?d +f64
    def __call__(
        self,
        x1: _DualArrayLike[np.dtype[_to_f64], float],
        x2: _DualArrayLike[np.dtype[_to_f64], float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d ~f80, ?d +f80
    def __call__(
        self,
        x1: _ArrayLike[np.longdouble],
        x2: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.longdouble]: ...
    @overload  # ?d +f80, ?d ~f80
    def __call__(
        self,
        x1: _ArrayLikeFloat_co,
        x2: _ArrayLike[np.longdouble],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.longdouble]: ...
    @overload  # ?d ~c128, ?d +c128
    def __call__(
        self,
        x1: _ArrayLike[np.complex128 | np.complex64],
        x2: _DualArrayLike[np.dtype[_to_c128], complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d +c128, ?d ~c128
    def __call__(
        self,
        x1: _DualArrayLike[np.dtype[_to_c128], complex],
        x2: _ArrayLike[np.complex128 | np.complex64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d ~c160, ?d +c160
    def __call__(
        self,
        x1: _ArrayLike[np.clongdouble],
        x2: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.clongdouble]: ...
    @overload  # ?d +c160, ?d ~c160
    def __call__(
        self,
        x1: _ArrayLikeNumber_co,
        x2: _ArrayLike[np.clongdouble],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.clongdouble]: ...
    @overload  # ?d ~f80, ?d ~c
    def __call__(
        self,
        x1: _ArrayLike[np.longdouble],
        x2: _ArrayLike[np.complexfloating] | list[complex] | _NestedSequence[list[complex]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.clongdouble]: ...
    @overload  # ?d ~c, ?d ~f80
    def __call__(
        self,
        x1: _ArrayLike[np.complexfloating] | list[complex] | _NestedSequence[list[complex]],
        x2: _ArrayLike[np.longdouble],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.clongdouble]: ...
    @overload  # ?d ~complex, ?d +complex
    def __call__(
        self,
        x1: list[complex] | _NestedSequence[list[complex]],
        x2: complex | _NestedSequence[complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d +complex, ?d ~complex
    def __call__(
        self,
        x1: complex | _NestedSequence[complex],
        x2: list[complex] | _NestedSequence[list[complex]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d _, ?d _, dtype=<known>
    def __call__[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self,
        x1: _ArrayLikeNumber_co,
        x2: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x1: _ArrayLikeNumber_co,
        x2: _ArrayLikeNumber_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # ?d ?, ?d ?  (fallback)
    def __call__(
        self,
        x1: _ArrayLikeNumber_co,
        x2: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: type[complex] | str | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[Any]: ...
    @overload  # x1.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: _CanUfuncCall2L[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncCall2R[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    # keep in sync with `__call__`
    @override
    @overload  # 0d +f64, 0d +f64
    def outer(  # pyrefly:ignore[bad-override]
        self,
        x1: float | _to_f64,
        x2: float | _to_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~f80, 0d +f80
    def outer(
        self,
        x1: np.longdouble,
        x2: float | _to_floating,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.longdouble: ...
    @overload  # 0d +f80, 0d ~f80
    def outer(
        self,
        x1: float | _to_floating,
        x2: np.longdouble,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.longdouble: ...
    @overload  # 0d ~c128, 0d +c128
    def outer(
        self,
        x1: np.complex128 | np.complex64,
        x2: complex | _to_c128,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d +c128, 0d ~c128
    def outer(
        self,
        x1: complex | _to_c128,
        x2: np.complex128 | np.complex64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d ~c160, 0d +c160
    def outer(
        self,
        x1: np.clongdouble,
        x2: complex | _to_number,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.clongdouble: ...
    @overload  # 0d +c160, 0d ~c160
    def outer(
        self,
        x1: complex | _to_number,
        x2: np.clongdouble,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.clongdouble: ...
    @overload  # 0d ~f80, 0d ~c
    def outer(
        self,
        x1: np.longdouble,
        x2: np.complexfloating,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.clongdouble: ...
    @overload  # 0d ~c, 0d ~f80
    def outer(
        self,
        x1: np.complexfloating,
        x2: np.longdouble,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.clongdouble: ...
    @overload  # 0d +complex, 0d +complex  (unavoidable overlap with first overload)
    def outer(
        self,
        x1: complex,
        x2: complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128 | Any: ...
    @overload  # 0d _, 0d _, dtype=<known>
    def outer[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self,
        x1: _NumberLike_co,
        x2: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d ?, 0d ?  (fallback)
    def outer(
        self,
        x1: _NumberLike_co,
        x2: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: type[complex] | str | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> Any: ...
    @overload  # ?d +f64, ?d +f64
    def outer(
        self,
        x1: _DualArrayLike[np.dtype[_to_f64], float],
        x2: _DualArrayLike[np.dtype[_to_f64], float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d ~f80, ?d +f80
    def outer(
        self,
        x1: _ArrayLike[np.longdouble],
        x2: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.longdouble]: ...
    @overload  # ?d +f80, ?d ~f80
    def outer(
        self,
        x1: _ArrayLikeFloat_co,
        x2: _ArrayLike[np.longdouble],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.longdouble]: ...
    @overload  # ?d ~c128, ?d +c128
    def outer(
        self,
        x1: _ArrayLike[np.complex128 | np.complex64],
        x2: _DualArrayLike[np.dtype[_to_c128], complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d +c128, ?d ~c128
    def outer(
        self,
        x1: _DualArrayLike[np.dtype[_to_c128], complex],
        x2: _ArrayLike[np.complex128 | np.complex64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d ~c160, ?d +c160
    def outer(
        self,
        x1: _ArrayLike[np.clongdouble],
        x2: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.clongdouble]: ...
    @overload  # ?d +c160, ?d ~c160
    def outer(
        self,
        x1: _ArrayLikeNumber_co,
        x2: _ArrayLike[np.clongdouble],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.clongdouble]: ...
    @overload  # ?d ~f80, ?d ~c
    def outer(
        self,
        x1: _ArrayLike[np.longdouble],
        x2: _ArrayLike[np.complexfloating] | list[complex] | _NestedSequence[list[complex]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.clongdouble]: ...
    @overload  # ?d ~c, ?d ~f80
    def outer(
        self,
        x1: _ArrayLike[np.complexfloating] | list[complex] | _NestedSequence[list[complex]],
        x2: _ArrayLike[np.longdouble],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.clongdouble]: ...
    @overload  # ?d ~complex, ?d +complex
    def outer(
        self,
        x1: list[complex] | _NestedSequence[list[complex]],
        x2: complex | _NestedSequence[complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d +complex, ?d ~complex
    def outer(
        self,
        x1: complex | _NestedSequence[complex],
        x2: list[complex] | _NestedSequence[list[complex]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d _, ?d _, dtype=<known>
    def outer[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self,
        x1: _ArrayLikeNumber_co,
        x2: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # out=<given>
    def outer[OutT: np.ndarray](
        self,
        x1: _ArrayLikeNumber_co,
        x2: _ArrayLikeNumber_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # ?d ?, ?d ?  (fallback)
    def outer(
        self,
        x1: _ArrayLikeNumber_co,
        x2: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: type[complex] | str | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[Any]: ...
    @overload  # x1.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: _CanUfuncOuterL[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncOuterR[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[np.inexact], indices: _ArrayLikeInt, b: _ArrayLikeNumber_co, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[OtherT, IxT, OutT](self, a: _CanUfuncAt2L[OtherT, IxT, OutT], indices: IxT, b: OtherT, /) -> OutT: ...
    @overload
    def at[OtherT, IxT, OutT](self, a: OtherT, indices: IxT, b: _CanUfuncAt2R[OtherT, IxT, OutT], /) -> OutT: ...

    #
    @override
    @overload  # known scalar type
    def reduce[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[ScalarT] | Any: ...
    @overload  # known scalar type, axis=None
    def reduce[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> ScalarT: ...
    @overload  # known scalar type, keepdims=True
    def reduce[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # +f64
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_to_f64], float],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64] | Any: ...
    @overload  # +f64, axis=None
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_to_f64], float],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.float64: ...
    @overload  # +f64, keepdims=True
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_to_f64], float],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ~c128
    def reduce(
        self,
        array: _ArrayLike[np.complex64] | list[complex] | _NestedSequence[list[complex]],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.complex128] | Any: ...
    @overload  # ~c128, axis=None
    def reduce(
        self,
        array: _ArrayLike[np.complex64] | list[complex] | _NestedSequence[list[complex]],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.complex128: ...
    @overload  # ~c128, keepdims=True
    def reduce(
        self,
        array: _ArrayLike[np.complex64] | list[complex] | _NestedSequence[list[complex]],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # dtype=<known>
    def reduce[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self,
        array: _ArrayLikeNumber_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLike[ScalarT],
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[ScalarT] | Any: ...
    @overload  # dtype=<unknown>
    def reduce(
        self,
        array: _ArrayLikeNumber_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike,
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[Any] | Any: ...
    @overload  # out=<given>
    def reduce[OutT: np.ndarray](
        self,
        array: _ArrayLikeNumber_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
        keepdims: bool = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduce", array, ...)
    def reduce[OutT](
        self,
        array: _CanUfuncReduce[OutT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | EllipsisType | None = None,
        keepdims: bool = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...

    #
    @override
    @overload  # known scalar type
    def reduceat[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[ScalarT],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # ~f64
    def reduceat(
        self,
        array: _DualArrayLike[np.dtype[_to_f64], float],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ~c128
    def reduceat(
        self,
        array: _ArrayLike[np.complex64] | list[complex] | _NestedSequence[list[complex]],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # dtype=<known>
    def reduceat[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self,
        array: _ArrayLikeNumber_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLike[ScalarT],
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # dtype=<unknown>
    def reduceat(
        self,
        array: _ArrayLikeNumber_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: str,
        out: None = None,
    ) -> npt.NDArray[Any]: ...
    @overload  # out=<given>
    def reduceat[OutT: npt.NDArray[_to_floating]](
        self,
        array: _ArrayLikeNumber_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduceat", array, ...)
    def reduceat[IxT, OutT](
        self,
        array: _CanUfuncReduceAt[IxT, OutT],
        /,
        indices: IxT,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

    #
    @override
    @overload  # known scalar type
    def accumulate[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # ~f64
    def accumulate(
        self,
        array: _DualArrayLike[np.dtype[_to_f64], float],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ~c128
    def accumulate(
        self,
        array: _ArrayLike[np.complex64] | list[complex] | _NestedSequence[list[complex]],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # dtype=<known>
    def accumulate[ScalarT: np.float64 | np.complex128 | np.longdouble | np.clongdouble](
        self,
        array: _ArrayLikeNumber_co,
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLike[ScalarT],
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # dtype=<unknown>
    def accumulate(
        self,
        array: _ArrayLikeNumber_co,
        /,
        *,
        axis: int = 0,
        dtype: str,
        out: None = None,
    ) -> npt.NDArray[Any]: ...
    @overload  # out=<given>
    def accumulate[OutT: npt.NDArray[_to_floating]](
        self,
        array: _ArrayLikeNumber_co,
        /,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "accumulate", array, ...)
    def accumulate[OutT](
        self,
        array: _CanUfuncAccumulate[OutT],
        /,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

# efdg => efdg
#
# promotion table:
#
#       b8   i8   u8  i16  u16  i32  u32  i64  u64  f16  f32  f64  f80
# b8   f16  f16  f16  f32  f32  f64  f64  f64  f64  f16  f32  f64  f80
# i8   f16  f16  f16  f32  f32  f64  f64  f64  f64  f16  f32  f64  f80
# u8   f16  f16  f16  f32  f32  f64  f64  f64  f64  f16  f32  f64  f80
# i16  f32  f32  f32  f32  f32  f64  f64  f64  f64  f32  f32  f64  f80
# u16  f32  f32  f32  f32  f32  f64  f64  f64  f64  f32  f32  f64  f80
# i32  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f80
# u32  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f80
# i64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f80
# u64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f80
# f16  f16  f16  f16  f32  f32  f64  f64  f64  f64  f16  f32  f64  f80
# f32  f32  f32  f32  f32  f32  f64  f64  f64  f64  f32  f32  f64  f80
# f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f64  f80
# f80  f80  f80  f80  f80  f80  f80  f80  f80  f80  f80  f80  f80  f80
@type_check_only
class _ufunc_21_f[IdT](_ufunc_21[IdT]):  # type: ignore[misc]
    # TODO(@jorenham): shape-typing

    #
    @override
    @overload  # 0d +float, 0d +float
    def __call__(
        self,
        x1: float,
        x2: float,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~f64, 0d +f64
    def __call__(
        self,
        x1: np.float64 | _as_f64,
        x2: _to_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d +f64, 0d ~f64
    def __call__(
        self,
        x1: _to_f64,
        x2: np.float64 | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~f32, 0d +f32
    def __call__(
        self,
        x1: np.float32 | _as_f32,
        x2: _to_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float32: ...
    @overload  # 0d +f32, 0d ~f32
    def __call__(
        self,
        x1: _to_f32,
        x2: np.float32 | _as_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float32: ...
    @overload  # 0d ~f16, 0d ~f16
    def __call__(
        self,
        x1: np.float16 | _as_f16,
        x2: np.float16 | _as_f16,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float16: ...
    @overload  # 0d ~f80, 0d +f80
    def __call__(
        self,
        x1: np.longdouble,
        x2: _to_floating,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.longdouble: ...
    @overload  # 0d +f80, 0d ~f80
    def __call__(
        self,
        x1: _to_floating,
        x2: np.longdouble,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.longdouble: ...
    @overload  # 0d ?, 0d ?
    def __call__(
        self,
        x1: np.floating | np.integer,
        x2: np.floating | np.integer,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.floating: ...
    @overload  # 0d _, 0d _, dtype=<known>
    def __call__[FloatT: np.floating](
        self,
        x1: _FloatLike_co,
        x2: _FloatLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[FloatT],
        **kwargs: Unpack[_Kwargs21],
    ) -> FloatT: ...
    @overload  # 0d T@floating, 0d +float
    def __call__[FloatT: np.floating](
        self,
        x1: FloatT,
        x2: float,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> FloatT: ...
    @overload  # 0d +float, 0d T@floating
    def __call__[FloatT: np.floating](
        self,
        x1: float,
        x2: FloatT,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> FloatT: ...
    @overload  # ?d ~f64, ?d +f64
    def __call__(
        self,
        x1: _ArrayLike[np.float64 | _as_f64],
        x2: _ArrayLike[_to_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d +f64, ?d ~f64
    def __call__(
        self,
        x1: _ArrayLike[_to_f64],
        x2: _ArrayLike[np.float64 | _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d ~f32, ?d +f32
    def __call__(
        self,
        x1: _ArrayLike[np.float32 | _as_f32],
        x2: _ArrayLike[_to_f32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ?d +f32, ?d ~f32
    def __call__(
        self,
        x1: _ArrayLike[_to_f32],
        x2: _ArrayLike[np.float32 | _as_f32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ?d ~f16, ?d ~f16
    def __call__(
        self,
        x1: _ArrayLike[np.float16 | _as_f16],
        x2: _ArrayLike[np.float16 | _as_f16],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float16]: ...
    @overload  # ?d ~f80, ?d +f80
    def __call__(
        self,
        x1: _ArrayLike[np.longdouble],
        x2: _ArrayLike[_to_floating],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.longdouble]: ...
    @overload  # ?d +f80, ?d ~f80
    def __call__(
        self,
        x1: _ArrayLike[_to_floating],
        x2: _ArrayLike[np.longdouble],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.longdouble]: ...
    @overload  # Nd T@floating, ?d +float
    def __call__[ArrayT: npt.NDArray[np.floating]](
        self,
        x1: ArrayT,
        x2: float | _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # ?d +float, Nd T@floating
    def __call__[ArrayT: npt.NDArray[np.floating]](
        self,
        x1: float | _NestedSequence[float],
        x2: ArrayT,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # Nd +float, ?d +float
    def __call__(
        self,
        x1: _NestedSequence[float],
        x2: float | _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d +float, Nd +float
    def __call__(
        self,
        x1: float | _NestedSequence[float],
        x2: _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # Nd _, ?d _, dtype=<known>
    def __call__[FloatT: np.floating](
        self,
        x1: npt.NDArray[_to_floating],
        x2: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[FloatT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[FloatT]: ...
    @overload  # ?d _, Nd _, dtype=<known>
    def __call__[FloatT: np.floating](
        self,
        x1: _ArrayLikeFloat_co,
        x2: npt.NDArray[_to_floating],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[FloatT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[FloatT]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x1: _ArrayLikeFloat_co,
        x2: _ArrayLikeFloat_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # ?d ?, ?d ?  (fallback)
    def __call__(
        self,
        x1: _ArrayLikeFloat_co,
        x2: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeFloat | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.floating]: ...
    @overload  # x1.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: _CanUfuncCall2L[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncCall2R[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    #
    @override
    @overload  # 0d +float, 0d +float
    def outer(  # pyrefly:ignore[bad-override]
        self,
        x1: float,
        x2: float,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~f64, 0d +f64
    def outer(
        self,
        x1: np.float64 | _as_f64,
        x2: _to_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d +f64, 0d ~f64
    def outer(
        self,
        x1: _to_f64,
        x2: np.float64 | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~f32, 0d +f32
    def outer(
        self,
        x1: np.float32 | _as_f32,
        x2: _to_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float32: ...
    @overload  # 0d +f32, 0d ~f32
    def outer(
        self,
        x1: _to_f32,
        x2: np.float32 | _as_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float32: ...
    @overload  # 0d ~f16, 0d ~f16
    def outer(
        self,
        x1: np.float16 | _as_f16,
        x2: np.float16 | _as_f16,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float16: ...
    @overload  # 0d ~f80, 0d +f80
    def outer(
        self,
        x1: np.longdouble,
        x2: _to_floating,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.longdouble: ...
    @overload  # 0d +f80, 0d ~f80
    def outer(
        self,
        x1: _to_floating,
        x2: np.longdouble,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.longdouble: ...
    @overload  # 0d ?, 0d ?
    def outer(
        self,
        x1: np.floating | np.integer,
        x2: np.floating | np.integer,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.floating: ...
    @overload  # 0d _, 0d _, dtype=<known>
    def outer[FloatT: np.floating](
        self,
        x1: _FloatLike_co,
        x2: _FloatLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[FloatT],
        **kwargs: Unpack[_Kwargs21],
    ) -> FloatT: ...
    @overload  # 0d T@floating, 0d +float
    def outer[FloatT: np.floating](
        self,
        x1: FloatT,
        x2: float,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> FloatT: ...
    @overload  # 0d +float, 0d T@floating
    def outer[FloatT: np.floating](
        self,
        x1: float,
        x2: FloatT,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> FloatT: ...
    @overload  # ?d ~f64, ?d +f64
    def outer(
        self,
        x1: _ArrayLike[np.float64 | _as_f64],
        x2: _ArrayLike[_to_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d +f64, ?d ~f64
    def outer(
        self,
        x1: _ArrayLike[_to_f64],
        x2: _ArrayLike[np.float64 | _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d ~f32, ?d +f32
    def outer(
        self,
        x1: _ArrayLike[np.float32 | _as_f32],
        x2: _ArrayLike[_to_f32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ?d +f32, ?d ~f32
    def outer(
        self,
        x1: _ArrayLike[_to_f32],
        x2: _ArrayLike[np.float32 | _as_f32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ?d ~f16, ?d ~f16
    def outer(
        self,
        x1: _ArrayLike[np.float16 | _as_f16],
        x2: _ArrayLike[np.float16 | _as_f16],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float16]: ...
    @overload  # ?d ~f80, ?d +f80
    def outer(
        self,
        x1: _ArrayLike[np.longdouble],
        x2: _ArrayLike[_to_floating],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.longdouble]: ...
    @overload  # ?d +f80, ?d ~f80
    def outer(
        self,
        x1: _ArrayLike[_to_floating],
        x2: _ArrayLike[np.longdouble],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.longdouble]: ...
    @overload  # Nd T@floating, ?d +float
    def outer[ArrayT: npt.NDArray[np.floating]](
        self,
        x1: ArrayT,
        x2: float | _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # ?d +float, Nd T@floating
    def outer[ArrayT: npt.NDArray[np.floating]](
        self,
        x1: float | _NestedSequence[float],
        x2: ArrayT,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # Nd +float, ?d +float
    def outer(
        self,
        x1: _NestedSequence[float],
        x2: float | _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d +float, Nd +float
    def outer(
        self,
        x1: float | _NestedSequence[float],
        x2: _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # Nd _, ?d _, dtype=<known>
    def outer[FloatT: np.floating](
        self,
        x1: npt.NDArray[_to_floating],
        x2: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[FloatT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[FloatT]: ...
    @overload  # ?d _, Nd _, dtype=<known>
    def outer[FloatT: np.floating](
        self,
        x1: _ArrayLikeFloat_co,
        x2: npt.NDArray[_to_floating],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[FloatT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[FloatT]: ...
    @overload  # out=<given>
    def outer[OutT: np.ndarray](
        self,
        x1: _ArrayLikeFloat_co,
        x2: _ArrayLikeFloat_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # ?d ?, ?d ?  (fallback)
    def outer(
        self,
        x1: _ArrayLikeFloat_co,
        x2: _ArrayLikeFloat_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLikeFloat | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.floating]: ...
    @overload  # x1.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: _CanUfuncCall2L[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncCall2R[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[_to_floating], indices: _ArrayLikeInt, b: _ArrayLikeFloat_co, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[OtherT, IxT, OutT](self, a: _CanUfuncAt2L[OtherT, IxT, OutT], indices: IxT, b: OtherT, /) -> OutT: ...
    @overload
    def at[OtherT, IxT, OutT](self, a: OtherT, indices: IxT, b: _CanUfuncAt2R[OtherT, IxT, OutT], /) -> OutT: ...

    #
    @override
    @overload  # T@floating
    def reduce[FloatT: np.floating](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[FloatT],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: float | FloatT = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[FloatT] | FloatT: ...
    @overload  # T@floating, axis=None
    def reduce[FloatT: np.floating](
        self,
        array: _ArrayLike[FloatT],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: float | FloatT = ...,
        where: _ArrayLikeBool_co = True,
    ) -> FloatT: ...
    @overload  # T@floating, keepdims=True
    def reduce[FloatT: np.floating](
        self,
        array: _ArrayLike[FloatT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: float | FloatT = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[FloatT]: ...
    @overload  # ~f64
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_as_f64], float],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: float | _to_f64 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64] | np.float64: ...
    @overload  # ~f64, axis=None
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_as_f64], float],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: float | _to_f64 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.float64: ...
    @overload  # ~f64, keepdims=True
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_as_f64], float],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: float | _to_f64 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ~f32
    def reduce(
        self,
        array: _ArrayLike[_as_f32],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: float | _to_f32 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float32] | np.float32: ...
    @overload  # ~f32, axis=None
    def reduce(
        self,
        array: _ArrayLike[_as_f32],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: float | _to_f32 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.float32: ...
    @overload  # ~f32, keepdims=True
    def reduce(
        self,
        array: _ArrayLike[_as_f32],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: float | _to_f32 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ~f16
    def reduce(
        self,
        array: _ArrayLike[_as_f16],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: float | _as_f16 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float16] | np.float16: ...
    @overload  # ~f16, axis=None
    def reduce(
        self,
        array: _ArrayLike[_as_f16],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: float | _as_f16 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.float16: ...
    @overload  # ~f16, keepdims=True
    def reduce(
        self,
        array: _ArrayLike[_as_f16],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: float | _as_f16 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float16]: ...
    @overload  # ~f80
    def reduce(
        self,
        array: _ArrayLike[np.longdouble],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.longdouble] | np.longdouble: ...
    @overload  # ~f80, axis=None
    def reduce(
        self,
        array: _ArrayLike[np.longdouble],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.longdouble: ...
    @overload  # ~f80, keepdims=True
    def reduce(
        self,
        array: _ArrayLike[np.longdouble],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.longdouble]: ...
    @overload  # dtype=<known>
    def reduce[ScalarT: np.floating](
        self,
        array: _ArrayLikeFloat_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLike[ScalarT],
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[ScalarT] | ScalarT: ...
    @overload  # dtype=float
    def reduce[ScalarT: np.floating](
        self,
        array: _ArrayLikeFloat_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: type[float],
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64] | np.float64: ...
    @overload  # dtype=<unknown>
    def reduce[ScalarT: np.floating](
        self,
        array: _ArrayLikeFloat_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike,
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.floating] | np.floating: ...
    @overload  # out=<given>
    def reduce[OutT: np.ndarray](
        self,
        array: _ArrayLikeFloat_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
        keepdims: bool = False,
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduce", array, ...)
    def reduce[OutT](
        self,
        array: _CanUfuncReduce[OutT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | EllipsisType | None = None,
        keepdims: bool = False,
        initial: _FloatLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...

    #
    @override
    @overload  # T@floating
    def reduceat[FloatT: np.floating](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[FloatT],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[FloatT]: ...
    @overload  # ~f64
    def reduceat(
        self,
        array: _DualArrayLike[np.dtype[_as_f64], float],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ~f32
    def reduceat(
        self,
        array: _ArrayLike[_as_f32],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ~f16
    def reduceat(
        self,
        array: _ArrayLike[_as_f16],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.float16]: ...
    @overload  # dtype=<known>
    def reduceat[ScalarT: np.floating](
        self,
        array: _ArrayLikeFloat_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLike[ScalarT],
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # dtype=float
    def reduceat(
        self,
        array: _ArrayLikeFloat_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: type[float],
        out: None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # dtype=<unknown>
    def reduceat(
        self,
        array: _ArrayLikeFloat_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: str,
        out: None = None,
    ) -> npt.NDArray[np.floating]: ...
    @overload  # out=<given>
    def reduceat[OutT: npt.NDArray[_to_floating]](
        self,
        array: _ArrayLikeFloat_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduceat", array, ...)
    def reduceat[IxT, OutT](
        self,
        array: _CanUfuncReduceAt[IxT, OutT],
        /,
        indices: IxT,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

    #
    @override
    @overload  # T@floating
    def accumulate[FloatT: np.floating](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[FloatT],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[FloatT]: ...
    @overload  # ~f64
    def accumulate(
        self,
        array: _DualArrayLike[np.dtype[_as_f64], float],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ~f32
    def accumulate(
        self,
        array: _ArrayLike[_as_f32],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ~f16
    def accumulate(
        self,
        array: _ArrayLike[_as_f16],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.float16]: ...
    @overload  # dtype=<known>
    def accumulate[ScalarT: np.floating](
        self,
        array: _ArrayLikeFloat_co,
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLike[ScalarT],
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # dtype=float
    def accumulate(
        self,
        array: _ArrayLikeFloat_co,
        /,
        *,
        axis: int = 0,
        dtype: type[float],
        out: None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # dtype=<unknown>
    def accumulate(
        self,
        array: _ArrayLikeFloat_co,
        /,
        *,
        axis: int = 0,
        dtype: str,
        out: None = None,
    ) -> npt.NDArray[np.floating]: ...
    @overload  # out=<given>
    def accumulate[OutT: npt.NDArray[_to_floating]](
        self,
        array: _ArrayLikeFloat_co,
        /,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "accumulate", array, ...)
    def accumulate[IxT, OutT](
        self,
        array: _CanUfuncReduceAt[IxT, OutT],
        /,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...
    @overload
    def accumulate[OutT](
        self,
        array: _CanUfuncAccumulate[OutT],
        /,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

# efdgFDGmO * efdgFDGqmO => efdgFDGmO
#
# In order to avoid the number of overloads from getting out of hand, the float16 and
# [c]longdouble promotion overloads are omitted for `__call__` and `__outer__`. Those
# will are instead be handled by the fallback overloads.
@type_check_only
class _ufunc_21_divide(_ufunc_21[None]):  # type: ignore[misc]
    @override
    @overload  # 0d +float | +integer, 0d +float | +integer
    def __call__(
        self,
        x1: float | _to_integer,
        x2: float | _to_integer,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~f64, 0d +f64
    def __call__(
        self,
        x1: np.float64 | _as_f64,
        x2: float | _to_floating,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d +f64, 0d ~f64
    def __call__(
        self,
        x1: float | _to_floating,
        x2: np.float64 | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~f32, 0d +f32
    def __call__(
        self,
        x1: np.float32,
        x2: _to_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float32: ...
    @overload  # 0d +f32, 0d ~f32
    def __call__(
        self,
        x1: _to_f32,
        x2: np.float32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float32: ...
    @overload  # 0d ~c128, 0d +c128
    def __call__(
        self,
        x1: np.complex128,
        x2: _to_c128 | complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d +c128, 0d ~c128
    def __call__(
        self,
        x1: _to_c128 | complex,
        x2: np.complex128,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d c64, 0d ~f64
    def __call__(
        self,
        x1: np.complex64,
        x2: np.float64 | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d ~f64, 0d c64
    def __call__(
        self,
        x1: np.float64 | _as_f64,
        x2: np.complex64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d c64, 0d +c64
    def __call__(
        self,
        x1: np.complex64,
        x2: np.complex64 | _to_f32 | complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex64: ...
    @overload  # 0d +c64, 0d c64
    def __call__(
        self,
        x1: _to_f32 | complex,
        x2: np.complex64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex64: ...
    @overload  # 0d ~m, 0d ~m
    def __call__(
        self,
        x1: np.timedelta64,
        x2: np.timedelta64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~m, 0d floating | integer  (asymmetric)
    def __call__[MT: np.timedelta64](
        self,
        x1: MT,
        x2: np.floating | np.integer,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> MT: ...
    @overload  # 0d T@(inexact | m), 0d +float
    def __call__[ScalarT: np.inexact | np.timedelta64](
        self,
        x1: ScalarT,
        x2: float,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d +float, 0d T@floating
    def __call__[ScalarT: np.inexact | np.timedelta64](
        self,
        x1: float,
        x2: ScalarT,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d _, 0d _, dtype=<known>
    def __call__[ScalarT: np.inexact | np.timedelta64](
        self,
        x1: _NumberLike_co,
        x2: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d ?, 0d ?, dtype=<unknown>
    def __call__(
        self,
        x1: _NumberLike_co,
        x2: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> Any: ...
    @overload  # ?d +float | +integer, ?d +float | +integer
    def __call__(
        self,
        x1: _DualArrayLike[np.dtype[_to_integer], float],
        x2: _DualArrayLike[np.dtype[_to_integer], float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d ~f64, ?d +f64
    def __call__(
        self,
        x1: _ArrayLike[np.float64 | _as_f64],
        x2: _DualArrayLike[np.dtype[_to_floating], float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d +f64, ?d ~f64
    def __call__(
        self,
        x1: _DualArrayLike[np.dtype[_to_floating], float],
        x2: _ArrayLike[np.float64 | _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d ~f32, ?d +f32
    def __call__(
        self,
        x1: _ArrayLike[np.float32],
        x2: _DualArrayLike[np.dtype[_to_f32], float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ?d +f32, ?d ~f32
    def __call__(
        self,
        x1: _DualArrayLike[np.dtype[_to_f32], float],
        x2: _ArrayLike[np.float32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ?d ~c128, ?d +c128
    def __call__(
        self,
        x1: _ArrayLike[np.complex128],
        x2: _DualArrayLike[np.dtype[_to_c128], complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d +c128, ?d ~c128
    def __call__(
        self,
        x1: _DualArrayLike[np.dtype[_to_c128], complex],
        x2: _ArrayLike[np.complex128],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d c64, ?d ~f64
    def __call__(
        self,
        x1: _ArrayLike[np.complex64],
        x2: _ArrayLike[np.float64 | _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d ~f64, ?d c64
    def __call__(
        self,
        x1: _ArrayLike[np.float64 | _as_f64],
        x2: _ArrayLike[np.complex64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d c64, ?d +c64
    def __call__(
        self,
        x1: _ArrayLike[np.complex64],
        x2: _DualArrayLike[np.dtype[np.complex64 | _to_f32], complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex64]: ...
    @overload  # ?d +c64, ?d c64
    def __call__(
        self,
        x1: _DualArrayLike[np.dtype[_to_f32], complex],
        x2: _ArrayLike[np.complex64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex64]: ...
    @overload  # ?d ~m, ?d ~m
    def __call__(
        self,
        x1: _ArrayLike[np.timedelta64],
        x2: _ArrayLike[np.timedelta64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d ~m, ?d floating | integer  (asymmetric)
    def __call__[MT: np.timedelta64](
        self,
        x1: _ArrayLike[MT],
        x2: _DualArrayLike[np.dtype[np.floating | np.integer], float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[MT]: ...
    @overload  # Nd ~O, ?d ?
    def __call__(
        self,
        x1: npt.NDArray[np.object_],
        x2: _ArrayLikeNumericObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # ?d ?, Nd ~O
    def __call__(
        self,
        x1: _ArrayLikeNumericObj_co,
        x2: npt.NDArray[np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # Nd T@floating, ?d +float
    def __call__[ArrayT: npt.NDArray[np.inexact | np.timedelta64]](
        self,
        x1: ArrayT,
        x2: float | _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # ?d +float, Nd T@floating
    def __call__[ArrayT: npt.NDArray[np.inexact | np.timedelta64]](
        self,
        x1: float | _NestedSequence[float],
        x2: ArrayT,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # Nd _, ?d _, dtype=<known>
    def __call__[ScalarT: np.inexact | np.timedelta64](
        self,
        x1: npt.NDArray[_to_numeric],
        x2: _ArrayLikeNumericObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # ?d _, Nd _, dtype=<known>
    def __call__[ScalarT: np.inexact | np.timedelta64](
        self,
        x1: _ArrayLikeNumericObj_co,
        x2: npt.NDArray[_to_numeric],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x1: _ArrayLikeNumericObj_co,
        x2: _ArrayLikeNumericObj_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # ?d ?, ?d ?  (fallback)
    def __call__(
        self,
        x1: _ArrayLikeNumericObj_co,
        x2: _ArrayLikeNumericObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[Any]: ...
    @overload  # x1.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: _CanUfuncCall2L[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncCall2R[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    # keep in sync with __call__
    @override
    @overload  # 0d +float | +integer, 0d +float | +integer
    def outer(  # pyrefly:ignore[bad-override]
        self,
        x1: float | _to_integer,
        x2: float | _to_integer,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~f64, 0d +f64
    def outer(
        self,
        x1: np.float64 | _as_f64,
        x2: float | _to_floating,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d +f64, 0d ~f64
    def outer(
        self,
        x1: float | _to_floating,
        x2: np.float64 | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~f32, 0d +f32
    def outer(
        self,
        x1: np.float32,
        x2: _to_f32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float32: ...
    @overload  # 0d +f32, 0d ~f32
    def outer(
        self,
        x1: _to_f32,
        x2: np.float32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float32: ...
    @overload  # 0d ~c128, 0d +c128
    def outer(
        self,
        x1: np.complex128,
        x2: _to_c128 | complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d +c128, 0d ~c128
    def outer(
        self,
        x1: _to_c128 | complex,
        x2: np.complex128,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d c64, 0d ~f64
    def outer(
        self,
        x1: np.complex64,
        x2: np.float64 | _as_f64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d ~f64, 0d c64
    def outer(
        self,
        x1: np.float64 | _as_f64,
        x2: np.complex64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex128: ...
    @overload  # 0d c64, 0d +c64
    def outer(
        self,
        x1: np.complex64,
        x2: np.complex64 | _to_f32 | complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex64: ...
    @overload  # 0d +c64, 0d c64
    def outer(
        self,
        x1: _to_f32 | complex,
        x2: np.complex64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.complex64: ...
    @overload  # 0d ~m, 0d ~m
    def outer(
        self,
        x1: np.timedelta64,
        x2: np.timedelta64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.float64: ...
    @overload  # 0d ~m, 0d floating | integer  (asymmetric)
    def outer[MT: np.timedelta64](
        self,
        x1: MT,
        x2: np.floating | np.integer,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> MT: ...
    @overload  # 0d T@(inexact | m), 0d +float
    def outer[ScalarT: np.inexact | np.timedelta64](
        self,
        x1: ScalarT,
        x2: float,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d +float, 0d T@floating
    def outer[ScalarT: np.inexact | np.timedelta64](
        self,
        x1: float,
        x2: ScalarT,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d _, 0d _, dtype=<known>
    def outer[ScalarT: np.inexact | np.timedelta64](
        self,
        x1: _NumberLike_co,
        x2: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d ?, 0d ?, dtype=<unknown>
    def outer(
        self,
        x1: _NumberLike_co,
        x2: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> Any: ...
    @overload  # ?d +float | +integer, ?d +float | +integer
    def outer(
        self,
        x1: _DualArrayLike[np.dtype[_to_integer], float],
        x2: _DualArrayLike[np.dtype[_to_integer], float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d ~f64, ?d +f64
    def outer(
        self,
        x1: _ArrayLike[np.float64 | _as_f64],
        x2: _DualArrayLike[np.dtype[_to_floating], float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d +f64, ?d ~f64
    def outer(
        self,
        x1: _DualArrayLike[np.dtype[_to_floating], float],
        x2: _ArrayLike[np.float64 | _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d ~f32, ?d +f32
    def outer(
        self,
        x1: _ArrayLike[np.float32],
        x2: _DualArrayLike[np.dtype[_to_f32], float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ?d +f32, ?d ~f32
    def outer(
        self,
        x1: _DualArrayLike[np.dtype[_to_f32], float],
        x2: _ArrayLike[np.float32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float32]: ...
    @overload  # ?d ~c128, ?d +c128
    def outer(
        self,
        x1: _ArrayLike[np.complex128],
        x2: _DualArrayLike[np.dtype[_to_c128], complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d +c128, ?d ~c128
    def outer(
        self,
        x1: _DualArrayLike[np.dtype[_to_c128], complex],
        x2: _ArrayLike[np.complex128],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d c64, ?d ~f64
    def outer(
        self,
        x1: _ArrayLike[np.complex64],
        x2: _ArrayLike[np.float64 | _as_f64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d ~f64, ?d c64
    def outer(
        self,
        x1: _ArrayLike[np.float64 | _as_f64],
        x2: _ArrayLike[np.complex64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # ?d c64, ?d +c64
    def outer(
        self,
        x1: _ArrayLike[np.complex64],
        x2: _DualArrayLike[np.dtype[np.complex64 | _to_f32], complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex64]: ...
    @overload  # ?d +c64, ?d c64
    def outer(
        self,
        x1: _DualArrayLike[np.dtype[_to_f32], complex],
        x2: _ArrayLike[np.complex64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.complex64]: ...
    @overload  # ?d ~m, ?d ~m
    def outer(
        self,
        x1: _ArrayLike[np.timedelta64],
        x2: _ArrayLike[np.timedelta64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ?d ~m, ?d floating | integer  (asymmetric)
    def outer[MT: np.timedelta64](
        self,
        x1: _ArrayLike[MT],
        x2: _DualArrayLike[np.dtype[np.floating | np.integer], float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[MT]: ...
    @overload  # Nd ~O, ?d ?
    def outer(
        self,
        x1: npt.NDArray[np.object_],
        x2: _ArrayLikeNumericObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # ?d ?, Nd ~O
    def outer(
        self,
        x1: _ArrayLikeNumericObj_co,
        x2: npt.NDArray[np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # Nd T@floating, ?d +float
    def outer[ArrayT: npt.NDArray[np.inexact | np.timedelta64]](
        self,
        x1: ArrayT,
        x2: float | _NestedSequence[float],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # ?d +float, Nd T@floating
    def outer[ArrayT: npt.NDArray[np.inexact | np.timedelta64]](
        self,
        x1: float | _NestedSequence[float],
        x2: ArrayT,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # Nd _, ?d _, dtype=<known>
    def outer[ScalarT: np.inexact | np.timedelta64](
        self,
        x1: npt.NDArray[_to_numeric],
        x2: _ArrayLikeNumericObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # ?d _, Nd _, dtype=<known>
    def outer[ScalarT: np.inexact | np.timedelta64](
        self,
        x1: _ArrayLikeNumericObj_co,
        x2: npt.NDArray[_to_numeric],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # out=<given>
    def outer[OutT: np.ndarray](
        self,
        x1: _ArrayLikeNumericObj_co,
        x2: _ArrayLikeNumericObj_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # ?d ?, ?d ?  (fallback)
    def outer(
        self,
        x1: _ArrayLikeNumericObj_co,
        x2: _ArrayLikeNumericObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[Any]: ...
    @overload  # x1.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: _CanUfuncOuterL[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncOuterR[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[_to_numeric | np.object_], indices: _ArrayLikeInt, b: _ArrayLikeNumericObj_co, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[OtherT, IxT, OutT](self, a: _CanUfuncAt2L[OtherT, IxT, OutT], indices: IxT, b: OtherT, /) -> OutT: ...
    @overload
    def at[OtherT, IxT, OutT](self, a: OtherT, indices: IxT, b: _CanUfuncAt2R[OtherT, IxT, OutT], /) -> OutT: ...

    #
    @override
    @overload  # T@inexact
    def reduce[ScalarT: np.inexact](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: complex | ScalarT = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[ScalarT] | ScalarT: ...
    @overload  # T@inexact, axis=None
    def reduce[ScalarT: np.inexact](
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: complex | ScalarT = ...,
        where: _ArrayLikeBool_co = True,
    ) -> ScalarT: ...
    @overload  # T@floating, keepdims=True
    def reduce[ScalarT: np.inexact | np.object_](
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: complex | ScalarT = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # object_
    def reduce(
        self,
        array: npt.NDArray[np.object_],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: object = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.object_] | Any: ...
    @overload  # object_, axis=None
    def reduce(
        self,
        array: npt.NDArray[np.object_],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: object = ...,
        where: _ArrayLikeBool_co = True,
    ) -> Any: ...
    @overload  # +f64
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_to_integer], float],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: float | _to_f64 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64] | np.float64: ...
    @overload  # +f64, axis=None
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_to_integer], float],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: float | _to_f64 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.float64: ...
    @overload  # +f64, keepdims=True
    def reduce(
        self,
        array: _DualArrayLike[np.dtype[_to_integer], float],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: float | _to_f64 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ~c128  (invariant `list` is used to avoid overlap with `float`)
    def reduce(
        self,
        array: _NestedSequence[list[complex]] | list[complex],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: complex | _to_c128 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.complex128] | np.complex128: ...
    @overload  # ~c128, axis=None
    def reduce(
        self,
        array: _NestedSequence[list[complex]] | list[complex],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: complex | _to_c128 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.complex128: ...
    @overload  # ~c128, keepdims=True
    def reduce(
        self,
        array: _NestedSequence[list[complex]] | list[complex],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: complex | _to_c128 = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # dtype=<known>
    def reduce[ScalarT: np.inexact | np.object_](
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLike[ScalarT],
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[ScalarT] | ScalarT: ...
    @overload  # dtype=float
    def reduce(
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: type[float],
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.float64] | np.float64: ...
    @overload  # dtype=<unknown>
    def reduce(
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: npt.DTypeLike,
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[Any] | Any: ...
    @overload  # dtype=<unknown>, axis=None
    def reduce(
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        *,
        axis: None,
        dtype: npt.DTypeLike,
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> Any: ...
    @overload  # out=<given>
    def reduce[OutT: np.ndarray](
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
        keepdims: bool = False,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduce", array, ...)
    def reduce[OutT](
        self,
        array: _CanUfuncReduce[OutT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | EllipsisType | None = None,
        keepdims: bool = False,
        initial: object = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...

    #
    @override
    @overload  # T@(inexact | object_)
    def reduceat[ScalarT: np.inexact | np.object_](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[ScalarT],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # +f64
    def reduceat(
        self,
        array: _DualArrayLike[np.dtype[_to_f64], float],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ~complex
    def reduceat(
        self,
        array: _NestedSequence[list[complex]] | list[complex],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # dtype=<known>
    def reduceat[ScalarT: np.inexact | np.object_](
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLike[ScalarT],
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # dtype=float
    def reduceat(
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: type[float],
        out: None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # dtype=<unknown>
    def reduceat(
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: str,
        out: None = None,
    ) -> npt.NDArray[Any]: ...
    @overload  # out=<given>
    def reduceat[OutT: npt.NDArray[_to_number | np.object_]](
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduceat", array, ...)
    def reduceat[IxT, OutT](
        self,
        array: _CanUfuncReduceAt[IxT, OutT],
        /,
        indices: IxT,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

    #
    @override
    @overload  # T@(inexact | object_)
    def accumulate[ScalarT: np.inexact | np.object_](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # +f64
    def accumulate(
        self,
        array: _DualArrayLike[np.dtype[_to_f64], float],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # ~complex
    def accumulate(
        self,
        array: _NestedSequence[list[complex]] | list[complex],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.complex128]: ...
    @overload  # dtype=<known>
    def accumulate[ScalarT: np.inexact | np.object_](
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLike[ScalarT],
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # dtype=float
    def accumulate(
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        *,
        axis: int = 0,
        dtype: type[float],
        out: None = None,
    ) -> npt.NDArray[np.float64]: ...
    @overload  # dtype=<unknown>
    def accumulate(
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        *,
        axis: int = 0,
        dtype: str,
        out: None = None,
    ) -> npt.NDArray[Any]: ...
    @overload  # out=<given>
    def accumulate[OutT: npt.NDArray[_to_number | np.object_]](
        self,
        array: _ArrayLikeNumberObj_co,
        /,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "accumulate", array, ...)
    def accumulate[OutT](
        self,
        array: _CanUfuncAccumulate[OutT],
        /,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

# [?]bBhHiIlLqQO, [?]bBhHiIlLqQO => [?]bBhHiIlLqQO
#
# promotion table:
#       b8   i8   u8  i16  u16  i32  u32  i64  u64  O
# b8   [b8]  i8   u8  i16  u16  i32  u32  i64  u64  O
# i8    i8   i8  i16  i16  i32  i32  i64  i64       O
# u8    u8  i16   u8  i16  u16  i32  u32  i64  u64  O
# i16  i16  i16  i16  i16  i32  i32  i64  i64       O
# u16  u16  i32  u16  i32  u16  i32  u32  i64  u64  O
# i32  i32  i32  i32  i32  i32  i32  i64  i64       O
# u32  u32  i64  u32  i64  u32  i64  u32  i64  u64  O
# i64  i64  i64  i64  i64  i64  i64  i64  i64       O
# u64  u64       u64       u64       u64       u64  O
# O      O    O    O    O    O    O    O    O    O  O
#
# Only the i64/i32/u8 int promotion rules are implemented (most gh code search hits).
@type_check_only
class _ufunc_21_bio(_ufunc_21[_IdT_co], Generic[_IdT_co, _ScalarT_contra]):  # type: ignore[misc]
    @override
    @overload  # 0d bool, 0d bool  (only if `bool` in domain)
    def __call__(
        self: _ufunc_21_bio[Any, np.bool],
        x1: bool | np.bool,
        x2: bool | np.bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.bool: ...
    @overload  # 0d int|bool, 0d int|bool  (inevitably overlaps with previous overload)
    def __call__(
        self,
        x1: int | np.bool,
        x2: int | np.bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.int_ | Any: ...
    @overload  # 0d ~i64, 0d +i64
    def __call__(
        self,
        x1: np.int64,
        x2: _to_i64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.int64: ...
    @overload  # 0d +i64, 0d ~i64
    def __call__(
        self,
        x1: _to_i64,
        x2: np.int64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.int64: ...
    @overload  # 0d ~i32, 0d +i32
    def __call__(
        self,
        x1: np.int32,
        x2: _to_i32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.int32: ...
    @overload  # 0d +i32, 0d ~i32
    def __call__(
        self,
        x1: _to_i32,
        x2: np.int32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.int32: ...
    @overload  # 0d ~u8, 0d +u8
    def __call__(
        self,
        x1: np.uint8,
        x2: _to_u8,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.uint8: ...
    @overload  # 0d +u8, 0d ~u8
    def __call__(
        self,
        x1: _to_u8,
        x2: np.uint8,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.uint8: ...
    @overload  # 0d unsigned, 0d unsigned
    def __call__(
        self,
        x1: np.unsignedinteger,
        x2: np.unsignedinteger,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.unsignedinteger: ...
    @overload  # 0d ?, 0d ?
    def __call__(
        self,
        x1: np.integer,
        x2: np.integer,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.integer: ...
    @overload  # 0d _, 0d _, dtype=<known>
    def __call__[ScalarT: np.integer | np.bool](
        self,
        x1: _IntLike_co,
        x2: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d T@integer, 0d +int
    def __call__[IntT: np.integer](
        self,
        x1: IntT,
        x2: int | np.bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> IntT: ...
    @overload  # 0d +int, 0d T@integer
    def __call__[IntT: np.integer](
        self,
        x1: int | np.bool,
        x2: IntT,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> IntT: ...
    @overload  # ?d +bool, ?d +bool  (only if `bool` in domain)
    def __call__(
        self: _ufunc_21_bio[Any, np.bool],
        x1: _ArrayLikeBool_co,
        x2: _ArrayLikeBool_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # ?d ~i64, ?d +i64
    def __call__(
        self,
        x1: _ArrayLike[np.int64],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int64]: ...
    @overload  # ?d +i64, ?d ~i64
    def __call__(
        self,
        x1: _ArrayLikeInt_co,
        x2: _ArrayLike[np.int64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int64]: ...
    @overload  # ?d ~i32, ?d +i32
    def __call__(
        self,
        x1: _ArrayLike[np.int32],
        x2: _ArrayLike[_to_i32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int32]: ...
    @overload  # ?d +i32, ?d ~i32
    def __call__(
        self,
        x1: _ArrayLike[_to_i32],
        x2: _ArrayLike[np.int32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int32]: ...
    @overload  # ?d ~u8, ?d +u8
    def __call__(
        self,
        x1: _ArrayLike[np.uint8],
        x2: _ArrayLike[_to_u8],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.uint8]: ...
    @overload  # ?d +u8, ?d ~u8
    def __call__(
        self,
        x1: _ArrayLike[_to_u8],
        x2: _ArrayLike[np.uint8],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.uint8]: ...
    @overload  # Nd ~obj, ?d +obj
    def __call__(
        self,
        x1: npt.NDArray[np.object_],
        x2: _ArrayLikeIntObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # ?d +obj, Nd ~obj
    def __call__(
        self,
        x1: _ArrayLikeIntObj_co,
        x2: npt.NDArray[np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # Nd T@integer, ?d +int
    def __call__[ArrayT: npt.NDArray[np.integer | np.object_]](
        self,
        x1: ArrayT,
        x2: _DualArrayLike[np.dtype[np.bool], int],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # ?d +int, Nd T@integer
    def __call__[ArrayT: npt.NDArray[np.integer | np.object_]](
        self,
        x1: _DualArrayLike[np.dtype[np.bool], int],
        x2: ArrayT,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # Nd ~int, ?d +int
    def __call__(
        self,
        x1: list[int] | _NestedSequence[list[int]],
        x2: int | _NestedSequence[int],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int_]: ...
    @overload  # ?d +int, Nd ~int
    def __call__(
        self,
        x1: int | _NestedSequence[int],
        x2: list[int] | _NestedSequence[list[int]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int_]: ...
    @overload  # Nd _, ?d _, dtype=<known>
    def __call__[ScalarT: _to_integer | np.object_](
        self,
        x1: npt.NDArray[_to_integer | np.object_],
        x2: _ArrayLikeIntObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # ?d _, Nd _, dtype=<known>
    def __call__[ScalarT: _to_integer | np.object_](
        self,
        x1: _ArrayLikeIntObj_co,
        x2: npt.NDArray[_to_integer | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x1: _ArrayLikeIntObj_co,
        x2: _ArrayLikeIntObj_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # ?d ?, ?d ?  (fallback)
    def __call__(
        self,
        x1: _ArrayLikeIntObj_co,
        x2: _ArrayLikeIntObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[Any]: ...
    @overload  # x1.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: _CanUfuncCall2L[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "__call__", x1, x2, ...)
    def __call__[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncCall2R[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    # keep in sync with `__call__`
    @override
    @overload  # 0d bool, 0d bool  (only if `bool` in domain)
    def outer(  # pyrefly:ignore[bad-override]
        self: _ufunc_21_bio[Any, np.bool],
        x1: bool | np.bool,
        x2: bool | np.bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.bool: ...
    @overload  # 0d int|bool, 0d int|bool  (inevitably overlaps with previous overload)
    def outer(
        self,
        x1: int | np.bool,
        x2: int | np.bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.int_ | Any: ...
    @overload  # 0d ~i64, 0d +i64
    def outer(
        self,
        x1: np.int64,
        x2: _to_i64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.int64: ...
    @overload  # 0d +i64, 0d ~i64
    def outer(
        self,
        x1: _to_i64,
        x2: np.int64,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.int64: ...
    @overload  # 0d ~i32, 0d +i32
    def outer(
        self,
        x1: np.int32,
        x2: _to_i32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.int32: ...
    @overload  # 0d +i32, 0d ~i32
    def outer(
        self,
        x1: _to_i32,
        x2: np.int32,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.int32: ...
    @overload  # 0d ~u8, 0d +u8
    def outer(
        self,
        x1: np.uint8,
        x2: _to_u8,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.uint8: ...
    @overload  # 0d +u8, 0d ~u8
    def outer(
        self,
        x1: _to_u8,
        x2: np.uint8,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.uint8: ...
    @overload  # 0d unsigned, 0d unsigned
    def outer(
        self,
        x1: np.unsignedinteger,
        x2: np.unsignedinteger,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.unsignedinteger: ...
    @overload  # 0d ?, 0d ?
    def outer(
        self,
        x1: np.integer,
        x2: np.integer,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> np.integer: ...
    @overload  # 0d _, 0d _, dtype=<known>
    def outer[ScalarT: np.integer | np.bool](
        self,
        x1: _IntLike_co,
        x2: _IntLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> ScalarT: ...
    @overload  # 0d T@integer, 0d +int
    def outer[IntT: np.integer](
        self,
        x1: IntT,
        x2: int | np.bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> IntT: ...
    @overload  # 0d +int, 0d T@integer
    def outer[IntT: np.integer](
        self,
        x1: int | np.bool,
        x2: IntT,
        /,
        *,
        out: None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> IntT: ...
    @overload  # ?d +bool, ?d +bool  (only if `bool` in domain)
    def outer(
        self: _ufunc_21_bio[Any, np.bool],
        x1: _ArrayLikeBool_co,
        x2: _ArrayLikeBool_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.bool]: ...
    @overload  # ?d ~i64, ?d +i64
    def outer(
        self,
        x1: _ArrayLike[np.int64],
        x2: _ArrayLikeInt_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int64]: ...
    @overload  # ?d +i64, ?d ~i64
    def outer(
        self,
        x1: _ArrayLikeInt_co,
        x2: _ArrayLike[np.int64],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int64]: ...
    @overload  # ?d ~i32, ?d +i32
    def outer(
        self,
        x1: _ArrayLike[np.int32],
        x2: _ArrayLike[_to_i32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int32]: ...
    @overload  # ?d +i32, ?d ~i32
    def outer(
        self,
        x1: _ArrayLike[_to_i32],
        x2: _ArrayLike[np.int32],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int32]: ...
    @overload  # ?d ~u8, ?d +u8
    def outer(
        self,
        x1: _ArrayLike[np.uint8],
        x2: _ArrayLike[_to_u8],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.uint8]: ...
    @overload  # ?d +u8, ?d ~u8
    def outer(
        self,
        x1: _ArrayLike[_to_u8],
        x2: _ArrayLike[np.uint8],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.uint8]: ...
    @overload  # Nd ~obj, ?d +obj
    def outer(
        self,
        x1: npt.NDArray[np.object_],
        x2: _ArrayLikeIntObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # ?d +obj, Nd ~obj
    def outer(
        self,
        x1: _ArrayLikeIntObj_co,
        x2: npt.NDArray[np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.object_]: ...
    @overload  # Nd T@integer, ?d +int
    def outer[ArrayT: npt.NDArray[np.integer | np.object_]](
        self,
        x1: ArrayT,
        x2: _DualArrayLike[np.dtype[np.bool], int],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # ?d +int, Nd T@integer
    def outer[ArrayT: npt.NDArray[np.integer | np.object_]](
        self,
        x1: _DualArrayLike[np.dtype[np.bool], int],
        x2: ArrayT,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> ArrayT: ...
    @overload  # Nd ~int, ?d +int
    def outer(
        self,
        x1: list[int] | _NestedSequence[list[int]],
        x2: int | _NestedSequence[int],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int_]: ...
    @overload  # ?d +int, Nd ~int
    def outer(
        self,
        x1: int | _NestedSequence[int],
        x2: list[int] | _NestedSequence[list[int]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[np.int_]: ...
    @overload  # Nd _, ?d _, dtype=<known>
    def outer[ScalarT: _to_integer | np.object_](
        self,
        x1: npt.NDArray[_to_integer | np.object_],
        x2: _ArrayLikeIntObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # ?d _, Nd _, dtype=<known>
    def outer[ScalarT: _to_integer | np.object_](
        self,
        x1: _ArrayLikeIntObj_co,
        x2: npt.NDArray[_to_integer | np.object_],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # out=<given>
    def outer[OutT: np.ndarray](
        self,
        x1: _ArrayLikeIntObj_co,
        x2: _ArrayLikeIntObj_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # ?d ?, ?d ?  (fallback)
    def outer(
        self,
        x1: _ArrayLikeIntObj_co,
        x2: _ArrayLikeIntObj_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> npt.NDArray[Any]: ...
    @overload  # x1.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: _CanUfuncOuterL[OtherT, OutT],
        x2: OtherT,
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...
    @overload  # x2.__array_ufunc__(self, "outer", x1, x2, ...)
    def outer[OtherT, OutT](
        self,
        x1: OtherT,
        x2: _CanUfuncOuterR[OtherT, OutT],
        /,
        *,
        out: object | None = None,
        dtype: npt.DTypeLike | None = None,
        **kwargs: Unpack[_Kwargs21],
    ) -> OutT: ...

    #
    @override
    @overload
    def at(  # pyrefly:ignore[bad-override]
        self: _ufunc_21_bio[Any, np.bool],
        a: npt.NDArray[np.bool],
        indices: _ArrayLikeInt,
        b: _ArrayLikeIntObj_co,
        /,
    ) -> None: ...
    @overload
    def at(self, a: npt.NDArray[np.integer | np.object_], indices: _ArrayLikeInt, b: _ArrayLikeIntObj_co, /) -> None: ...
    @overload
    def at[OtherT, IxT, OutT](self, a: _CanUfuncAt2L[OtherT, IxT, OutT], indices: IxT, b: OtherT, /) -> OutT: ...
    @overload
    def at[OtherT, IxT, OutT](self, a: OtherT, indices: IxT, b: _CanUfuncAt2R[OtherT, IxT, OutT], /) -> OutT: ...

    #
    @override
    @overload  # known scalar type
    def reduce[ScalarT: np.integer](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: int | ScalarT = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[ScalarT] | Any: ...
    @overload  # known scalar type, axis=None
    def reduce[ScalarT: np.integer](
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: int | ScalarT = ...,
        where: _ArrayLikeBool_co = True,
    ) -> ScalarT: ...
    @overload  # known scalar type, keepdims=True
    def reduce[ScalarT: np.integer | np.object_](
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: int | ScalarT = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # ~object_
    def reduce(
        self,
        array: npt.NDArray[np.object_],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: object = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.object_] | Any: ...
    @overload  # object_, axis=None
    def reduce(
        self,
        array: npt.NDArray[np.object_],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: object = ...,
        where: _ArrayLikeBool_co = True,
    ) -> Any: ...
    @overload  # ~bool
    def reduce(
        self: _ufunc_21_bio[Any, np.bool],
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.bool] | Any: ...
    @overload  # ~bool, axis=None
    def reduce(
        self: _ufunc_21_bio[Any, np.bool],
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.bool: ...
    @overload  # ~bool, keepdims=True
    def reduce(
        self: _ufunc_21_bio[Any, np.bool],
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: bool | np.bool = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # ~int
    def reduce(
        self,
        array: list[int] | _NestedSequence[list[int]],
        /,
        *,
        axis: int | tuple[int, ...] = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[False] = False,
        initial: _IntLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.int_] | Any: ...
    @overload  # ~int, axis=None
    def reduce(
        self,
        array: list[int] | _NestedSequence[list[int]],
        /,
        *,
        axis: None,
        dtype: None = None,
        out: None = None,
        keepdims: Literal[False] = False,
        initial: _IntLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> np.int_: ...
    @overload  # ~int, keepdims=True
    def reduce(
        self,
        array: list[int] | _NestedSequence[list[int]],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: None = None,
        out: EllipsisType | None = None,
        keepdims: Literal[True],
        initial: _IntLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[np.int_]: ...
    @overload  # dtype=<known>
    def reduce[ScalarT: _to_integer | np.object_](
        self,
        array: _ArrayLikeIntObj_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: _DTypeLike[ScalarT],
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: _IntLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[ScalarT] | Any: ...
    @overload  # dtype=<unknown>
    def reduce(
        self,
        array: _ArrayLikeIntObj_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike,
        out: EllipsisType | None = None,
        keepdims: bool = False,
        initial: type | str = ...,
        where: _ArrayLikeBool_co = True,
    ) -> npt.NDArray[Any] | Any: ...
    @overload  # out=<given>
    def reduce[OutT: np.ndarray](
        self,
        array: _ArrayLikeIntObj_co,
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
        keepdims: bool = False,
        initial: _IntLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduce", array, ...)
    def reduce[OutT](
        self,
        array: _CanUfuncReduce[OutT],
        /,
        *,
        axis: int | tuple[int, ...] | None = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | EllipsisType | None = None,
        keepdims: bool = False,
        initial: _IntLike_co = ...,
        where: _ArrayLikeBool_co = True,
    ) -> OutT: ...

    #
    @override
    @overload  # known scalar type
    def reduceat[ScalarT: np.integer | np.object_](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[ScalarT],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # ~bool
    def reduceat(
        self: _ufunc_21_bio[Any, np.bool],
        array: _ArrayLikeBool_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # ~int
    def reduceat(
        self,
        array: list[int] | _NestedSequence[list[int]],
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.int_]: ...
    @overload  # dtype=<known>
    def reduceat[ScalarT: _to_integer | np.object_](
        self,
        array: _ArrayLikeIntObj_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: _DTypeLike[ScalarT],
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # dtype=<unknown>
    def reduceat(
        self,
        array: _ArrayLikeIntObj_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: str,
        out: None = None,
    ) -> npt.NDArray[Any]: ...
    @overload  # out=<given>
    def reduceat[OutT: npt.NDArray[_to_integer | np.object_]](
        self,
        array: _ArrayLikeIntObj_co,
        /,
        indices: tuple[int, ...],
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "reduceat", array, indices, ...)
    def reduceat[IxT, OutT](
        self,
        array: _CanUfuncReduceAt[IxT, OutT],
        /,
        indices: IxT,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

    #
    @override
    @overload  # known scalar type
    def accumulate[ScalarT: np.integer | np.object_](  # pyrefly:ignore[bad-override]
        self,
        array: _ArrayLike[ScalarT],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # ~bool
    def accumulate(
        self: _ufunc_21_bio[Any, np.bool],
        array: _ArrayLikeBool_co,
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.bool]: ...
    @overload  # ~int
    def accumulate(
        self,
        array: list[int] | _NestedSequence[list[int]],
        /,
        *,
        axis: int = 0,
        dtype: None = None,
        out: None = None,
    ) -> npt.NDArray[np.int_]: ...
    @overload  # dtype=<known>
    def accumulate[ScalarT: _to_integer | np.object_](
        self,
        array: _ArrayLikeIntObj_co,
        /,
        *,
        axis: int = 0,
        dtype: _DTypeLike[ScalarT],
        out: None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # dtype=<unknown>
    def accumulate(
        self,
        array: _ArrayLikeIntObj_co,
        /,
        *,
        axis: int = 0,
        dtype: str,
        out: None = None,
    ) -> npt.NDArray[Any]: ...
    @overload  # out=<given>
    def accumulate[OutT: npt.NDArray[_to_integer | np.object_]](
        self,
        array: _ArrayLikeIntObj_co,
        /,
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: OutT,
    ) -> OutT: ...
    @overload  # array.__array_ufunc__(self, "accumulate", array, ...)
    def accumulate[OutT](
        self,
        array: _CanUfuncAccumulate[OutT],
        *,
        axis: int = 0,
        dtype: npt.DTypeLike | None = None,
        out: np.ndarray | None = None,
    ) -> OutT: ...

logical_and: Final[_ufunc_21_logical[Literal[True]]] = ...
logical_or: Final[_ufunc_21_logical[Literal[False]]] = ...
logical_xor: Final[_ufunc_21_logical[Literal[False]]] = ...

equal: Final[_ufunc_21_cmp] = ...
greater: Final[_ufunc_21_cmp] = ...
greater_equal: Final[_ufunc_21_cmp] = ...
less: Final[_ufunc_21_cmp] = ...
less_equal: Final[_ufunc_21_cmp] = ...
not_equal: Final[_ufunc_21_cmp] = ...

ldexp: Final[_ufunc_21_ldexp] = ...

float_power: Final[_ufunc_21_float_power] = ...

copysign: Final[_ufunc_21_f[None]] = ...
heaviside: Final[_ufunc_21_f[None]] = ...
logaddexp: Final[_ufunc_21_f[float]] = ...
logaddexp2: Final[_ufunc_21_f[float]] = ...
nextafter: Final[_ufunc_21_f[None]] = ...

# technically these also accept `object_` dtypes, but that requires the underlying python
# object to have a `.arctan2`/`.hypot()` method, which is unlikely to exist on non-numpy
# types, so it would be type-unsafe to (explicitly) allow `object_` dtypes here.
arctan2: Final[_ufunc_21_f[None]] = ...
hypot: Final[_ufunc_21_f[Literal[0]]] = ...

divide: Final[_ufunc_21_divide] = ...

# technically these reject `bool * bool`, but that would lead to a lot of code duplication
gcd: Final[_ufunc_21_bio[Literal[0], np.integer | np.object_]] = ...
lcm: Final[_ufunc_21_bio[None, np.integer | np.object_]] = ...
left_shift: Final[_ufunc_21_bio[None, np.integer | np.object_]] = ...
right_shift: Final[_ufunc_21_bio[None, np.integer | np.object_]] = ...

bitwise_and: Final[_ufunc_21_bio[Literal[-1], np.bool | np.integer | np.object_]] = ...
bitwise_or: Final[_ufunc_21_bio[Literal[0], np.bool | np.integer | np.object_]] = ...
bitwise_xor: Final[_ufunc_21_bio[Literal[0], np.bool | np.integer | np.object_]] = ...

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
