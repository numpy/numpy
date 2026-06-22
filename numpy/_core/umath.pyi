import contextvars
from _typeshed import SupportsWrite
from collections.abc import Callable, Sequence
from types import EllipsisType
from typing import (
    Any,
    Final,
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
from typing_extensions import CapsuleType

import numpy as np
import numpy.typing as npt
from numpy import (
    _CastingKind,
    _OrderKACF,
    absolute,
    add,
    arctan2,
    bitwise_and,
    bitwise_count,
    bitwise_or,
    bitwise_xor,
    cbrt,
    ceil,
    conj,
    conjugate,
    copysign,
    deg2rad,
    degrees,
    divide,
    divmod,
    e,
    equal,
    euler_gamma,
    fabs,
    float_power,
    floor,
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
    invert,
    isfinite,
    isinf,
    isnan,
    isnat,
    lcm,
    ldexp,
    left_shift,
    less,
    less_equal,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    matmul,
    matvec,
    maximum,
    minimum,
    mod,
    modf,
    multiply,
    negative,
    nextafter,
    not_equal,
    pi,
    positive,
    power,
    rad2deg,
    radians,
    reciprocal,
    remainder,
    right_shift,
    sign,
    signbit,
    spacing,
    square,
    subtract,
    true_divide,
    trunc,
    vecdot,
    vecmat,
)
from numpy._typing import (
    _ArrayLikeBool_co,
    _ArrayLikeInt,
    _ArrayLikeNumber_co,
    _DTypeLike,
    _NestedSequence,
    _NumberLike_co,
    _Shape,
)

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

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]

type _ErrKind = Literal["ignore", "warn", "raise", "call", "print", "log"]
type _ErrCall = Callable[[str, int], Any] | SupportsWrite[str]

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

# efdgFDGO => efdgFDGO
@type_check_only
class _ufunc_11_fco(np.ufunc):  # type: ignore[misc]
    @property
    @override
    def identity(self) -> None: ...
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
    @overload  # known shape, known scalar/array
    def __call__[T: np.inexact | npt.NDArray[np.inexact | np.object_]](
        self,
        x: T,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> T: ...
    @overload  # Nd, +f64
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.integer | np.bool]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: None = None,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> np.ndarray[ShapeT, np.dtype[np.float64]]: ...
    @overload  # scalar, float | +f64
    def __call__(
        self,
        x: float | np.integer | np.bool,
        /,
        *,
        out: None = None,
        dtype: None = None,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> np.float64: ...
    @overload  # 1d, +float
    def __call__(
        self,
        x: Sequence[float],
        /,
        *,
        out: None = None,
        dtype: None = None,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> _Array1D[np.float64]: ...
    @overload  # 1d, ~complex
    def __call__(
        self,
        x: list[complex],
        /,
        *,
        out: None = None,
        dtype: None = None,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> _Array1D[np.complex128]: ...
    @overload  # 2d, +float
    def __call__(
        self,
        x: Sequence[Sequence[float]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> _Array2D[np.float64]: ...
    @overload  # 2d, ~complex
    def __call__(
        self,
        x: Sequence[list[complex]],
        /,
        *,
        out: None = None,
        dtype: None = None,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> _Array2D[np.complex128]: ...
    @overload  # scalar, +complex  (overlaps with float)
    def __call__(
        self,
        x: complex,
        /,
        *,
        out: None = None,
        dtype: None = None,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> np.complex128 | Any: ...
    @overload  # scalar, dtype=<known>
    def __call__[ScalarT: np.inexact](
        self,
        x: _NumberLike_co,
        /,
        *,
        out: None = None,
        dtype: _DTypeLike[ScalarT],
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> ScalarT: ...
    @overload  # Nd, dtype=<known>
    def __call__[ShapeT: _Shape, ScalarT: np.inexact](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__[ShapeT: _Shape](
        self,
        x: np.ndarray[ShapeT, np.dtype[np.number | np.bool | np.object_]],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> np.ndarray[ShapeT]: ...
    @overload  # Nd, dtype=<known>
    def __call__[ScalarT: np.inexact | np.object_](
        self,
        x: _NestedSequence[complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> npt.NDArray[ScalarT]: ...
    @overload  # Nd, dtype=<unknown>
    def __call__(
        self,
        x: _NestedSequence[complex],
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> np.ndarray: ...
    @overload  # ?d, dtype=<known>
    def __call__[ScalarT: np.inexact | np.object_](
        self,
        x: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: _DTypeLike[ScalarT],
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> npt.NDArray[ScalarT] | Any: ...  # `| Any` because of overlap
    @overload  # ?d, dtype=<unknown>
    def __call__(
        self,
        x: _ArrayLikeNumber_co,
        /,
        *,
        out: EllipsisType | None = None,
        dtype: npt.DTypeLike,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> Any: ...
    @overload  # out=<given>
    def __call__[OutT: np.ndarray](
        self,
        x: _ArrayLikeNumber_co,
        /,
        out: OutT,
        *,
        dtype: npt.DTypeLike | None = None,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> OutT: ...
    @overload  # out=<given>
    def __call__[OutT](
        self,
        x: _CanUfuncCall1[OutT],
        /,
        out: object | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        where: _ArrayLikeBool_co = True,
        casting: _CastingKind = "same_kind",
        order: _OrderKACF = "K",
        subok: bool = True,
        signature: _Signature1 | None = None,
    ) -> OutT: ...

    #
    @override
    @overload
    def at(self, a: npt.NDArray[np.inexact | np.object_], indices: _ArrayLikeInt, /) -> None: ...  # pyrefly:ignore[bad-override]
    @overload
    def at[IxT, OutT](self, a: _CanUfuncAt1[IxT, OutT], indices: IxT, /) -> OutT: ...

    #
    @override
    def accumulate(self, array: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]
    @override
    def reduce(self, array: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]
    @override
    def reduceat(self, array: Never, /, indices: Never) -> Never: ...  # pyrefly:ignore[bad-override]
    @override
    def outer(self, A: Never, B: Never, /) -> Never: ...  # pyrefly:ignore[bad-override]

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
