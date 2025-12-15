# mypy: disable-error-code=no-untyped-def
# pyright: reportIncompatibleMethodOverride=false

import datetime as dt
import types
from _typeshed import Incomplete
from collections.abc import Buffer, Callable, Sequence
from typing import (
    Any,
    Concatenate,
    Final,
    Generic,
    Literal,
    Never,
    NoReturn,
    Self,
    SupportsComplex,
    SupportsFloat,
    SupportsIndex,
    SupportsInt,
    Unpack,
    final,
    overload,
    override,
)
from typing_extensions import TypeIs, TypeVar

import numpy as np
from numpy import (
    _HasDType,
    _HasDTypeWithRealAndImag,
    _ModeKind,
    _OrderACF,
    _OrderCF,
    _OrderKACF,
    _PartitionKind,
    _SortKind,
    _ToIndices,
    amax,
    amin,
    bool_,
    bytes_,
    complex128,
    complexfloating,
    datetime64,
    dtype,
    expand_dims,
    float64,
    floating,
    generic,
    inexact,
    int8,
    int64,
    int_,
    integer,
    intp,
    ndarray,
    number,
    object_,
    signedinteger,
    str_,
    timedelta64,
    unsignedinteger,
)
from numpy._core.fromnumeric import _UFuncKwargs  # type-check only
from numpy._globals import _NoValueType
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _32Bit,
    _64Bit,
    _AnyShape,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeBytes_co,
    _ArrayLikeComplex128_co,
    _ArrayLikeComplex_co,
    _ArrayLikeDT64_co,
    _ArrayLikeFloat64_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt,
    _ArrayLikeInt_co,
    _ArrayLikeNumber_co,
    _ArrayLikeObject_co,
    _ArrayLikeStr_co,
    _ArrayLikeString_co,
    _ArrayLikeTD64_co,
    _ArrayLikeUInt_co,
    _CharLike_co,
    _DTypeLike,
    _DTypeLikeBool,
    _DTypeLikeVoid,
    _FloatLike_co,
    _IntLike_co,
    _NestedSequence,
    _ScalarLike_co,
    _Shape,
    _ShapeLike,
    _SupportsArrayFunc,
    _SupportsDType,
    _TD64Like_co,
)
from numpy._typing._dtype_like import _VoidDTypeLike

__all__ = [
    "MAError",
    "MaskError",
    "MaskType",
    "MaskedArray",
    "abs",
    "absolute",
    "add",
    "all",
    "allclose",
    "allequal",
    "alltrue",
    "amax",
    "amin",
    "angle",
    "anom",
    "anomalies",
    "any",
    "append",
    "arange",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "argmax",
    "argmin",
    "argsort",
    "around",
    "array",
    "asanyarray",
    "asarray",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bool_",
    "ceil",
    "choose",
    "clip",
    "common_fill_value",
    "compress",
    "compressed",
    "concatenate",
    "conjugate",
    "convolve",
    "copy",
    "correlate",
    "cos",
    "cosh",
    "count",
    "cumprod",
    "cumsum",
    "default_fill_value",
    "diag",
    "diagonal",
    "diff",
    "divide",
    "empty",
    "empty_like",
    "equal",
    "exp",
    "expand_dims",
    "fabs",
    "filled",
    "fix_invalid",
    "flatten_mask",
    "flatten_structured_array",
    "floor",
    "floor_divide",
    "fmod",
    "frombuffer",
    "fromflex",
    "fromfunction",
    "getdata",
    "getmask",
    "getmaskarray",
    "greater",
    "greater_equal",
    "harden_mask",
    "hypot",
    "identity",
    "ids",
    "indices",
    "inner",
    "innerproduct",
    "isMA",
    "isMaskedArray",
    "is_mask",
    "is_masked",
    "isarray",
    "left_shift",
    "less",
    "less_equal",
    "log",
    "log2",
    "log10",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "make_mask",
    "make_mask_descr",
    "make_mask_none",
    "mask_or",
    "masked",
    "masked_array",
    "masked_equal",
    "masked_greater",
    "masked_greater_equal",
    "masked_inside",
    "masked_invalid",
    "masked_less",
    "masked_less_equal",
    "masked_not_equal",
    "masked_object",
    "masked_outside",
    "masked_print_option",
    "masked_singleton",
    "masked_values",
    "masked_where",
    "max",
    "maximum",
    "maximum_fill_value",
    "mean",
    "min",
    "minimum",
    "minimum_fill_value",
    "mod",
    "multiply",
    "mvoid",
    "ndim",
    "negative",
    "nomask",
    "nonzero",
    "not_equal",
    "ones",
    "ones_like",
    "outer",
    "outerproduct",
    "power",
    "prod",
    "product",
    "ptp",
    "put",
    "putmask",
    "ravel",
    "remainder",
    "repeat",
    "reshape",
    "resize",
    "right_shift",
    "round",
    "round_",
    "set_fill_value",
    "shape",
    "sin",
    "sinh",
    "size",
    "soften_mask",
    "sometrue",
    "sort",
    "sqrt",
    "squeeze",
    "std",
    "subtract",
    "sum",
    "swapaxes",
    "take",
    "tan",
    "tanh",
    "trace",
    "transpose",
    "true_divide",
    "var",
    "where",
    "zeros",
    "zeros_like",
]

_ShapeT_co = TypeVar("_ShapeT_co", bound=_Shape, default=_AnyShape, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype, default=np.dtype, covariant=True)
# the additional `Callable[...]` bound simplifies self-binding to the ufunc's callable signature
_UFuncT_co = TypeVar("_UFuncT_co", bound=np.ufunc | Callable[..., object], default=np.ufunc, covariant=True)

type _RealNumber = np.floating | np.integer

type _Ignored = object

# A subset of `MaskedArray` that can be parametrized w.r.t. `np.generic`
type _MaskedArray[ScalarT: np.generic] = MaskedArray[_AnyShape, np.dtype[ScalarT]]
type _Masked1D[ScalarT: np.generic] = MaskedArray[tuple[int], np.dtype[ScalarT]]

type _MaskedArrayUInt_co = _MaskedArray[np.unsignedinteger | np.bool]
type _MaskedArrayInt_co = _MaskedArray[np.integer | np.bool]
type _MaskedArrayFloat64_co = _MaskedArray[np.floating[_64Bit] | np.float32 | np.float16 | np.integer | np.bool]
type _MaskedArrayFloat_co = _MaskedArray[np.floating | np.integer | np.bool]
type _MaskedArrayComplex128_co = _MaskedArray[np.number[_64Bit] | np.number[_32Bit] | np.float16 | np.integer | np.bool]
type _MaskedArrayComplex_co = _MaskedArray[np.inexact | np.integer | np.bool]
type _MaskedArrayNumber_co = _MaskedArray[np.number | np.bool]
type _MaskedArrayTD64_co = _MaskedArray[np.timedelta64 | np.integer | np.bool]

type _ArrayInt_co = NDArray[np.integer | np.bool]
type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]

type _ConvertibleToInt = SupportsInt | SupportsIndex | _CharLike_co
type _ConvertibleToFloat = SupportsFloat | SupportsIndex | _CharLike_co
type _ConvertibleToComplex = SupportsComplex | SupportsFloat | SupportsIndex | _CharLike_co
type _ConvertibleToTD64 = dt.timedelta | int | _CharLike_co | np.character | np.number | np.timedelta64 | np.bool | None
type _ConvertibleToDT64 = dt.date | int | _CharLike_co | np.character | np.number | np.datetime64 | np.bool | None
type _ArangeScalar = _RealNumber | np.datetime64 | np.timedelta64

type _NoMaskType = np.bool_[Literal[False]]  # type of `np.False_`
type _MaskArray[ShapeT: _Shape] = np.ndarray[ShapeT, np.dtype[np.bool]]

type _FillValue = complex | None  # int | float | complex | None
type _FillValueCallable = Callable[[np.dtype | ArrayLike], _FillValue]
type _DomainCallable = Callable[..., NDArray[np.bool]]

###

MaskType = np.bool_

nomask: Final[_NoMaskType] = ...

class MaskedArrayFutureWarning(FutureWarning): ...
class MAError(Exception): ...
class MaskError(MAError): ...

# not generic at runtime
class _MaskedUFunc(Generic[_UFuncT_co]):
    f: _UFuncT_co  # readonly
    def __init__(self, /, ufunc: _UFuncT_co) -> None: ...

# not generic at runtime
class _MaskedUnaryOperation(_MaskedUFunc[_UFuncT_co], Generic[_UFuncT_co]):
    fill: Final[_FillValue]
    domain: Final[_DomainCallable | None]

    def __init__(self, /, mufunc: _UFuncT_co, fill: _FillValue = 0, domain: _DomainCallable | None = None) -> None: ...

    # NOTE: This might not work with overloaded callable signatures might not work on
    # pyright, which is a long-standing issue, and is unique to pyright:
    # https://github.com/microsoft/pyright/issues/9663
    # https://github.com/microsoft/pyright/issues/10849
    # https://github.com/microsoft/pyright/issues/10899
    # https://github.com/microsoft/pyright/issues/11049
    def __call__[**Tss, T](
        self: _MaskedUnaryOperation[Callable[Concatenate[Any, Tss], T]],
        /,
        a: ArrayLike,
        *args: Tss.args,
        **kwargs: Tss.kwargs,
    ) -> T: ...

# not generic at runtime
class _MaskedBinaryOperation(_MaskedUFunc[_UFuncT_co], Generic[_UFuncT_co]):
    fillx: Final[_FillValue]
    filly: Final[_FillValue]

    def __init__(self, /, mbfunc: _UFuncT_co, fillx: _FillValue = 0, filly: _FillValue = 0) -> None: ...

    # NOTE: See the comment in `_MaskedUnaryOperation.__call__`
    def __call__[**Tss, T](
        self: _MaskedBinaryOperation[Callable[Concatenate[Any, Any, Tss], T]],
        /,
        a: ArrayLike,
        b: ArrayLike,
        *args: Tss.args,
        **kwargs: Tss.kwargs,
    ) -> T: ...

    # NOTE: We cannot meaningfully annotate the return (d)types of these methods until
    # the signatures of the corresponding `numpy.ufunc` methods are specified.
    def reduce(self, /, target: ArrayLike, axis: SupportsIndex = 0, dtype: DTypeLike | None = None) -> Incomplete: ...
    def outer(self, /, a: ArrayLike, b: ArrayLike) -> _MaskedArray[Incomplete]: ...
    def accumulate(self, /, target: ArrayLike, axis: SupportsIndex = 0) -> _MaskedArray[Incomplete]: ...

# not generic at runtime
class _DomainedBinaryOperation(_MaskedUFunc[_UFuncT_co], Generic[_UFuncT_co]):
    domain: Final[_DomainCallable]
    fillx: Final[_FillValue]
    filly: Final[_FillValue]

    def __init__(
        self,
        /,
        dbfunc: _UFuncT_co,
        domain: _DomainCallable,
        fillx: _FillValue = 0,
        filly: _FillValue = 0,
    ) -> None: ...

    # NOTE: See the comment in `_MaskedUnaryOperation.__call__`
    def __call__[**Tss, T](
        self: _DomainedBinaryOperation[Callable[Concatenate[Any, Any, Tss], T]],
        /,
        a: ArrayLike,
        b: ArrayLike,
        *args: Tss.args,
        **kwargs: Tss.kwargs,
    ) -> T: ...

# not generic at runtime
class _extrema_operation(_MaskedUFunc[_UFuncT_co], Generic[_UFuncT_co]):
    compare: Final[_MaskedBinaryOperation]
    fill_value_func: Final[_FillValueCallable]

    def __init__(
        self,
        /,
        ufunc: _UFuncT_co,
        compare: _MaskedBinaryOperation,
        fill_value: _FillValueCallable,
    ) -> None: ...

    # NOTE: This class is only used internally for `maximum` and `minimum`, so we are
    # able to annotate the `__call__` method specifically for those two functions.
    @overload
    def __call__[ScalarT: np.generic](self, /, a: _ArrayLike[ScalarT], b: _ArrayLike[ScalarT]) -> _MaskedArray[ScalarT]: ...
    @overload
    def __call__(self, /, a: ArrayLike, b: ArrayLike) -> _MaskedArray[Incomplete]: ...

    # NOTE: We cannot meaningfully annotate the return (d)types of these methods until
    # the signatures of the corresponding `numpy.ufunc` methods are specified.
    def reduce(self, /, target: ArrayLike, axis: SupportsIndex | _NoValueType = ...) -> Incomplete: ...
    def outer(self, /, a: ArrayLike, b: ArrayLike) -> _MaskedArray[Incomplete]: ...

@final
class _MaskedPrintOption:
    _display: str
    _enabled: bool | Literal[0, 1]
    def __init__(self, /, display: str) -> None: ...
    def display(self, /) -> str: ...
    def set_display(self, /, s: str) -> None: ...
    def enabled(self, /) -> bool: ...
    def enable(self, /, shrink: bool | Literal[0, 1] = 1) -> None: ...

masked_print_option: Final[_MaskedPrintOption] = ...

exp: _MaskedUnaryOperation = ...
conjugate: _MaskedUnaryOperation = ...
sin: _MaskedUnaryOperation = ...
cos: _MaskedUnaryOperation = ...
arctan: _MaskedUnaryOperation = ...
arcsinh: _MaskedUnaryOperation = ...
sinh: _MaskedUnaryOperation = ...
cosh: _MaskedUnaryOperation = ...
tanh: _MaskedUnaryOperation = ...
abs: _MaskedUnaryOperation = ...
absolute: _MaskedUnaryOperation = ...
angle: _MaskedUnaryOperation = ...
fabs: _MaskedUnaryOperation = ...
negative: _MaskedUnaryOperation = ...
floor: _MaskedUnaryOperation = ...
ceil: _MaskedUnaryOperation = ...
around: _MaskedUnaryOperation = ...
logical_not: _MaskedUnaryOperation = ...
sqrt: _MaskedUnaryOperation = ...
log: _MaskedUnaryOperation = ...
log2: _MaskedUnaryOperation = ...
log10: _MaskedUnaryOperation = ...
tan: _MaskedUnaryOperation = ...
arcsin: _MaskedUnaryOperation = ...
arccos: _MaskedUnaryOperation = ...
arccosh: _MaskedUnaryOperation = ...
arctanh: _MaskedUnaryOperation = ...

add: _MaskedBinaryOperation = ...
subtract: _MaskedBinaryOperation = ...
multiply: _MaskedBinaryOperation = ...
arctan2: _MaskedBinaryOperation = ...
equal: _MaskedBinaryOperation = ...
not_equal: _MaskedBinaryOperation = ...
less_equal: _MaskedBinaryOperation = ...
greater_equal: _MaskedBinaryOperation = ...
less: _MaskedBinaryOperation = ...
greater: _MaskedBinaryOperation = ...
logical_and: _MaskedBinaryOperation = ...
def alltrue(target: ArrayLike, axis: SupportsIndex | None = 0, dtype: _DTypeLikeBool | None = None) -> Incomplete: ...
logical_or: _MaskedBinaryOperation = ...
def sometrue(target: ArrayLike, axis: SupportsIndex | None = 0, dtype: _DTypeLikeBool | None = None) -> Incomplete: ...
logical_xor: _MaskedBinaryOperation = ...
bitwise_and: _MaskedBinaryOperation = ...
bitwise_or: _MaskedBinaryOperation = ...
bitwise_xor: _MaskedBinaryOperation = ...
hypot: _MaskedBinaryOperation = ...

divide: _DomainedBinaryOperation = ...
true_divide: _DomainedBinaryOperation = ...
floor_divide: _DomainedBinaryOperation = ...
remainder: _DomainedBinaryOperation = ...
fmod: _DomainedBinaryOperation = ...
mod: _DomainedBinaryOperation = ...

# `obj` can be anything (even `object()`), and is too "flexible", so we can't
# meaningfully annotate it, or its return type.
def default_fill_value(obj: object) -> Any: ...
def minimum_fill_value(obj: object) -> Any: ...
def maximum_fill_value(obj: object) -> Any: ...

#
@overload  # returns `a.fill_value` if `a` is a `MaskedArray`
def get_fill_value[ScalarT: np.generic](a: _MaskedArray[ScalarT]) -> ScalarT: ...
@overload  # otherwise returns `default_fill_value(a)`
def get_fill_value(a: object) -> Any: ...

# this is a noop if `a` isn't a `MaskedArray`, so we only accept `MaskedArray` input
def set_fill_value(a: MaskedArray, fill_value: _ScalarLike_co) -> None: ...

# the return type depends on the *values* of `a` and `b` (which cannot be known
# statically), which is why we need to return an awkward `_ | None`
@overload
def common_fill_value[ScalarT: np.generic](a: _MaskedArray[ScalarT], b: MaskedArray) -> ScalarT | None: ...
@overload
def common_fill_value(a: object, b: object) -> Any: ...

# keep in sync with `fix_invalid`, but return `ndarray` instead of `MaskedArray`
@overload
def filled[ShapeT: _Shape, DTypeT: np.dtype](
    a: ndarray[ShapeT, DTypeT],
    fill_value: _ScalarLike_co | None = None,
) -> ndarray[ShapeT, DTypeT]: ...
@overload
def filled[ScalarT: np.generic](a: _ArrayLike[ScalarT], fill_value: _ScalarLike_co | None = None) -> NDArray[ScalarT]: ...
@overload
def filled(a: ArrayLike, fill_value: _ScalarLike_co | None = None) -> NDArray[Incomplete]: ...

# keep in sync with `filled`, but return `MaskedArray` instead of `ndarray`
@overload
def fix_invalid[ShapeT: _Shape, DTypeT: np.dtype](
    a: np.ndarray[ShapeT, DTypeT],
    mask: _ArrayLikeBool_co = nomask,
    copy: bool = True,
    fill_value: _ScalarLike_co | None = None,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload
def fix_invalid[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    mask: _ArrayLikeBool_co = nomask,
    copy: bool = True,
    fill_value: _ScalarLike_co | None = None,
) -> _MaskedArray[ScalarT]: ...
@overload
def fix_invalid(
    a: ArrayLike,
    mask: _ArrayLikeBool_co = nomask,
    copy: bool = True,
    fill_value: _ScalarLike_co | None = None,
) -> _MaskedArray[Incomplete]: ...

#
def get_masked_subclass(*arrays: object) -> type[MaskedArray]: ...

#
@overload
def getdata[ShapeT: _Shape, DTypeT: np.dtype](
    a: np.ndarray[ShapeT, DTypeT],
    subok: bool = True,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload
def getdata[ScalarT: np.generic](a: _ArrayLike[ScalarT], subok: bool = True) -> NDArray[ScalarT]: ...
@overload
def getdata(a: ArrayLike, subok: bool = True) -> NDArray[Incomplete]: ...

get_data = getdata

#
@overload
def getmask(a: _ScalarLike_co) -> _NoMaskType: ...
@overload
def getmask[ShapeT: _Shape](a: MaskedArray[ShapeT, Any]) -> _MaskArray[ShapeT] | _NoMaskType: ...
@overload
def getmask(a: ArrayLike) -> _MaskArray[_AnyShape] | _NoMaskType: ...

get_mask = getmask

# like `getmask`, but instead of `nomask` returns `make_mask_none(arr, arr.dtype?)`
@overload
def getmaskarray(arr: _ScalarLike_co) -> _MaskArray[tuple[()]]: ...
@overload
def getmaskarray[ShapeT: _Shape](arr: np.ndarray[ShapeT, Any]) -> _MaskArray[ShapeT]: ...

# It's sufficient for `m` to have dtype with type: `type[np.bool_]`,
# which isn't necessarily a ndarray. Please open an issue if this causes issues.
def is_mask(m: object) -> TypeIs[NDArray[bool_]]: ...

#
@overload
def make_mask_descr(ndtype: _VoidDTypeLike) -> np.dtype[np.void]: ...
@overload
def make_mask_descr(ndtype: _DTypeLike[np.generic] | str | type) -> np.dtype[np.bool_]: ...

#
@overload  # m is nomask
def make_mask(
    m: _NoMaskType,
    copy: bool = False,
    shrink: bool = True,
    dtype: _DTypeLikeBool = ...,
) -> _NoMaskType: ...
@overload  # m: ndarray, shrink=True (default), dtype: bool-like (default)
def make_mask[ShapeT: _Shape](
    m: np.ndarray[ShapeT],
    copy: bool = False,
    shrink: Literal[True] = True,
    dtype: _DTypeLikeBool = ...,
) -> _MaskArray[ShapeT] | _NoMaskType: ...
@overload  # m: ndarray, shrink=False (kwarg), dtype: bool-like (default)
def make_mask[ShapeT: _Shape](
    m: np.ndarray[ShapeT],
    copy: bool = False,
    *,
    shrink: Literal[False],
    dtype: _DTypeLikeBool = ...,
) -> _MaskArray[ShapeT]: ...
@overload  # m: ndarray, dtype: void-like
def make_mask[ShapeT: _Shape](
    m: np.ndarray[ShapeT],
    copy: bool = False,
    shrink: bool = True,
    *,
    dtype: _DTypeLikeVoid,
) -> np.ndarray[ShapeT, np.dtype[np.void]]: ...
@overload  # m: array-like, shrink=True (default), dtype: bool-like (default)
def make_mask(
    m: ArrayLike,
    copy: bool = False,
    shrink: Literal[True] = True,
    dtype: _DTypeLikeBool = ...,
) -> _MaskArray[_AnyShape] | _NoMaskType: ...
@overload  # m: array-like, shrink=False (kwarg), dtype: bool-like (default)
def make_mask(
    m: ArrayLike,
    copy: bool = False,
    *,
    shrink: Literal[False],
    dtype: _DTypeLikeBool = ...,
) -> _MaskArray[_AnyShape]: ...
@overload  # m: array-like, dtype: void-like
def make_mask(
    m: ArrayLike,
    copy: bool = False,
    shrink: bool = True,
    *,
    dtype: _DTypeLikeVoid,
) -> NDArray[np.void]: ...
@overload  # fallback
def make_mask(
    m: ArrayLike,
    copy: bool = False,
    shrink: bool = True,
    *,
    dtype: DTypeLike = ...,
) -> NDArray[Incomplete] | _NoMaskType: ...

#
@overload  # known shape, dtype: unstructured (default)
def make_mask_none[ShapeT: _Shape](newshape: ShapeT, dtype: np.dtype | type | str | None = None) -> _MaskArray[ShapeT]: ...
@overload  # known shape, dtype: structured
def make_mask_none[ShapeT: _Shape](newshape: ShapeT, dtype: _VoidDTypeLike) -> np.ndarray[ShapeT, dtype[np.void]]: ...
@overload  # unknown shape, dtype: unstructured (default)
def make_mask_none(newshape: _ShapeLike, dtype: np.dtype | type | str | None = None) -> _MaskArray[_AnyShape]: ...
@overload  # unknown shape, dtype: structured
def make_mask_none(newshape: _ShapeLike, dtype: _VoidDTypeLike) -> NDArray[np.void]: ...

#
@overload  # nomask, scalar-like, shrink=True (default)
def mask_or(
    m1: _NoMaskType | Literal[False],
    m2: _ScalarLike_co,
    copy: bool = False,
    shrink: Literal[True] = True,
) -> _NoMaskType: ...
@overload  # nomask, scalar-like, shrink=False (kwarg)
def mask_or(
    m1: _NoMaskType | Literal[False],
    m2: _ScalarLike_co,
    copy: bool = False,
    *,
    shrink: Literal[False],
) -> _MaskArray[tuple[()]]: ...
@overload  # scalar-like, nomask, shrink=True (default)
def mask_or(
    m1: _ScalarLike_co,
    m2: _NoMaskType | Literal[False],
    copy: bool = False,
    shrink: Literal[True] = True,
) -> _NoMaskType: ...
@overload  # scalar-like, nomask, shrink=False (kwarg)
def mask_or(
    m1: _ScalarLike_co,
    m2: _NoMaskType | Literal[False],
    copy: bool = False,
    *,
    shrink: Literal[False],
) -> _MaskArray[tuple[()]]: ...
@overload  # ndarray, ndarray | nomask, shrink=True (default)
def mask_or[ShapeT: _Shape, ScalarT: np.generic](
    m1: np.ndarray[ShapeT, np.dtype[ScalarT]],
    m2: np.ndarray[ShapeT, np.dtype[ScalarT]] | _NoMaskType | Literal[False],
    copy: bool = False,
    shrink: Literal[True] = True,
) -> _MaskArray[ShapeT] | _NoMaskType: ...
@overload  # ndarray, ndarray | nomask, shrink=False (kwarg)
def mask_or[ShapeT: _Shape, ScalarT: np.generic](
    m1: np.ndarray[ShapeT, np.dtype[ScalarT]],
    m2: np.ndarray[ShapeT, np.dtype[ScalarT]] | _NoMaskType | Literal[False],
    copy: bool = False,
    *,
    shrink: Literal[False],
) -> _MaskArray[ShapeT]: ...
@overload  # ndarray | nomask, ndarray, shrink=True (default)
def mask_or[ShapeT: _Shape, ScalarT: np.generic](
    m1: np.ndarray[ShapeT, np.dtype[ScalarT]] | _NoMaskType | Literal[False],
    m2: np.ndarray[ShapeT, np.dtype[ScalarT]],
    copy: bool = False,
    shrink: Literal[True] = True,
) -> _MaskArray[ShapeT] | _NoMaskType: ...
@overload  # ndarray | nomask, ndarray, shrink=False (kwarg)
def mask_or[ShapeT: _Shape, ScalarT: np.generic](
    m1: np.ndarray[ShapeT, np.dtype[ScalarT]] | _NoMaskType | Literal[False],
    m2: np.ndarray[ShapeT, np.dtype[ScalarT]],
    copy: bool = False,
    *,
    shrink: Literal[False],
) -> _MaskArray[ShapeT]: ...

#
@overload
def flatten_mask[ShapeT: _Shape](mask: np.ndarray[ShapeT]) -> _MaskArray[ShapeT]: ...
@overload
def flatten_mask(mask: ArrayLike) -> _MaskArray[_AnyShape]: ...

# NOTE: we currently don't know the field types of `void` dtypes, so it's not possible
# to know the output dtype of the returned array.
@overload
def flatten_structured_array[ShapeT: _Shape](a: MaskedArray[ShapeT, np.dtype[np.void]]) -> MaskedArray[ShapeT]: ...
@overload
def flatten_structured_array[ShapeT: _Shape](a: np.ndarray[ShapeT, np.dtype[np.void]]) -> np.ndarray[ShapeT]: ...
@overload  # for some reason this accepts unstructured array-likes, hence this fallback overload
def flatten_structured_array(a: ArrayLike) -> np.ndarray: ...

# keep in sync with other the `masked_*` functions
@overload  # known array with known shape and dtype
def masked_invalid[ShapeT: _Shape, DTypeT: np.dtype](
    a: ndarray[ShapeT, DTypeT],
    copy: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload  # array-like of known scalar-type
def masked_invalid[ScalarT: np.generic](a: _ArrayLike[ScalarT], copy: bool = True) -> _MaskedArray[ScalarT]: ...
@overload  # unknown array-like
def masked_invalid(a: ArrayLike, copy: bool = True) -> _MaskedArray[Incomplete]: ...

# keep in sync with other the `masked_*` functions
@overload  # array-like of known scalar-type
def masked_where[ShapeT: _Shape, DTypeT: np.dtype](
    condition: _ArrayLikeBool_co,
    a: ndarray[ShapeT, DTypeT],
    copy: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload  # array-like of known scalar-type
def masked_where[ScalarT: np.generic](
    condition: _ArrayLikeBool_co,
    a: _ArrayLike[ScalarT],
    copy: bool = True,
) -> _MaskedArray[ScalarT]: ...
@overload  # unknown array-like
def masked_where(condition: _ArrayLikeBool_co, a: ArrayLike, copy: bool = True) -> _MaskedArray[Incomplete]: ...

# keep in sync with other the `masked_*` functions
@overload  # known array with known shape and dtype
def masked_greater[ShapeT: _Shape, DTypeT: np.dtype](
    x: ndarray[ShapeT, DTypeT],
    value: ArrayLike,
    copy: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload  # array-like of known scalar-type
def masked_greater[ScalarT: np.generic](x: _ArrayLike[ScalarT], value: ArrayLike, copy: bool = True) -> _MaskedArray[ScalarT]: ...
@overload  # unknown array-like
def masked_greater(x: ArrayLike, value: ArrayLike, copy: bool = True) -> _MaskedArray[Incomplete]: ...

# keep in sync with other the `masked_*` functions
@overload  # known array with known shape and dtype
def masked_greater_equal[ShapeT: _Shape, DTypeT: np.dtype](
    x: ndarray[ShapeT, DTypeT],
    value: ArrayLike,
    copy: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload  # array-like of known scalar-type
def masked_greater_equal[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    value: ArrayLike,
    copy: bool = True,
) -> _MaskedArray[ScalarT]: ...
@overload  # unknown array-like
def masked_greater_equal(x: ArrayLike, value: ArrayLike, copy: bool = True) -> _MaskedArray[Incomplete]: ...

# keep in sync with other the `masked_*` functions
@overload  # known array with known shape and dtype
def masked_less[ShapeT: _Shape, DTypeT: np.dtype](
    x: ndarray[ShapeT, DTypeT],
    value: ArrayLike,
    copy: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload  # array-like of known scalar-type
def masked_less[ScalarT: np.generic](x: _ArrayLike[ScalarT], value: ArrayLike, copy: bool = True) -> _MaskedArray[ScalarT]: ...
@overload  # unknown array-like
def masked_less(x: ArrayLike, value: ArrayLike, copy: bool = True) -> _MaskedArray[Incomplete]: ...

# keep in sync with other the `masked_*` functions
@overload  # known array with known shape and dtype
def masked_less_equal[ShapeT: _Shape, DTypeT: np.dtype](
    x: ndarray[ShapeT, DTypeT],
    value: ArrayLike,
    copy: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload  # array-like of known scalar-type
def masked_less_equal[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    value: ArrayLike,
    copy: bool = True,
) -> _MaskedArray[ScalarT]: ...
@overload  # unknown array-like
def masked_less_equal(x: ArrayLike, value: ArrayLike, copy: bool = True) -> _MaskedArray[Incomplete]: ...

# keep in sync with other the `masked_*` functions
@overload  # known array with known shape and dtype
def masked_not_equal[ShapeT: _Shape, DTypeT: np.dtype](
    x: ndarray[ShapeT, DTypeT],
    value: ArrayLike,
    copy: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload  # array-like of known scalar-type
def masked_not_equal[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    value: ArrayLike,
    copy: bool = True,
) -> _MaskedArray[ScalarT]: ...
@overload  # unknown array-like
def masked_not_equal(x: ArrayLike, value: ArrayLike, copy: bool = True) -> _MaskedArray[Incomplete]: ...

# keep in sync with other the `masked_*` functions
@overload  # known array with known shape and dtype
def masked_equal[ShapeT: _Shape, DTypeT: np.dtype](
    x: ndarray[ShapeT, DTypeT],
    value: ArrayLike,
    copy: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload  # array-like of known scalar-type
def masked_equal[ScalarT: np.generic](x: _ArrayLike[ScalarT], value: ArrayLike, copy: bool = True) -> _MaskedArray[ScalarT]: ...
@overload  # unknown array-like
def masked_equal(x: ArrayLike, value: ArrayLike, copy: bool = True) -> _MaskedArray[Incomplete]: ...

# keep in sync with other the `masked_*` functions
@overload  # known array with known shape and dtype
def masked_inside[ShapeT: _Shape, DTypeT: np.dtype](
    x: ndarray[ShapeT, DTypeT],
    v1: ArrayLike,
    v2: ArrayLike,
    copy: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload  # array-like of known scalar-type
def masked_inside[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    v1: ArrayLike,
    v2: ArrayLike,
    copy: bool = True,
) -> _MaskedArray[ScalarT]: ...
@overload  # unknown array-like
def masked_inside(x: ArrayLike, v1: ArrayLike, v2: ArrayLike, copy: bool = True) -> _MaskedArray[Incomplete]: ...

# keep in sync with other the `masked_*` functions
@overload  # known array with known shape and dtype
def masked_outside[ShapeT: _Shape, DTypeT: np.dtype](
    x: ndarray[ShapeT, DTypeT],
    v1: ArrayLike,
    v2: ArrayLike,
    copy: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload  # array-like of known scalar-type
def masked_outside[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    v1: ArrayLike,
    v2: ArrayLike,
    copy: bool = True,
) -> _MaskedArray[ScalarT]: ...
@overload  # unknown array-like
def masked_outside(x: ArrayLike, v1: ArrayLike, v2: ArrayLike, copy: bool = True) -> _MaskedArray[Incomplete]: ...

# only intended for object arrays, so we assume that's how it's always used in practice
@overload
def masked_object[ShapeT: _Shape](
    x: np.ndarray[ShapeT, np.dtype[np.object_]],
    value: object,
    copy: bool = True,
    shrink: bool = True,
) -> MaskedArray[ShapeT, np.dtype[np.object_]]: ...
@overload
def masked_object(
    x: _ArrayLikeObject_co,
    value: object,
    copy: bool = True,
    shrink: bool = True,
) -> _MaskedArray[np.object_]: ...

# keep roughly in sync with `filled`
@overload
def masked_values[ShapeT: _Shape, DTypeT: np.dtype](
    x: np.ndarray[ShapeT, DTypeT],
    value: _ScalarLike_co,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    copy: bool = True,
    shrink: bool = True,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload
def masked_values[ScalarT: np.generic](
    x: _ArrayLike[ScalarT],
    value: _ScalarLike_co,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    copy: bool = True,
    shrink: bool = True,
) -> _MaskedArray[ScalarT]: ...
@overload
def masked_values(
    x: ArrayLike,
    value: _ScalarLike_co,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    copy: bool = True,
    shrink: bool = True,
) -> _MaskedArray[Incomplete]: ...

# TODO: Support non-boolean mask dtypes, such as `np.void`. This will require adding an
# additional generic type parameter to (at least) `MaskedArray` and `MaskedIterator` to
# hold the dtype of the mask.

class MaskedIterator(Generic[_ShapeT_co, _DTypeT_co]):
    ma: MaskedArray[_ShapeT_co, _DTypeT_co]  # readonly
    dataiter: np.flatiter[ndarray[_ShapeT_co, _DTypeT_co]]  # readonly
    maskiter: Final[np.flatiter[NDArray[np.bool]]]

    def __init__(self, ma: MaskedArray[_ShapeT_co, _DTypeT_co]) -> None: ...
    def __iter__(self) -> Self: ...

    # Similar to `MaskedArray.__getitem__` but without the `void` case.
    @overload
    def __getitem__(self, indx: _ArrayInt_co | tuple[_ArrayInt_co, ...], /) -> MaskedArray[_AnyShape, _DTypeT_co]: ...
    @overload
    def __getitem__(self, indx: SupportsIndex | tuple[SupportsIndex, ...], /) -> Incomplete: ...
    @overload
    def __getitem__(self, indx: _ToIndices, /) -> MaskedArray[_AnyShape, _DTypeT_co]: ...

    # Similar to `ndarray.__setitem__` but without the `void` case.
    @overload  # flexible | object_ | bool
    def __setitem__(
        self: MaskedIterator[Any, dtype[np.flexible | object_ | np.bool] | np.dtypes.StringDType],
        index: _ToIndices,
        value: object,
        /,
    ) -> None: ...
    @overload  # integer
    def __setitem__(
        self: MaskedIterator[Any, dtype[integer]],
        index: _ToIndices,
        value: _ConvertibleToInt | _NestedSequence[_ConvertibleToInt] | _ArrayLikeInt_co,
        /,
    ) -> None: ...
    @overload  # floating
    def __setitem__(
        self: MaskedIterator[Any, dtype[floating]],
        index: _ToIndices,
        value: _ConvertibleToFloat | _NestedSequence[_ConvertibleToFloat | None] | _ArrayLikeFloat_co | None,
        /,
    ) -> None: ...
    @overload  # complexfloating
    def __setitem__(
        self: MaskedIterator[Any, dtype[complexfloating]],
        index: _ToIndices,
        value: _ConvertibleToComplex | _NestedSequence[_ConvertibleToComplex | None] | _ArrayLikeNumber_co | None,
        /,
    ) -> None: ...
    @overload  # timedelta64
    def __setitem__(
        self: MaskedIterator[Any, dtype[timedelta64]],
        index: _ToIndices,
        value: _ConvertibleToTD64 | _NestedSequence[_ConvertibleToTD64],
        /,
    ) -> None: ...
    @overload  # datetime64
    def __setitem__(
        self: MaskedIterator[Any, dtype[datetime64]],
        index: _ToIndices,
        value: _ConvertibleToDT64 | _NestedSequence[_ConvertibleToDT64],
        /,
    ) -> None: ...
    @overload  # catch-all
    def __setitem__(self, index: _ToIndices, value: ArrayLike, /) -> None: ...

    # TODO: Returns `mvoid[(), _DTypeT_co]` for masks with `np.void` dtype.
    def __next__[ScalarT: np.generic](self: MaskedIterator[Any, np.dtype[ScalarT]]) -> ScalarT: ...

class MaskedArray(ndarray[_ShapeT_co, _DTypeT_co]):
    __array_priority__: Final[Literal[15]] = 15

    @overload
    def __new__[ScalarT: np.generic](
        cls,
        data: _ArrayLike[ScalarT],
        mask: _ArrayLikeBool_co = nomask,
        dtype: None = None,
        copy: bool = False,
        subok: bool = True,
        ndmin: int = 0,
        fill_value: _ScalarLike_co | None = None,
        keep_mask: bool = True,
        hard_mask: bool | None = None,
        shrink: bool = True,
        order: _OrderKACF | None = None,
    ) -> _MaskedArray[ScalarT]: ...
    @overload
    def __new__[ScalarT: np.generic](
        cls,
        data: object,
        mask: _ArrayLikeBool_co,
        dtype: _DTypeLike[ScalarT],
        copy: bool = False,
        subok: bool = True,
        ndmin: int = 0,
        fill_value: _ScalarLike_co | None = None,
        keep_mask: bool = True,
        hard_mask: bool | None = None,
        shrink: bool = True,
        order: _OrderKACF | None = None,
    ) -> _MaskedArray[ScalarT]: ...
    @overload
    def __new__[ScalarT: np.generic](
        cls,
        data: object,
        mask: _ArrayLikeBool_co = nomask,
        *,
        dtype: _DTypeLike[ScalarT],
        copy: bool = False,
        subok: bool = True,
        ndmin: int = 0,
        fill_value: _ScalarLike_co | None = None,
        keep_mask: bool = True,
        hard_mask: bool | None = None,
        shrink: bool = True,
        order: _OrderKACF | None = None,
    ) -> _MaskedArray[ScalarT]: ...
    @overload
    def __new__(
        cls,
        data: object = None,
        mask: _ArrayLikeBool_co = nomask,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        subok: bool = True,
        ndmin: int = 0,
        fill_value: _ScalarLike_co | None = None,
        keep_mask: bool = True,
        hard_mask: bool | None = None,
        shrink: bool = True,
        order: _OrderKACF | None = None,
    ) -> _MaskedArray[Any]: ...

    def __array_wrap__[ShapeT: _Shape, DTypeT: np.dtype](
        self,
        obj: ndarray[ShapeT, DTypeT],
        context: tuple[np.ufunc, tuple[Any, ...], int] | None = None,
        return_scalar: bool = False,
    ) -> MaskedArray[ShapeT, DTypeT]: ...

    @overload  # type: ignore[override]  # ()
    def view(self, /, dtype: None = None, type: None = None, fill_value: _ScalarLike_co | None = None) -> Self: ...
    @overload  # (dtype: DTypeT)
    def view[DTypeT: np.dtype](
        self,
        /,
        dtype: DTypeT | _HasDType[DTypeT],
        type: None = None,
        fill_value: _ScalarLike_co | None = None,
    ) -> MaskedArray[_ShapeT_co, DTypeT]: ...
    @overload  # (dtype: dtype[ScalarT])
    def view[ScalarT: np.generic](
        self,
        /,
        dtype: _DTypeLike[ScalarT],
        type: None = None,
        fill_value: _ScalarLike_co | None = None,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload  # ([dtype: _, ]*, type: ArrayT)
    def view[ArrayT: np.ndarray](
        self,
        /,
        dtype: DTypeLike | None = None,
        *,
        type: type[ArrayT],
        fill_value: _ScalarLike_co | None = None,
    ) -> ArrayT: ...
    @overload  # (dtype: _, type: ArrayT)
    def view[ArrayT: np.ndarray](
        self,
        /,
        dtype: DTypeLike | None,
        type: type[ArrayT],
        fill_value: _ScalarLike_co | None = None,
    ) -> ArrayT: ...
    @overload  # (dtype: ArrayT, /)
    def view[ArrayT: np.ndarray](
        self,
        /,
        dtype: type[ArrayT],
        type: None = None,
        fill_value: _ScalarLike_co | None = None,
    ) -> ArrayT: ...
    @overload  # (dtype: ?)
    def view(
        self,
        /,
        # `_VoidDTypeLike | str | None` is like `DTypeLike` but without `_DTypeLike[Any]` to avoid
        # overlaps with previous overloads.
        dtype: _VoidDTypeLike | str | None,
        type: None = None,
        fill_value: _ScalarLike_co | None = None,
    ) -> MaskedArray[_ShapeT_co, np.dtype]: ...

    # Keep in sync with `ndarray.__getitem__`
    @overload
    def __getitem__(self, key: _ArrayInt_co | tuple[_ArrayInt_co, ...], /) -> MaskedArray[_AnyShape, _DTypeT_co]: ...
    @overload
    def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...], /) -> Any: ...
    @overload
    def __getitem__(self, key: _ToIndices, /) -> MaskedArray[_AnyShape, _DTypeT_co]: ...
    @overload
    def __getitem__(self: _MaskedArray[np.void], indx: str, /) -> MaskedArray[_ShapeT_co]: ...
    @overload
    def __getitem__(self: _MaskedArray[np.void], indx: list[str], /) -> MaskedArray[_ShapeT_co, dtype[np.void]]: ...

    @property
    def shape(self) -> _ShapeT_co: ...
    @shape.setter  # type: ignore[override]
    def shape[ShapeT: _Shape](self: MaskedArray[ShapeT, Any], shape: ShapeT, /) -> None: ...

    def __setmask__(self, mask: _ArrayLikeBool_co, copy: bool = False) -> None: ...
    @property
    def mask(self) -> np.ndarray[_ShapeT_co, dtype[MaskType]] | MaskType: ...
    @mask.setter
    def mask(self, value: _ArrayLikeBool_co, /) -> None: ...
    @property
    def recordmask(self) -> np.ndarray[_ShapeT_co, dtype[MaskType]] | MaskType: ...
    @recordmask.setter
    def recordmask(self, mask: Never, /) -> NoReturn: ...
    def harden_mask(self) -> Self: ...
    def soften_mask(self) -> Self: ...
    @property
    def hardmask(self) -> bool: ...
    def unshare_mask(self) -> Self: ...
    @property
    def sharedmask(self) -> bool: ...
    def shrink_mask(self) -> Self: ...

    @property
    def baseclass(self) -> type[ndarray]: ...

    @property
    def _data(self) -> ndarray[_ShapeT_co, _DTypeT_co]: ...
    @property
    def data(self) -> ndarray[_ShapeT_co, _DTypeT_co]: ...  # type: ignore[override]

    @property  # type: ignore[override]
    def flat(self) -> MaskedIterator[_ShapeT_co, _DTypeT_co]: ...
    @flat.setter
    def flat(self, value: ArrayLike, /) -> None: ...

    @property
    def fill_value[ScalarT: np.generic](self: _MaskedArray[ScalarT]) -> ScalarT: ...
    @fill_value.setter
    def fill_value(self, value: _ScalarLike_co | None = None, /) -> None: ...

    def get_fill_value[ScalarT: np.generic](self: _MaskedArray[ScalarT]) -> ScalarT: ...
    def set_fill_value(self, /, value: _ScalarLike_co | None = None) -> None: ...

    def filled(self, /, fill_value: _ScalarLike_co | None = None) -> ndarray[_ShapeT_co, _DTypeT_co]: ...
    def compressed(self) -> ndarray[tuple[int], _DTypeT_co]: ...

    # keep roughly in sync with `ma.core.compress`, but swap the first two arguments
    @overload  # type: ignore[override]
    def compress[ArrayT: np.ndarray](
        self,
        condition: _ArrayLikeBool_co,
        axis: _ShapeLike | None,
        out: ArrayT,
    ) -> ArrayT: ...
    @overload
    def compress[ArrayT: np.ndarray](
        self,
        condition: _ArrayLikeBool_co,
        axis: _ShapeLike | None = None,
        *,
        out: ArrayT,
    ) -> ArrayT: ...
    @overload
    def compress(
        self,
        condition: _ArrayLikeBool_co,
        axis: None = None,
        out: None = None,
    ) -> MaskedArray[tuple[int], _DTypeT_co]: ...
    @overload
    def compress(
        self,
        condition: _ArrayLikeBool_co,
        axis: _ShapeLike | None = None,
        out: None = None,
    ) -> MaskedArray[_AnyShape, _DTypeT_co]: ...

    # TODO: How to deal with the non-commutative nature of `==` and `!=`?
    # xref numpy/numpy#17368
    def __eq__(self, other: Incomplete, /) -> Incomplete: ...
    def __ne__(self, other: Incomplete, /) -> Incomplete: ...

    def __ge__(self, other: ArrayLike, /) -> _MaskedArray[bool_]: ...  # type: ignore[override]
    def __gt__(self, other: ArrayLike, /) -> _MaskedArray[bool_]: ...  # type: ignore[override]
    def __le__(self, other: ArrayLike, /) -> _MaskedArray[bool_]: ...  # type: ignore[override]
    def __lt__(self, other: ArrayLike, /) -> _MaskedArray[bool_]: ...  # type: ignore[override]

    # Keep in sync with `ndarray.__add__`
    @overload  # type: ignore[override]
    def __add__[ScalarT: np.number](
        self: _MaskedArray[ScalarT],
        other: int | np.bool,
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload
    def __add__[ScalarT: np.number](self: _MaskedArray[ScalarT], other: _ArrayLikeBool_co, /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __add__(self: _MaskedArray[np.bool], other: _ArrayLikeBool_co, /) -> _MaskedArray[np.bool]: ...
    @overload
    def __add__[ScalarT: np.number](self: _MaskedArray[np.bool], other: _ArrayLike[ScalarT], /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __add__(self: _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __add__(self: _MaskedArrayFloat64_co, other: _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __add__(self: _MaskedArray[complex128], other: _ArrayLikeComplex128_co, /) -> _MaskedArray[complex128]: ...
    @overload
    def __add__(self: _MaskedArrayComplex128_co, other: _ArrayLike[complexfloating[_64Bit]], /) -> _MaskedArray[complex128]: ...
    @overload
    def __add__(self: _MaskedArrayUInt_co, other: _ArrayLikeUInt_co, /) -> _MaskedArray[unsignedinteger]: ...
    @overload
    def __add__(self: _MaskedArrayInt_co, other: _ArrayLikeInt_co, /) -> _MaskedArray[signedinteger]: ...
    @overload
    def __add__(self: _MaskedArrayFloat_co, other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __add__(self: _MaskedArrayComplex_co, other: _ArrayLikeComplex_co, /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __add__(self: _MaskedArray[number], other: _ArrayLikeNumber_co, /) -> _MaskedArray[number]: ...
    @overload
    def __add__(self: _MaskedArrayTD64_co, other: _ArrayLikeTD64_co, /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __add__(self: _MaskedArrayTD64_co, other: _ArrayLikeDT64_co, /) -> _MaskedArray[datetime64]: ...
    @overload
    def __add__(self: _MaskedArray[datetime64], other: _ArrayLikeTD64_co, /) -> _MaskedArray[datetime64]: ...
    @overload
    def __add__(self: _MaskedArray[bytes_], other: _ArrayLikeBytes_co, /) -> _MaskedArray[bytes_]: ...
    @overload
    def __add__(self: _MaskedArray[str_], other: _ArrayLikeStr_co, /) -> _MaskedArray[str_]: ...
    @overload
    def __add__(
        self: MaskedArray[Any, np.dtypes.StringDType],
        other: _ArrayLikeStr_co | _ArrayLikeString_co,
        /,
    ) -> MaskedArray[_AnyShape, np.dtypes.StringDType]: ...
    @overload
    def __add__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __add__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__radd__`
    @overload  # type: ignore[override]  # signature equivalent to __add__
    def __radd__[ScalarT: np.number](
        self: _MaskedArray[ScalarT],
        other: int | np.bool,
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload
    def __radd__[ScalarT: np.number](self: _MaskedArray[ScalarT], other: _ArrayLikeBool_co, /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __radd__(self: _MaskedArray[np.bool], other: _ArrayLikeBool_co, /) -> _MaskedArray[np.bool]: ...
    @overload
    def __radd__[ScalarT: np.number](self: _MaskedArray[np.bool], other: _ArrayLike[ScalarT], /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __radd__(self: _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __radd__(self: _MaskedArrayFloat64_co, other: _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __radd__(self: _MaskedArray[complex128], other: _ArrayLikeComplex128_co, /) -> _MaskedArray[complex128]: ...
    @overload
    def __radd__(self: _MaskedArrayComplex128_co, other: _ArrayLike[complexfloating[_64Bit]], /) -> _MaskedArray[complex128]: ...
    @overload
    def __radd__(self: _MaskedArrayUInt_co, other: _ArrayLikeUInt_co, /) -> _MaskedArray[unsignedinteger]: ...
    @overload
    def __radd__(self: _MaskedArrayInt_co, other: _ArrayLikeInt_co, /) -> _MaskedArray[signedinteger]: ...
    @overload
    def __radd__(self: _MaskedArrayFloat_co, other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __radd__(self: _MaskedArrayComplex_co, other: _ArrayLikeComplex_co, /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __radd__(self: _MaskedArray[number], other: _ArrayLikeNumber_co, /) -> _MaskedArray[number]: ...
    @overload
    def __radd__(self: _MaskedArrayTD64_co, other: _ArrayLikeTD64_co, /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __radd__(self: _MaskedArrayTD64_co, other: _ArrayLikeDT64_co, /) -> _MaskedArray[datetime64]: ...
    @overload
    def __radd__(self: _MaskedArray[datetime64], other: _ArrayLikeTD64_co, /) -> _MaskedArray[datetime64]: ...
    @overload
    def __radd__(self: _MaskedArray[bytes_], other: _ArrayLikeBytes_co, /) -> _MaskedArray[bytes_]: ...
    @overload
    def __radd__(self: _MaskedArray[str_], other: _ArrayLikeStr_co, /) -> _MaskedArray[str_]: ...
    @overload
    def __radd__(
        self: MaskedArray[Any, np.dtypes.StringDType],
        other: _ArrayLikeStr_co | _ArrayLikeString_co,
        /,
    ) -> MaskedArray[_AnyShape, np.dtypes.StringDType]: ...
    @overload
    def __radd__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __radd__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__sub__`
    @overload  # type: ignore[override]
    def __sub__[ScalarT: np.number](
        self: _MaskedArray[ScalarT],
        other: int | np.bool,
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload
    def __sub__[ScalarT: np.number](self: _MaskedArray[ScalarT], other: _ArrayLikeBool_co, /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __sub__(self: _MaskedArray[np.bool], other: _ArrayLikeBool_co, /) -> NoReturn: ...
    @overload
    def __sub__[ScalarT: np.number](self: _MaskedArray[np.bool], other: _ArrayLike[ScalarT], /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __sub__(self: _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __sub__(self: _MaskedArrayFloat64_co, other: _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __sub__(self: _MaskedArray[complex128], other: _ArrayLikeComplex128_co, /) -> _MaskedArray[complex128]: ...
    @overload
    def __sub__(self: _MaskedArrayComplex128_co, other: _ArrayLike[complexfloating[_64Bit]], /) -> _MaskedArray[complex128]: ...
    @overload
    def __sub__(self: _MaskedArrayUInt_co, other: _ArrayLikeUInt_co, /) -> _MaskedArray[unsignedinteger]: ...
    @overload
    def __sub__(self: _MaskedArrayInt_co, other: _ArrayLikeInt_co, /) -> _MaskedArray[signedinteger]: ...
    @overload
    def __sub__(self: _MaskedArrayFloat_co, other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __sub__(self: _MaskedArrayComplex_co, other: _ArrayLikeComplex_co, /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __sub__(self: _MaskedArray[number], other: _ArrayLikeNumber_co, /) -> _MaskedArray[number]: ...
    @overload
    def __sub__(self: _MaskedArrayTD64_co, other: _ArrayLikeTD64_co, /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __sub__(self: _MaskedArray[datetime64], other: _ArrayLikeTD64_co, /) -> _MaskedArray[datetime64]: ...
    @overload
    def __sub__(self: _MaskedArray[datetime64], other: _ArrayLikeDT64_co, /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __sub__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __sub__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__rsub__`
    @overload  # type: ignore[override]
    def __rsub__[ScalarT: np.number](
        self: _MaskedArray[ScalarT],
        other: int | np.bool,
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload
    def __rsub__[ScalarT: np.number](self: _MaskedArray[ScalarT], other: _ArrayLikeBool_co, /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __rsub__(self: _MaskedArray[np.bool], other: _ArrayLikeBool_co, /) -> NoReturn: ...
    @overload
    def __rsub__[ScalarT: np.number](self: _MaskedArray[np.bool], other: _ArrayLike[ScalarT], /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __rsub__(self: _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __rsub__(self: _MaskedArrayFloat64_co, other: _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __rsub__(self: _MaskedArray[complex128], other: _ArrayLikeComplex128_co, /) -> _MaskedArray[complex128]: ...
    @overload
    def __rsub__(self: _MaskedArrayComplex128_co, other: _ArrayLike[complexfloating[_64Bit]], /) -> _MaskedArray[complex128]: ...
    @overload
    def __rsub__(self: _MaskedArrayUInt_co, other: _ArrayLikeUInt_co, /) -> _MaskedArray[unsignedinteger]: ...
    @overload
    def __rsub__(self: _MaskedArrayInt_co, other: _ArrayLikeInt_co, /) -> _MaskedArray[signedinteger]: ...
    @overload
    def __rsub__(self: _MaskedArrayFloat_co, other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __rsub__(self: _MaskedArrayComplex_co, other: _ArrayLikeComplex_co, /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __rsub__(self: _MaskedArray[number], other: _ArrayLikeNumber_co, /) -> _MaskedArray[number]: ...
    @overload
    def __rsub__(self: _MaskedArrayTD64_co, other: _ArrayLikeTD64_co, /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __rsub__(self: _MaskedArrayTD64_co, other: _ArrayLikeDT64_co, /) -> _MaskedArray[datetime64]: ...
    @overload
    def __rsub__(self: _MaskedArray[datetime64], other: _ArrayLikeDT64_co, /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __rsub__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __rsub__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__mul__`
    @overload  # type: ignore[override]
    def __mul__[ScalarT: np.number](
        self: _MaskedArray[ScalarT],
        other: int | np.bool,
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload
    def __mul__[ScalarT: np.number](self: _MaskedArray[ScalarT], other: _ArrayLikeBool_co, /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __mul__(self: _MaskedArray[np.bool], other: _ArrayLikeBool_co, /) -> _MaskedArray[np.bool]: ...
    @overload
    def __mul__[ScalarT: np.number](self: _MaskedArray[np.bool], other: _ArrayLike[ScalarT], /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __mul__(self: _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __mul__(self: _MaskedArrayFloat64_co, other: _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __mul__(self: _MaskedArray[complex128], other: _ArrayLikeComplex128_co, /) -> _MaskedArray[complex128]: ...
    @overload
    def __mul__(self: _MaskedArrayComplex128_co, other: _ArrayLike[complexfloating[_64Bit]], /) -> _MaskedArray[complex128]: ...
    @overload
    def __mul__(self: _MaskedArrayUInt_co, other: _ArrayLikeUInt_co, /) -> _MaskedArray[unsignedinteger]: ...
    @overload
    def __mul__(self: _MaskedArrayInt_co, other: _ArrayLikeInt_co, /) -> _MaskedArray[signedinteger]: ...
    @overload
    def __mul__(self: _MaskedArrayFloat_co, other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __mul__(self: _MaskedArrayComplex_co, other: _ArrayLikeComplex_co, /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __mul__(self: _MaskedArray[number], other: _ArrayLikeNumber_co, /) -> _MaskedArray[number]: ...
    @overload
    def __mul__(self: _MaskedArray[timedelta64], other: _ArrayLikeFloat_co, /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __mul__(self: _MaskedArrayFloat_co, other: _ArrayLike[timedelta64], /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __mul__(
        self: MaskedArray[Any, dtype[np.character] | np.dtypes.StringDType],
        other: _ArrayLikeInt,
        /,
    ) -> MaskedArray[tuple[Any, ...], _DTypeT_co]: ...
    @overload
    def __mul__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __mul__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__rmul__`
    @overload  # type: ignore[override]  # signature equivalent to __mul__
    def __rmul__[ScalarT: np.number](
        self: _MaskedArray[ScalarT],
        other: int | np.bool,
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload
    def __rmul__[ScalarT: np.number](self: _MaskedArray[ScalarT], other: _ArrayLikeBool_co, /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __rmul__(self: _MaskedArray[np.bool], other: _ArrayLikeBool_co, /) -> _MaskedArray[np.bool]: ...
    @overload
    def __rmul__[ScalarT: np.number](self: _MaskedArray[np.bool], other: _ArrayLike[ScalarT], /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __rmul__(self: _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __rmul__(self: _MaskedArrayFloat64_co, other: _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __rmul__(self: _MaskedArray[complex128], other: _ArrayLikeComplex128_co, /) -> _MaskedArray[complex128]: ...
    @overload
    def __rmul__(self: _MaskedArrayComplex128_co, other: _ArrayLike[complexfloating[_64Bit]], /) -> _MaskedArray[complex128]: ...
    @overload
    def __rmul__(self: _MaskedArrayUInt_co, other: _ArrayLikeUInt_co, /) -> _MaskedArray[unsignedinteger]: ...
    @overload
    def __rmul__(self: _MaskedArrayInt_co, other: _ArrayLikeInt_co, /) -> _MaskedArray[signedinteger]: ...
    @overload
    def __rmul__(self: _MaskedArrayFloat_co, other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __rmul__(self: _MaskedArrayComplex_co, other: _ArrayLikeComplex_co, /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __rmul__(self: _MaskedArray[number], other: _ArrayLikeNumber_co, /) -> _MaskedArray[number]: ...
    @overload
    def __rmul__(self: _MaskedArray[timedelta64], other: _ArrayLikeFloat_co, /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __rmul__(self: _MaskedArrayFloat_co, other: _ArrayLike[timedelta64], /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __rmul__(
        self: MaskedArray[Any, dtype[np.character] | np.dtypes.StringDType],
        other: _ArrayLikeInt,
        /,
    ) -> MaskedArray[tuple[Any, ...], _DTypeT_co]: ...
    @overload
    def __rmul__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __rmul__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__truediv__`
    @overload  # type: ignore[override]
    def __truediv__(self: _MaskedArrayInt_co | _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __truediv__(self: _MaskedArrayFloat64_co, other: _ArrayLikeInt_co | _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __truediv__(self: _MaskedArray[complex128], other: _ArrayLikeComplex128_co, /) -> _MaskedArray[complex128]: ...
    @overload
    def __truediv__(self: _MaskedArrayComplex128_co, other: _ArrayLike[complexfloating[_64Bit]], /) -> _MaskedArray[complex128]: ...
    @overload
    def __truediv__(self: _MaskedArray[floating], other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __truediv__(self: _MaskedArrayFloat_co, other: _ArrayLike[floating], /) -> _MaskedArray[floating]: ...
    @overload
    def __truediv__(self: _MaskedArray[complexfloating], other: _ArrayLikeNumber_co, /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __truediv__(self: _MaskedArrayNumber_co, other: _ArrayLike[complexfloating], /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __truediv__(self: _MaskedArray[inexact], other: _ArrayLikeNumber_co, /) -> _MaskedArray[inexact]: ...
    @overload
    def __truediv__(self: _MaskedArray[number], other: _ArrayLikeNumber_co, /) -> _MaskedArray[number]: ...
    @overload
    def __truediv__(self: _MaskedArray[timedelta64], other: _ArrayLike[timedelta64], /) -> _MaskedArray[float64]: ...
    @overload
    def __truediv__(self: _MaskedArray[timedelta64], other: _ArrayLikeBool_co, /) -> NoReturn: ...
    @overload
    def __truediv__(self: _MaskedArray[timedelta64], other: _ArrayLikeFloat_co, /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __truediv__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __truediv__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__rtruediv__`
    @overload  # type: ignore[override]
    def __rtruediv__(self: _MaskedArrayInt_co | _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __rtruediv__(self: _MaskedArrayFloat64_co, other: _ArrayLikeInt_co | _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __rtruediv__(self: _MaskedArray[complex128], other: _ArrayLikeComplex128_co, /) -> _MaskedArray[complex128]: ...
    @overload
    def __rtruediv__(self: _MaskedArrayComplex128_co, other: _ArrayLike[complexfloating[_64Bit]], /) -> _MaskedArray[complex128]: ...
    @overload
    def __rtruediv__(self: _MaskedArray[floating], other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __rtruediv__(self: _MaskedArrayFloat_co, other: _ArrayLike[floating], /) -> _MaskedArray[floating]: ...
    @overload
    def __rtruediv__(self: _MaskedArray[complexfloating], other: _ArrayLikeNumber_co, /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __rtruediv__(self: _MaskedArrayNumber_co, other: _ArrayLike[complexfloating], /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __rtruediv__(self: _MaskedArray[inexact], other: _ArrayLikeNumber_co, /) -> _MaskedArray[inexact]: ...
    @overload
    def __rtruediv__(self: _MaskedArray[number], other: _ArrayLikeNumber_co, /) -> _MaskedArray[number]: ...
    @overload
    def __rtruediv__(self: _MaskedArray[timedelta64], other: _ArrayLike[timedelta64], /) -> _MaskedArray[float64]: ...
    @overload
    def __rtruediv__(self: _MaskedArray[integer | floating], other: _ArrayLike[timedelta64], /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __rtruediv__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __rtruediv__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__floordiv__`
    @overload  # type: ignore[override]
    def __floordiv__[ScalarT: _RealNumber](
        self: _MaskedArray[ScalarT],
        other: int | np.bool,
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload
    def __floordiv__[ScalarT: _RealNumber](self: _MaskedArray[ScalarT], other: _ArrayLikeBool_co, /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __floordiv__(self: _MaskedArray[np.bool], other: _ArrayLikeBool_co, /) -> _MaskedArray[int8]: ...
    @overload
    def __floordiv__[ScalarT: _RealNumber](
        self: _MaskedArray[np.bool],
        other: _ArrayLike[ScalarT],
        /,
    ) -> _MaskedArray[ScalarT]: ...
    @overload
    def __floordiv__(self: _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __floordiv__(self: _MaskedArrayFloat64_co, other: _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __floordiv__(self: _MaskedArrayUInt_co, other: _ArrayLikeUInt_co, /) -> _MaskedArray[unsignedinteger]: ...
    @overload
    def __floordiv__(self: _MaskedArrayInt_co, other: _ArrayLikeInt_co, /) -> _MaskedArray[signedinteger]: ...
    @overload
    def __floordiv__(self: _MaskedArrayFloat_co, other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __floordiv__(self: _MaskedArray[timedelta64], other: _ArrayLike[timedelta64], /) -> _MaskedArray[int64]: ...
    @overload
    def __floordiv__(self: _MaskedArray[timedelta64], other: _ArrayLikeBool_co, /) -> NoReturn: ...
    @overload
    def __floordiv__(self: _MaskedArray[timedelta64], other: _ArrayLikeFloat_co, /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __floordiv__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __floordiv__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__rfloordiv__`
    @overload  # type: ignore[override]
    def __rfloordiv__[ScalarT: _RealNumber](
        self: _MaskedArray[ScalarT],
        other: int | np.bool,
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload
    def __rfloordiv__[ScalarT: _RealNumber](
        self: _MaskedArray[ScalarT],
        other: _ArrayLikeBool_co,
        /,
    ) -> _MaskedArray[ScalarT]: ...
    @overload
    def __rfloordiv__(self: _MaskedArray[np.bool], other: _ArrayLikeBool_co, /) -> _MaskedArray[int8]: ...
    @overload
    def __rfloordiv__[ScalarT: _RealNumber](
        self: _MaskedArray[np.bool],
        other: _ArrayLike[ScalarT],
        /,
    ) -> _MaskedArray[ScalarT]: ...
    @overload
    def __rfloordiv__(self: _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __rfloordiv__(self: _MaskedArrayFloat64_co, other: _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __rfloordiv__(self: _MaskedArrayUInt_co, other: _ArrayLikeUInt_co, /) -> _MaskedArray[unsignedinteger]: ...
    @overload
    def __rfloordiv__(self: _MaskedArrayInt_co, other: _ArrayLikeInt_co, /) -> _MaskedArray[signedinteger]: ...
    @overload
    def __rfloordiv__(self: _MaskedArrayFloat_co, other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __rfloordiv__(self: _MaskedArray[timedelta64], other: _ArrayLike[timedelta64], /) -> _MaskedArray[int64]: ...
    @overload
    def __rfloordiv__(self: _MaskedArray[floating | integer], other: _ArrayLike[timedelta64], /) -> _MaskedArray[timedelta64]: ...
    @overload
    def __rfloordiv__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __rfloordiv__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__pow__` (minus the `mod` parameter)
    @overload  # type: ignore[override]
    def __pow__[ScalarT: np.number](
        self: _MaskedArray[ScalarT],
        other: int | np.bool,
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload
    def __pow__[ScalarT: np.number](self: _MaskedArray[ScalarT], other: _ArrayLikeBool_co, /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __pow__(self: _MaskedArray[np.bool], other: _ArrayLikeBool_co, /) -> _MaskedArray[int8]: ...
    @overload
    def __pow__[ScalarT: np.number](self: _MaskedArray[np.bool], other: _ArrayLike[ScalarT], /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __pow__(self: _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __pow__(self: _MaskedArrayFloat64_co, other: _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __pow__(self: _MaskedArray[complex128], other: _ArrayLikeComplex128_co, /) -> _MaskedArray[complex128]: ...
    @overload
    def __pow__(self: _MaskedArrayComplex128_co, other: _ArrayLike[complexfloating[_64Bit]], /) -> _MaskedArray[complex128]: ...
    @overload
    def __pow__(self: _MaskedArrayUInt_co, other: _ArrayLikeUInt_co, /) -> _MaskedArray[unsignedinteger]: ...
    @overload
    def __pow__(self: _MaskedArrayInt_co, other: _ArrayLikeInt_co, /) -> _MaskedArray[signedinteger]: ...
    @overload
    def __pow__(self: _MaskedArrayFloat_co, other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __pow__(self: _MaskedArrayComplex_co, other: _ArrayLikeComplex_co, /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __pow__(self: _MaskedArray[number], other: _ArrayLikeNumber_co, /) -> _MaskedArray[number]: ...
    @overload
    def __pow__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __pow__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    # Keep in sync with `ndarray.__rpow__` (minus the `mod` parameter)
    @overload  # type: ignore[override]
    def __rpow__[ScalarT: np.number](
        self: _MaskedArray[ScalarT],
        other: int | np.bool,
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    @overload
    def __rpow__[ScalarT: np.number](self: _MaskedArray[ScalarT], other: _ArrayLikeBool_co, /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __rpow__(self: _MaskedArray[np.bool], other: _ArrayLikeBool_co, /) -> _MaskedArray[int8]: ...
    @overload
    def __rpow__[ScalarT: np.number](self: _MaskedArray[np.bool], other: _ArrayLike[ScalarT], /) -> _MaskedArray[ScalarT]: ...
    @overload
    def __rpow__(self: _MaskedArray[float64], other: _ArrayLikeFloat64_co, /) -> _MaskedArray[float64]: ...
    @overload
    def __rpow__(self: _MaskedArrayFloat64_co, other: _ArrayLike[floating[_64Bit]], /) -> _MaskedArray[float64]: ...
    @overload
    def __rpow__(self: _MaskedArray[complex128], other: _ArrayLikeComplex128_co, /) -> _MaskedArray[complex128]: ...
    @overload
    def __rpow__(self: _MaskedArrayComplex128_co, other: _ArrayLike[complexfloating[_64Bit]], /) -> _MaskedArray[complex128]: ...
    @overload
    def __rpow__(self: _MaskedArrayUInt_co, other: _ArrayLikeUInt_co, /) -> _MaskedArray[unsignedinteger]: ...
    @overload
    def __rpow__(self: _MaskedArrayInt_co, other: _ArrayLikeInt_co, /) -> _MaskedArray[signedinteger]: ...
    @overload
    def __rpow__(self: _MaskedArrayFloat_co, other: _ArrayLikeFloat_co, /) -> _MaskedArray[floating]: ...
    @overload
    def __rpow__(self: _MaskedArrayComplex_co, other: _ArrayLikeComplex_co, /) -> _MaskedArray[complexfloating]: ...
    @overload
    def __rpow__(self: _MaskedArray[number], other: _ArrayLikeNumber_co, /) -> _MaskedArray[number]: ...
    @overload
    def __rpow__(self: _MaskedArray[object_], other: Any, /) -> Any: ...
    @overload
    def __rpow__(self: _MaskedArray[Any], other: _ArrayLikeObject_co, /) -> Any: ...

    #
    @property  # type: ignore[misc]
    def imag[ScalarT: np.generic](  # type: ignore[override]
        self: _HasDTypeWithRealAndImag[object, ScalarT],
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    def get_imag[ScalarT: np.generic](
        self: _HasDTypeWithRealAndImag[object, ScalarT],
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...

    #
    @property  # type: ignore[misc]
    def real[ScalarT: np.generic](  # type: ignore[override]
        self: _HasDTypeWithRealAndImag[ScalarT, object],
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...
    def get_real[ScalarT: np.generic](
        self: _HasDTypeWithRealAndImag[ScalarT, object],
        /,
    ) -> MaskedArray[_ShapeT_co, dtype[ScalarT]]: ...

    # keep in sync with `np.ma.count`
    @overload
    def count(self, axis: None = None, keepdims: Literal[False] | _NoValueType = ...) -> int: ...
    @overload
    def count(self, axis: _ShapeLike, keepdims: bool | _NoValueType = ...) -> NDArray[int_]: ...
    @overload
    def count(self, axis: _ShapeLike | None = None, *, keepdims: Literal[True]) -> NDArray[int_]: ...
    @overload
    def count(self, axis: _ShapeLike | None, keepdims: Literal[True]) -> NDArray[int_]: ...

    # Keep in sync with `ndarray.reshape`
    # NOTE: reshape also accepts negative integers, so we can't use integer literals
    @overload  # (None)
    def reshape(self, shape: None, /, *, order: _OrderACF = "C", copy: bool | None = None) -> Self: ...
    @overload  # (empty_sequence)
    def reshape(
        self,
        shape: Sequence[Never],
        /,
        *,
        order: _OrderACF = "C",
        copy: bool | None = None,
    ) -> MaskedArray[tuple[()], _DTypeT_co]: ...
    @overload  # (() | (int) | (int, int) | ....)  # up to 8-d
    def reshape[ShapeT: _Shape](
        self,
        shape: ShapeT,
        /,
        *,
        order: _OrderACF = "C",
        copy: bool | None = None,
    ) -> MaskedArray[ShapeT, _DTypeT_co]: ...
    @overload  # (index)
    def reshape(
        self,
        size1: SupportsIndex,
        /,
        *,
        order: _OrderACF = "C",
        copy: bool | None = None,
    ) -> MaskedArray[tuple[int], _DTypeT_co]: ...
    @overload  # (index, index)
    def reshape(
        self,
        size1: SupportsIndex,
        size2: SupportsIndex,
        /,
        *,
        order: _OrderACF = "C",
        copy: bool | None = None,
    ) -> MaskedArray[tuple[int, int], _DTypeT_co]: ...
    @overload  # (index, index, index)
    def reshape(
        self,
        size1: SupportsIndex,
        size2: SupportsIndex,
        size3: SupportsIndex,
        /,
        *,
        order: _OrderACF = "C",
        copy: bool | None = None,
    ) -> MaskedArray[tuple[int, int, int], _DTypeT_co]: ...
    @overload  # (index, index, index, index)
    def reshape(
        self,
        size1: SupportsIndex,
        size2: SupportsIndex,
        size3: SupportsIndex,
        size4: SupportsIndex,
        /,
        *,
        order: _OrderACF = "C",
        copy: bool | None = None,
    ) -> MaskedArray[tuple[int, int, int, int], _DTypeT_co]: ...
    @overload  # (int, *(index, ...))
    def reshape(
        self,
        size0: SupportsIndex,
        /,
        *shape: SupportsIndex,
        order: _OrderACF = "C",
        copy: bool | None = None,
    ) -> MaskedArray[_AnyShape, _DTypeT_co]: ...
    @overload  # (sequence[index])
    def reshape(
        self,
        shape: Sequence[SupportsIndex],
        /,
        *,
        order: _OrderACF = "C",
        copy: bool | None = None,
    ) -> MaskedArray[_AnyShape, _DTypeT_co]: ...

    def resize(self, newshape: Never, refcheck: bool = True, order: bool = False) -> NoReturn: ...  # type: ignore[override]
    def put(self, indices: _ArrayLikeInt_co, values: ArrayLike, mode: _ModeKind = "raise") -> None: ...
    def ids(self) -> tuple[int, int]: ...
    def iscontiguous(self) -> bool: ...

    # Keep in sync with `ma.core.all`
    @overload  # type: ignore[override]
    def all(
        self,
        axis: None = None,
        out: None = None,
        keepdims: Literal[False] | _NoValueType = ...,
    ) -> bool_: ...
    @overload
    def all(
        self,
        axis: _ShapeLike | None = None,
        out: None = None,
        *,
        keepdims: Literal[True],
    ) -> _MaskedArray[bool_]: ...
    @overload
    def all(
        self,
        axis: _ShapeLike | None,
        out: None,
        keepdims: Literal[True],
    ) -> _MaskedArray[bool_]: ...
    @overload
    def all(
        self,
        axis: _ShapeLike | None = None,
        out: None = None,
        keepdims: bool | _NoValueType = ...,
    ) -> bool_ | _MaskedArray[bool_]: ...
    @overload
    def all[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None = None,
        *,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def all[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...

    # Keep in sync with `ma.core.any`
    @overload  # type: ignore[override]
    def any(
        self,
        axis: None = None,
        out: None = None,
        keepdims: Literal[False] | _NoValueType = ...,
    ) -> bool_: ...
    @overload
    def any(
        self,
        axis: _ShapeLike | None = None,
        out: None = None,
        *,
        keepdims: Literal[True],
    ) -> _MaskedArray[bool_]: ...
    @overload
    def any(
        self,
        axis: _ShapeLike | None,
        out: None,
        keepdims: Literal[True],
    ) -> _MaskedArray[bool_]: ...
    @overload
    def any(
        self,
        axis: _ShapeLike | None = None,
        out: None = None,
        keepdims: bool | _NoValueType = ...,
    ) -> bool_ | _MaskedArray[bool_]: ...
    @overload
    def any[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None = None,
        *,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def any[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...

    # Keep in sync with `ndarray.trace` and `ma.core.trace`
    @overload
    def trace(
        self,  # >= 2D MaskedArray
        offset: SupportsIndex = 0,
        axis1: SupportsIndex = 0,
        axis2: SupportsIndex = 1,
        dtype: DTypeLike | None = None,
        out: None = None,
    ) -> Any: ...
    @overload
    def trace[ArrayT: np.ndarray](
        self,  # >= 2D MaskedArray
        offset: SupportsIndex = 0,
        axis1: SupportsIndex = 0,
        axis2: SupportsIndex = 1,
        dtype: DTypeLike | None = None,
        *,
        out: ArrayT,
    ) -> ArrayT: ...
    @overload
    def trace[ArrayT: np.ndarray](
        self,  # >= 2D MaskedArray
        offset: SupportsIndex,
        axis1: SupportsIndex,
        axis2: SupportsIndex,
        dtype: DTypeLike | None,
        out: ArrayT,
    ) -> ArrayT: ...

    # This differs from `ndarray.dot`, in that 1D dot 1D returns a 0D array.
    @overload
    def dot(self, b: ArrayLike, out: None = None, strict: bool = False) -> _MaskedArray[Any]: ...
    @overload
    def dot[ArrayT: np.ndarray](self, b: ArrayLike, out: ArrayT, strict: bool = False) -> ArrayT: ...

    # Keep in sync with `ma.core.sum`
    @overload  # type: ignore[override]
    def sum(
        self,
        /,
        axis: _ShapeLike | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool | _NoValueType = ...,
    ) -> Any: ...
    @overload
    def sum[ArrayT: np.ndarray](
        self,
        /,
        axis: _ShapeLike | None,
        dtype: DTypeLike | None,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def sum[ArrayT: np.ndarray](
        self,
        /,
        axis: _ShapeLike | None = None,
        dtype: DTypeLike | None = None,
        *,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...

    # Keep in sync with `ndarray.cumsum` and `ma.core.cumsum`
    @overload  # out: None (default)
    def cumsum(self, /, axis: SupportsIndex | None = None, dtype: DTypeLike | None = None, out: None = None) -> MaskedArray: ...
    @overload  # out: ndarray
    def cumsum[ArrayT: np.ndarray](self, /, axis: SupportsIndex | None, dtype: DTypeLike | None, out: ArrayT) -> ArrayT: ...
    @overload
    def cumsum[ArrayT: np.ndarray](
        self,
        /,
        axis: SupportsIndex | None = None,
        dtype: DTypeLike | None = None,
        *,
        out: ArrayT,
    ) -> ArrayT: ...

    # Keep in sync with `ma.core.prod`
    @overload  # type: ignore[override]
    def prod(
        self,
        /,
        axis: _ShapeLike | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool | _NoValueType = ...,
    ) -> Any: ...
    @overload
    def prod[ArrayT: np.ndarray](
        self,
        /,
        axis: _ShapeLike | None,
        dtype: DTypeLike | None,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def prod[ArrayT: np.ndarray](
        self,
        /,
        axis: _ShapeLike | None = None,
        dtype: DTypeLike | None = None,
        *,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...

    product = prod

    # Keep in sync with `ndarray.cumprod` and `ma.core.cumprod`
    @overload  # out: None (default)
    def cumprod(self, /, axis: SupportsIndex | None = None, dtype: DTypeLike | None = None, out: None = None) -> MaskedArray: ...
    @overload  # out: ndarray
    def cumprod[ArrayT: np.ndarray](self, /, axis: SupportsIndex | None, dtype: DTypeLike | None, out: ArrayT) -> ArrayT: ...
    @overload
    def cumprod[ArrayT: np.ndarray](
        self,
        /,
        axis: SupportsIndex | None = None,
        dtype: DTypeLike | None = None,
        *,
        out: ArrayT,
    ) -> ArrayT: ...

    # Keep in sync with `ma.core.mean`
    @overload  # type: ignore[override]
    def mean(
        self,
        axis: _ShapeLike | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: bool | _NoValueType = ...,
    ) -> Any: ...
    @overload
    def mean[ArrayT: np.ndarray](
        self,
        /,
        axis: _ShapeLike | None,
        dtype: DTypeLike | None,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def mean[ArrayT: np.ndarray](
        self,
        /,
        axis: _ShapeLike | None = None,
        dtype: DTypeLike | None = None,
        *,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...

    # keep roughly in sync with `ma.core.anom`
    @overload
    def anom(self, axis: SupportsIndex | None = None, dtype: None = None) -> Self: ...
    @overload
    def anom(self, axis: SupportsIndex | None = None, *, dtype: DTypeLike) -> MaskedArray[_ShapeT_co, dtype]: ...
    @overload
    def anom(self, axis: SupportsIndex | None, dtype: DTypeLike) -> MaskedArray[_ShapeT_co, dtype]: ...

    # keep in sync with `std` and `ma.core.var`
    @overload  # type: ignore[override]
    def var(
        self,
        axis: _ShapeLike | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        ddof: float = 0,
        keepdims: bool | _NoValueType = ...,
        mean: _ArrayLikeNumber_co | _NoValueType = ...,
    ) -> Any: ...
    @overload
    def var[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None,
        dtype: DTypeLike | None,
        out: ArrayT,
        ddof: float = 0,
        keepdims: bool | _NoValueType = ...,
        mean: _ArrayLikeNumber_co | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def var[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None = None,
        dtype: DTypeLike | None = None,
        *,
        out: ArrayT,
        ddof: float = 0,
        keepdims: bool | _NoValueType = ...,
        mean: _ArrayLikeNumber_co | _NoValueType = ...,
    ) -> ArrayT: ...

    # keep in sync with `var` and `ma.core.std`
    @overload  # type: ignore[override]
    def std(
        self,
        axis: _ShapeLike | None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        ddof: float = 0,
        keepdims: bool | _NoValueType = ...,
        mean: _ArrayLikeNumber_co | _NoValueType = ...,
    ) -> Any: ...
    @overload
    def std[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None,
        dtype: DTypeLike | None,
        out: ArrayT,
        ddof: float = 0,
        keepdims: bool | _NoValueType = ...,
        mean: _ArrayLikeNumber_co | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def std[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None = None,
        dtype: DTypeLike | None = None,
        *,
        out: ArrayT,
        ddof: float = 0,
        keepdims: bool | _NoValueType = ...,
        mean: _ArrayLikeNumber_co | _NoValueType = ...,
    ) -> ArrayT: ...

    # Keep in sync with `ndarray.round`
    @overload  # out=None (default)
    def round(self, /, decimals: SupportsIndex = 0, out: None = None) -> Self: ...
    @overload  # out=ndarray
    def round[ArrayT: np.ndarray](self, /, decimals: SupportsIndex, out: ArrayT) -> ArrayT: ...
    @overload
    def round[ArrayT: np.ndarray](self, /, decimals: SupportsIndex = 0, *, out: ArrayT) -> ArrayT: ...

    def argsort(  # type: ignore[override]
        self,
        axis: SupportsIndex | _NoValueType = ...,
        kind: _SortKind | None = None,
        order: str | Sequence[str] | None = None,
        endwith: bool = True,
        fill_value: _ScalarLike_co | None = None,
        *,
        stable: bool = False,
    ) -> _MaskedArray[intp]: ...

    # Keep in-sync with np.ma.argmin
    @overload  # type: ignore[override]
    def argmin(
        self,
        axis: None = None,
        fill_value: _ScalarLike_co | None = None,
        out: None = None,
        *,
        keepdims: Literal[False] | _NoValueType = ...,
    ) -> intp: ...
    @overload
    def argmin(
        self,
        axis: SupportsIndex | None = None,
        fill_value: _ScalarLike_co | None = None,
        out: None = None,
        *,
        keepdims: bool | _NoValueType = ...,
    ) -> Any: ...
    @overload
    def argmin[ArrayT: np.ndarray](
        self,
        axis: SupportsIndex | None = None,
        fill_value: _ScalarLike_co | None = None,
        *,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def argmin[ArrayT: np.ndarray](
        self,
        axis: SupportsIndex | None,
        fill_value: _ScalarLike_co | None,
        out: ArrayT,
        *,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...

    # Keep in-sync with np.ma.argmax
    @overload  # type: ignore[override]
    def argmax(
        self,
        axis: None = None,
        fill_value: _ScalarLike_co | None = None,
        out: None = None,
        *,
        keepdims: Literal[False] | _NoValueType = ...,
    ) -> intp: ...
    @overload
    def argmax(
        self,
        axis: SupportsIndex | None = None,
        fill_value: _ScalarLike_co | None = None,
        out: None = None,
        *,
        keepdims: bool | _NoValueType = ...,
    ) -> Any: ...
    @overload
    def argmax[ArrayT: np.ndarray](
        self,
        axis: SupportsIndex | None = None,
        fill_value: _ScalarLike_co | None = None,
        *,
        out: ArrayT,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def argmax[ArrayT: np.ndarray](
        self,
        axis: SupportsIndex | None,
        fill_value: _ScalarLike_co | None,
        out: ArrayT,
        *,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...

    #
    def sort(  # type: ignore[override]
        self,
        axis: SupportsIndex = -1,
        kind: _SortKind | None = None,
        order: str | Sequence[str] | None = None,
        endwith: bool | None = True,
        fill_value: _ScalarLike_co | None = None,
        *,
        stable: Literal[False] | None = False,
    ) -> None: ...

    #
    @overload  # type: ignore[override]
    def min[ScalarT: np.generic](
        self: _MaskedArray[ScalarT],
        axis: None = None,
        out: None = None,
        fill_value: _ScalarLike_co | None = None,
        keepdims: Literal[False] | _NoValueType = ...,
    ) -> ScalarT: ...
    @overload
    def min(
        self,
        axis: _ShapeLike | None = None,
        out: None = None,
        fill_value: _ScalarLike_co | None = None,
        keepdims: bool | _NoValueType = ...
    ) -> Any: ...
    @overload
    def min[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None,
        out: ArrayT,
        fill_value: _ScalarLike_co | None = None,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def min[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None = None,
        *,
        out: ArrayT,
        fill_value: _ScalarLike_co | None = None,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...

    #
    @overload  # type: ignore[override]
    def max[ScalarT: np.generic](
        self: _MaskedArray[ScalarT],
        axis: None = None,
        out: None = None,
        fill_value: _ScalarLike_co | None = None,
        keepdims: Literal[False] | _NoValueType = ...,
    ) -> ScalarT: ...
    @overload
    def max(
        self,
        axis: _ShapeLike | None = None,
        out: None = None,
        fill_value: _ScalarLike_co | None = None,
        keepdims: bool | _NoValueType = ...
    ) -> Any: ...
    @overload
    def max[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None,
        out: ArrayT,
        fill_value: _ScalarLike_co | None = None,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...
    @overload
    def max[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None = None,
        *,
        out: ArrayT,
        fill_value: _ScalarLike_co | None = None,
        keepdims: bool | _NoValueType = ...,
    ) -> ArrayT: ...

    #
    @overload
    def ptp[ScalarT: np.generic](
        self: _MaskedArray[ScalarT],
        axis: None = None,
        out: None = None,
        fill_value: _ScalarLike_co | None = None,
        keepdims: Literal[False] = False,
    ) -> ScalarT: ...
    @overload
    def ptp(
        self,
        axis: _ShapeLike | None = None,
        out: None = None,
        fill_value: _ScalarLike_co | None = None,
        keepdims: bool = False,
    ) -> Any: ...
    @overload
    def ptp[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None,
        out: ArrayT,
        fill_value: _ScalarLike_co | None = None,
        keepdims: bool = False,
    ) -> ArrayT: ...
    @overload
    def ptp[ArrayT: np.ndarray](
        self,
        axis: _ShapeLike | None = None,
        *,
        out: ArrayT,
        fill_value: _ScalarLike_co | None = None,
        keepdims: bool = False,
    ) -> ArrayT: ...

    #
    @overload
    def partition(
        self,
        /,
        kth: _ArrayLikeInt,
        axis: SupportsIndex = -1,
        kind: _PartitionKind = "introselect",
        order: None = None
    ) -> None: ...
    @overload
    def partition(
        self: _MaskedArray[np.void],
        /,
        kth: _ArrayLikeInt,
        axis: SupportsIndex = -1,
        kind: _PartitionKind = "introselect",
        order: str | Sequence[str] | None = None,
    ) -> None: ...

    #
    @overload
    def argpartition(
        self,
        /,
        kth: _ArrayLikeInt,
        axis: SupportsIndex | None = -1,
        kind: _PartitionKind = "introselect",
        order: None = None,
    ) -> _MaskedArray[intp]: ...
    @overload
    def argpartition(
        self: _MaskedArray[np.void],
        /,
        kth: _ArrayLikeInt,
        axis: SupportsIndex | None = -1,
        kind: _PartitionKind = "introselect",
        order: str | Sequence[str] | None = None,
    ) -> _MaskedArray[intp]: ...

    # Keep in-sync with np.ma.take
    @overload  # type: ignore[override]
    def take[ScalarT: np.generic](
        self: _MaskedArray[ScalarT],
        indices: _IntLike_co,
        axis: None = None,
        out: None = None,
        mode: _ModeKind = "raise"
    ) -> ScalarT: ...
    @overload
    def take[ScalarT: np.generic](
        self: _MaskedArray[ScalarT],
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex | None = None,
        out: None = None,
        mode: _ModeKind = "raise",
    ) -> _MaskedArray[ScalarT]: ...
    @overload
    def take[ArrayT: np.ndarray](
        self,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex | None,
        out: ArrayT,
        mode: _ModeKind = "raise",
    ) -> ArrayT: ...
    @overload
    def take[ArrayT: np.ndarray](
        self,
        indices: _ArrayLikeInt_co,
        axis: SupportsIndex | None = None,
        *,
        out: ArrayT,
        mode: _ModeKind = "raise",
    ) -> ArrayT: ...

    # keep in sync with `ndarray.diagonal`
    @override
    def diagonal(
        self,
        /,
        offset: SupportsIndex = 0,
        axis1: SupportsIndex = 0,
        axis2: SupportsIndex = 1,
    ) -> MaskedArray[_AnyShape, _DTypeT_co]: ...

    # keep in sync with `ndarray.repeat`
    @override
    @overload
    def repeat(
        self,
        /,
        repeats: _ArrayLikeInt_co,
        axis: None = None,
    ) -> MaskedArray[tuple[int], _DTypeT_co]: ...
    @overload
    def repeat(
        self,
        /,
        repeats: _ArrayLikeInt_co,
        axis: SupportsIndex,
    ) -> MaskedArray[_AnyShape, _DTypeT_co]: ...

    # keep in sync with `ndarray.flatten` and `ndarray.ravel`
    @override
    def flatten(self, /, order: _OrderKACF = "C") -> MaskedArray[tuple[int], _DTypeT_co]: ...
    @override
    def ravel(self, order: _OrderKACF = "C") -> MaskedArray[tuple[int], _DTypeT_co]: ...

    # keep in sync with `ndarray.squeeze`
    @override
    def squeeze(
        self,
        /,
        axis: SupportsIndex | tuple[SupportsIndex, ...] | None = None,
    ) -> MaskedArray[_AnyShape, _DTypeT_co]: ...

    #
    def toflex(self) -> MaskedArray[_ShapeT_co, np.dtype[np.void]]: ...
    def torecords(self) -> MaskedArray[_ShapeT_co, np.dtype[np.void]]: ...

    #
    @override
    def tobytes(self, /, fill_value: Incomplete | None = None, order: _OrderKACF = "C") -> bytes: ...  # type: ignore[override]

    # keep in sync with `ndarray.tolist`
    @override
    @overload
    def tolist[T](self: MaskedArray[tuple[Never], dtype[generic[T]]], /, fill_value: _ScalarLike_co | None = None) -> Any: ...
    @overload
    def tolist[T](self: MaskedArray[tuple[()], dtype[generic[T]]], /, fill_value: _ScalarLike_co | None = None) -> T: ...
    @overload
    def tolist[T](self: _Masked1D[np.generic[T]], /, fill_value: _ScalarLike_co | None = None) -> list[T]: ...
    @overload
    def tolist[T](
        self: MaskedArray[tuple[int, int], dtype[generic[T]]],
        /,
        fill_value: _ScalarLike_co | None = None,
    ) -> list[list[T]]: ...
    @overload
    def tolist[T](
        self: MaskedArray[tuple[int, int, int], dtype[generic[T]]],
        /,
        fill_value: _ScalarLike_co | None = None,
    ) -> list[list[list[T]]]: ...
    @overload
    def tolist(self, /, fill_value: _ScalarLike_co | None = None) -> Any: ...

    # NOTE: will raise `NotImplementedError`
    @override
    def tofile(self, /, fid: Never, sep: str = "", format: str = "%s") -> NoReturn: ...  # type: ignore[override]

    #
    @override
    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Self: ...

    # Keep `dtype` at the bottom to avoid name conflicts with `np.dtype`
    @property
    def dtype(self) -> _DTypeT_co: ...
    @dtype.setter
    def dtype[DTypeT: np.dtype](self: MaskedArray[_AnyShape, DTypeT], dtype: DTypeT, /) -> None: ...

class mvoid(MaskedArray[_ShapeT_co, _DTypeT_co]):
    def __new__(
        self,  # pyright: ignore[reportSelfClsParameterName]
        data,
        mask=...,
        dtype=...,
        fill_value=...,
        hardmask=...,
        copy=...,
        subok=...,
    ): ...
    def __getitem__(self, indx): ...
    def __setitem__(self, indx, value): ...
    def __iter__(self): ...
    def __len__(self): ...
    def filled(self, fill_value=None): ...
    def tolist(self): ...  # type: ignore[override]

def isMaskedArray(x: object) -> TypeIs[MaskedArray]: ...
def isarray(x: object) -> TypeIs[MaskedArray]: ...  # alias to isMaskedArray
def isMA(x: object) -> TypeIs[MaskedArray]: ...  # alias to isMaskedArray

# 0D float64 array
class MaskedConstant(MaskedArray[tuple[()], dtype[float64]]):
    def __new__(cls) -> Self: ...

    # these overrides are no-ops
    @override
    def __iadd__(self, other: _Ignored, /) -> Self: ...  # type: ignore[override]
    @override
    def __isub__(self, other: _Ignored, /) -> Self: ...  # type: ignore[override]
    @override
    def __imul__(self, other: _Ignored, /) -> Self: ...  # type: ignore[override]
    @override
    def __ifloordiv__(self, other: _Ignored, /) -> Self: ...
    @override
    def __itruediv__(self, other: _Ignored, /) -> Self: ...  # type: ignore[override]
    @override
    def __ipow__(self, other: _Ignored, /) -> Self: ...  # type: ignore[override]
    @override
    def __deepcopy__(self, /, memo: _Ignored) -> Self: ...  # type: ignore[override]
    @override
    def copy(self, /, *args: _Ignored, **kwargs: _Ignored) -> Self: ...

masked: Final[MaskedConstant] = ...
masked_singleton: Final[MaskedConstant] = ...

type masked_array = MaskedArray

# keep in sync with `MaskedArray.__new__`
@overload
def array[ScalarT: np.generic](
    data: _ArrayLike[ScalarT],
    dtype: None = None,
    copy: bool = False,
    order: _OrderKACF | None = None,
    mask: _ArrayLikeBool_co = nomask,
    fill_value: _ScalarLike_co | None = None,
    keep_mask: bool = True,
    hard_mask: bool = False,
    shrink: bool = True,
    subok: bool = True,
    ndmin: int = 0,
) -> _MaskedArray[ScalarT]: ...
@overload
def array[ScalarT: np.generic](
    data: object,
    dtype: _DTypeLike[ScalarT],
    copy: bool = False,
    order: _OrderKACF | None = None,
    mask: _ArrayLikeBool_co = nomask,
    fill_value: _ScalarLike_co | None = None,
    keep_mask: bool = True,
    hard_mask: bool = False,
    shrink: bool = True,
    subok: bool = True,
    ndmin: int = 0,
) -> _MaskedArray[ScalarT]: ...
@overload
def array[ScalarT: np.generic](
    data: object,
    dtype: DTypeLike | None = None,
    copy: bool = False,
    order: _OrderKACF | None = None,
    mask: _ArrayLikeBool_co = nomask,
    fill_value: _ScalarLike_co | None = None,
    keep_mask: bool = True,
    hard_mask: bool = False,
    shrink: bool = True,
    subok: bool = True,
    ndmin: int = 0,
) -> _MaskedArray[ScalarT]: ...

# keep in sync with `array`
@overload
def asarray[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    dtype: None = None,
    order: _OrderKACF | None = None,
) -> _MaskedArray[ScalarT]: ...
@overload
def asarray[ScalarT: np.generic](
    a: object,
    dtype: _DTypeLike[ScalarT],
    order: _OrderKACF | None = None,
) -> _MaskedArray[ScalarT]: ...
@overload
def asarray[ScalarT: np.generic](
    a: object,
    dtype: DTypeLike | None = None,
    order: _OrderKACF | None = None,
) -> _MaskedArray[ScalarT]: ...

# keep in sync with `asarray` (but note the additional first overload)
@overload
def asanyarray[MArrayT: MaskedArray](a: MArrayT, dtype: None = None, order: _OrderKACF | None = None) -> MArrayT: ...
@overload
def asanyarray[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    dtype: None = None,
    order: _OrderKACF | None = None,
) -> _MaskedArray[ScalarT]: ...
@overload
def asanyarray[ScalarT: np.generic](
    a: object,
    dtype: _DTypeLike[ScalarT],
    order: _OrderKACF | None = None,
) -> _MaskedArray[ScalarT]: ...
@overload
def asanyarray[ScalarT: np.generic](
    a: object,
    dtype: DTypeLike | None = None,
    order: _OrderKACF | None = None,
) -> _MaskedArray[ScalarT]: ...

#
def is_masked(x: object) -> bool: ...

@overload
def min[ScalarT: np.generic](
    obj: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
    fill_value: _ScalarLike_co | None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> ScalarT: ...
@overload
def min(
    obj: ArrayLike,
    axis: _ShapeLike | None = None,
    out: None = None,
    fill_value: _ScalarLike_co | None = None,
    keepdims: bool | _NoValueType = ...
) -> Any: ...
@overload
def min[ArrayT: np.ndarray](
    obj: ArrayLike,
    axis: _ShapeLike | None,
    out: ArrayT,
    fill_value: _ScalarLike_co | None = None,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def min[ArrayT: np.ndarray](
    obj: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    out: ArrayT,
    fill_value: _ScalarLike_co | None = None,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

@overload
def max[ScalarT: np.generic](
    obj: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
    fill_value: _ScalarLike_co | None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> ScalarT: ...
@overload
def max(
    obj: ArrayLike,
    axis: _ShapeLike | None = None,
    out: None = None,
    fill_value: _ScalarLike_co | None = None,
    keepdims: bool | _NoValueType = ...
) -> Any: ...
@overload
def max[ArrayT: np.ndarray](
    obj: ArrayLike,
    axis: _ShapeLike | None,
    out: ArrayT,
    fill_value: _ScalarLike_co | None = None,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def max[ArrayT: np.ndarray](
    obj: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    out: ArrayT,
    fill_value: _ScalarLike_co | None = None,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

@overload
def ptp[ScalarT: np.generic](
    obj: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
    fill_value: _ScalarLike_co | None = None,
    keepdims: Literal[False] | _NoValueType = ...,
) -> ScalarT: ...
@overload
def ptp(
    obj: ArrayLike,
    axis: _ShapeLike | None = None,
    out: None = None,
    fill_value: _ScalarLike_co | None = None,
    keepdims: bool | _NoValueType = ...
) -> Any: ...
@overload
def ptp[ArrayT: np.ndarray](
    obj: ArrayLike,
    axis: _ShapeLike | None,
    out: ArrayT,
    fill_value: _ScalarLike_co | None = None,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def ptp[ArrayT: np.ndarray](
    obj: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    out: ArrayT,
    fill_value: _ScalarLike_co | None = None,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

# we cannot meaningfully annotate `frommethod` further, because the callable signature
# of the return type fully depends on the *value* of `methodname` and `reversed` in
# a way that cannot be expressed in the Python type system.
def _frommethod(methodname: str, reversed: bool = False) -> types.FunctionType: ...

# NOTE: The following `*_mask` functions will accept any array-like input runtime, but
# since their use-cases are specific to masks, they only accept `MaskedArray` inputs.

# keep in sync with `MaskedArray.harden_mask`
def harden_mask[MArrayT: MaskedArray](a: MArrayT) -> MArrayT: ...
# keep in sync with `MaskedArray.soften_mask`
def soften_mask[MArrayT: MaskedArray](a: MArrayT) -> MArrayT: ...
# keep in sync with `MaskedArray.shrink_mask`
def shrink_mask[MArrayT: MaskedArray](a: MArrayT) -> MArrayT: ...

# keep in sync with `MaskedArray.ids`
def ids(a: ArrayLike) -> tuple[int, int]: ...

# keep in sync with `ndarray.nonzero`
def nonzero(a: ArrayLike) -> tuple[_Array1D[np.intp], ...]: ...

# keep first overload in sync with `MaskedArray.ravel`
@overload
def ravel[DTypeT: np.dtype](a: np.ndarray[Any, DTypeT], order: _OrderKACF = "C") -> MaskedArray[tuple[int], DTypeT]: ...
@overload
def ravel[ScalarT: np.generic](a: _ArrayLike[ScalarT], order: _OrderKACF = "C") -> _Masked1D[ScalarT]: ...
@overload
def ravel(a: ArrayLike, order: _OrderKACF = "C") -> MaskedArray[tuple[int], _DTypeT_co]: ...

# keep roughly in sync with `lib._function_base_impl.copy`
@overload
def copy[MArrayT: MaskedArray](a: MArrayT, order: _OrderKACF = "C") -> MArrayT: ...
@overload
def copy[ShapeT: _Shape, DTypeT: np.dtype](
    a: np.ndarray[ShapeT, DTypeT],
    order: _OrderKACF = "C",
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload
def copy[ScalarT: np.generic](a: _ArrayLike[ScalarT], order: _OrderKACF = "C") -> _MaskedArray[ScalarT]: ...
@overload
def copy(a: ArrayLike, order: _OrderKACF = "C") -> _MaskedArray[Incomplete]: ...

# keep in sync with `_core.fromnumeric.diagonal`
@overload
def diagonal[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
) -> NDArray[ScalarT]: ...
@overload
def diagonal(
    a: ArrayLike,
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
) -> NDArray[Incomplete]: ...

# keep in sync with `_core.fromnumeric.repeat`
@overload
def repeat[ScalarT: np.generic](a: _ArrayLike[ScalarT], repeats: _ArrayLikeInt_co, axis: None = None) -> _Masked1D[ScalarT]: ...
@overload
def repeat[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    repeats: _ArrayLikeInt_co,
    axis: SupportsIndex,
) -> _MaskedArray[ScalarT]: ...
@overload
def repeat(a: ArrayLike, repeats: _ArrayLikeInt_co, axis: None = None) -> _Masked1D[Incomplete]: ...
@overload
def repeat(a: ArrayLike, repeats: _ArrayLikeInt_co, axis: SupportsIndex) -> _MaskedArray[Incomplete]: ...

# keep in sync with `_core.fromnumeric.swapaxes`
@overload
def swapaxes[MArrayT: MaskedArray](a: MArrayT, axis1: SupportsIndex, axis2: SupportsIndex) -> MArrayT: ...
@overload
def swapaxes[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis1: SupportsIndex,
    axis2: SupportsIndex,
) -> _MaskedArray[ScalarT]: ...
@overload
def swapaxes(a: ArrayLike, axis1: SupportsIndex, axis2: SupportsIndex) -> _MaskedArray[Incomplete]: ...

# NOTE: The `MaskedArray.anom` definition is specific to `MaskedArray`, so we need
# additional overloads to cover the array-like input here.
@overload  # a: MaskedArray, dtype=None
def anom[MArrayT: MaskedArray](a: MArrayT, axis: SupportsIndex | None = None, dtype: None = None) -> MArrayT: ...
@overload  # a: array-like, dtype=None
def anom[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: SupportsIndex | None = None,
    dtype: None = None,
) -> _MaskedArray[ScalarT]: ...
@overload  # a: unknown array-like, dtype: dtype-like (positional)
def anom[ScalarT: np.generic](a: ArrayLike, axis: SupportsIndex | None, dtype: _DTypeLike[ScalarT]) -> _MaskedArray[ScalarT]: ...
@overload  # a: unknown array-like, dtype: dtype-like (keyword)
def anom[ScalarT: np.generic](
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    *,
    dtype: _DTypeLike[ScalarT],
) -> _MaskedArray[ScalarT]: ...
@overload  # a: unknown array-like, dtype: unknown dtype-like (positional)
def anom(a: ArrayLike, axis: SupportsIndex | None, dtype: DTypeLike) -> _MaskedArray[Incomplete]: ...
@overload  # a: unknown array-like, dtype: unknown dtype-like (keyword)
def anom(a: ArrayLike, axis: SupportsIndex | None = None, *, dtype: DTypeLike) -> _MaskedArray[Incomplete]: ...

anomalies = anom

# Keep in sync with `any` and `MaskedArray.all`
@overload
def all(a: ArrayLike, axis: None = None, out: None = None, keepdims: Literal[False] | _NoValueType = ...) -> np.bool: ...
@overload
def all(a: ArrayLike, axis: _ShapeLike | None, out: None, keepdims: Literal[True]) -> _MaskedArray[np.bool]: ...
@overload
def all(a: ArrayLike, axis: _ShapeLike | None = None, out: None = None, *, keepdims: Literal[True]) -> _MaskedArray[np.bool]: ...
@overload
def all(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
) -> np.bool | _MaskedArray[np.bool]: ...
@overload
def all[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def all[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

# Keep in sync with `all` and `MaskedArray.any`
@overload
def any(a: ArrayLike, axis: None = None, out: None = None, keepdims: Literal[False] | _NoValueType = ...) -> np.bool: ...
@overload
def any(a: ArrayLike, axis: _ShapeLike | None, out: None, keepdims: Literal[True]) -> _MaskedArray[np.bool]: ...
@overload
def any(a: ArrayLike, axis: _ShapeLike | None = None, out: None = None, *, keepdims: Literal[True]) -> _MaskedArray[np.bool]: ...
@overload
def any(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
) -> np.bool | _MaskedArray[np.bool]: ...
@overload
def any[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def any[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    out: ArrayT, keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

# NOTE: The `MaskedArray.compress` definition uses its `DTypeT_co` type parameter,
# which wouldn't work here for array-like inputs, so we need additional overloads.
@overload
def compress[ScalarT: np.generic](
    condition: _ArrayLikeBool_co,
    a: _ArrayLike[ScalarT],
    axis: None = None,
    out: None = None,
) -> _Masked1D[ScalarT]: ...
@overload
def compress[ScalarT: np.generic](
    condition: _ArrayLikeBool_co,
    a: _ArrayLike[ScalarT],
    axis: _ShapeLike | None = None,
    out: None = None,
) -> _MaskedArray[ScalarT]: ...
@overload
def compress(condition: _ArrayLikeBool_co, a: ArrayLike, axis: None = None, out: None = None) -> _Masked1D[Incomplete]: ...
@overload
def compress(
    condition: _ArrayLikeBool_co,
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    out: None = None,
) -> _MaskedArray[Incomplete]: ...
@overload
def compress[ArrayT: np.ndarray](condition: _ArrayLikeBool_co, a: ArrayLike, axis: _ShapeLike | None, out: ArrayT) -> ArrayT: ...
@overload
def compress[ArrayT: np.ndarray](
    condition: _ArrayLikeBool_co,
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# Keep in sync with `cumprod` and `MaskedArray.cumsum`
@overload  # out: None (default)
def cumsum(
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> _MaskedArray[Incomplete]: ...
@overload  # out: ndarray (positional)
def cumsum[ArrayT: np.ndarray](a: ArrayLike, axis: SupportsIndex | None, dtype: DTypeLike | None, out: ArrayT) -> ArrayT: ...
@overload  # out: ndarray (kwarg)
def cumsum[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# Keep in sync with `cumsum` and `MaskedArray.cumsum`
@overload  # out: None (default)
def cumprod(
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> _MaskedArray[Incomplete]: ...
@overload  # out: ndarray (positional)
def cumprod[ArrayT: np.ndarray](a: ArrayLike, axis: SupportsIndex | None, dtype: DTypeLike | None, out: ArrayT) -> ArrayT: ...
@overload  # out: ndarray (kwarg)
def cumprod[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# Keep in sync with `sum`, `prod`, `product`, and `MaskedArray.mean`
@overload
def mean(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
) -> Incomplete: ...
@overload
def mean[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def mean[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

# Keep in sync with `mean`, `prod`, `product`, and `MaskedArray.sum`
@overload
def sum(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
) -> Incomplete: ...
@overload
def sum[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def sum[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

# Keep in sync with `product` and `MaskedArray.prod`
@overload
def prod(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
) -> Incomplete: ...
@overload
def prod[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def prod[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

# Keep in sync with `prod` and `MaskedArray.prod`
@overload
def product(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    keepdims: bool | _NoValueType = ...,
) -> Incomplete: ...
@overload
def product[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def product[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

# Keep in sync with `MaskedArray.trace` and `_core.fromnumeric.trace`
@overload
def trace(
    a: ArrayLike,
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
    dtype: DTypeLike | None = None,
    out: None = None,
) -> Incomplete: ...
@overload
def trace[ArrayT: np.ndarray](
    a: ArrayLike,
    offset: SupportsIndex,
    axis1: SupportsIndex,
    axis2: SupportsIndex,
    dtype: DTypeLike | None,
    out: ArrayT,
) -> ArrayT: ...
@overload
def trace[ArrayT: np.ndarray](
    a: ArrayLike,
    offset: SupportsIndex = 0,
    axis1: SupportsIndex = 0,
    axis2: SupportsIndex = 1,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
) -> ArrayT: ...

# keep in sync with `std` and `MaskedArray.var`
@overload
def std(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    mean: _ArrayLikeNumber_co | _NoValueType = ...,
) -> Incomplete: ...
@overload
def std[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    mean: _ArrayLikeNumber_co | _NoValueType = ...,
) -> ArrayT: ...
@overload
def std[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    mean: _ArrayLikeNumber_co | _NoValueType = ...,
) -> ArrayT: ...

# keep in sync with `std` and `MaskedArray.var`
@overload
def var(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    out: None = None,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    mean: _ArrayLikeNumber_co | _NoValueType = ...,
) -> Incomplete: ...
@overload
def var[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None,
    dtype: DTypeLike | None,
    out: ArrayT,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    mean: _ArrayLikeNumber_co | _NoValueType = ...,
) -> ArrayT: ...
@overload
def var[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    dtype: DTypeLike | None = None,
    *,
    out: ArrayT,
    ddof: float = 0,
    keepdims: bool | _NoValueType = ...,
    mean: _ArrayLikeNumber_co | _NoValueType = ...,
) -> ArrayT: ...

# (a, b)
minimum: _extrema_operation = ...
maximum: _extrema_operation = ...

# NOTE: this is a `_frommethod` instance at runtime
@overload
def count(a: ArrayLike, axis: None = None, keepdims: Literal[False] | _NoValueType = ...) -> int: ...
@overload
def count(a: ArrayLike, axis: _ShapeLike, keepdims: bool | _NoValueType = ...) -> NDArray[int_]: ...
@overload
def count(a: ArrayLike, axis: _ShapeLike | None = None, *, keepdims: Literal[True]) -> NDArray[int_]: ...
@overload
def count(a: ArrayLike, axis: _ShapeLike | None, keepdims: Literal[True]) -> NDArray[int_]: ...

# NOTE: this is a `_frommethod` instance at runtime
@overload
def argmin(
    a: ArrayLike,
    axis: None = None,
    fill_value: _ScalarLike_co | None = None,
    out: None = None,
    *,
    keepdims: Literal[False] | _NoValueType = ...,
) -> intp: ...
@overload
def argmin(
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    fill_value: _ScalarLike_co | None = None,
    out: None = None,
    *,
    keepdims: bool | _NoValueType = ...,
) -> Any: ...
@overload
def argmin[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    fill_value: _ScalarLike_co | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def argmin[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: SupportsIndex | None,
    fill_value: _ScalarLike_co | None,
    out: ArrayT,
    *,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

# keep in sync with `argmin`
@overload
def argmax(
    a: ArrayLike,
    axis: None = None,
    fill_value: _ScalarLike_co | None = None,
    out: None = None,
    *,
    keepdims: Literal[False] | _NoValueType = ...,
) -> intp: ...
@overload
def argmax(
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    fill_value: _ScalarLike_co | None = None,
    out: None = None,
    *,
    keepdims: bool | _NoValueType = ...,
) -> Any: ...
@overload
def argmax[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: SupportsIndex | None = None,
    fill_value: _ScalarLike_co | None = None,
    *,
    out: ArrayT,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...
@overload
def argmax[ArrayT: np.ndarray](
    a: ArrayLike,
    axis: SupportsIndex | None,
    fill_value: _ScalarLike_co | None,
    out: ArrayT,
    *,
    keepdims: bool | _NoValueType = ...,
) -> ArrayT: ...

@overload
def take[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    indices: _IntLike_co,
    axis: None = None,
    out: None = None,
    mode: _ModeKind = "raise",
) -> ScalarT: ...
@overload
def take[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    indices: _ArrayLikeInt_co,
    axis: SupportsIndex | None = None,
    out: None = None,
    mode: _ModeKind = "raise",
) -> _MaskedArray[ScalarT]: ...
@overload
def take(
    a: ArrayLike,
    indices: _IntLike_co,
    axis: SupportsIndex | None = None,
    out: None = None,
    mode: _ModeKind = "raise",
) -> Any: ...
@overload
def take(
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: SupportsIndex | None = None,
    out: None = None,
    mode: _ModeKind = "raise",
) -> _MaskedArray[Any]: ...
@overload
def take[ArrayT: np.ndarray](
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: SupportsIndex | None,
    out: ArrayT,
    mode: _ModeKind = "raise",
) -> ArrayT: ...
@overload
def take[ArrayT: np.ndarray](
    a: ArrayLike,
    indices: _ArrayLikeInt_co,
    axis: SupportsIndex | None = None,
    *,
    out: ArrayT,
    mode: _ModeKind = "raise",
) -> ArrayT: ...

def power(a, b, third=None): ...
def argsort(a, axis=..., kind=None, order=None, endwith=True, fill_value=None, *, stable=None): ...

@overload
def sort[ArrayT: np.ndarray](
    a: ArrayT,
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    endwith: bool | None = True,
    fill_value: _ScalarLike_co | None = None,
    *,
    stable: Literal[False] | None = None,
) -> ArrayT: ...
@overload
def sort(
    a: ArrayLike,
    axis: SupportsIndex = -1,
    kind: _SortKind | None = None,
    order: str | Sequence[str] | None = None,
    endwith: bool | None = True,
    fill_value: _ScalarLike_co | None = None,
    *,
    stable: Literal[False] | None = None,
) -> NDArray[Any]: ...

@overload
def compressed[ScalarT: np.generic](x: _ArrayLike[ScalarT]) -> _Array1D[ScalarT]: ...
@overload
def compressed(x: ArrayLike) -> _Array1D[Any]: ...

def concatenate(arrays, axis=0): ...
def diag(v, k=0): ...
def left_shift(a, n): ...
def right_shift(a, n): ...
def put(a: NDArray[Any], indices: _ArrayLikeInt_co, values: ArrayLike, mode: _ModeKind = "raise") -> None: ...
def putmask(a: NDArray[Any], mask: _ArrayLikeBool_co, values: ArrayLike) -> None: ...
def transpose(a, axes=None): ...
def reshape(a, new_shape, order="C"): ...
def resize(x, new_shape): ...
def ndim(obj: ArrayLike) -> int: ...
def shape(obj): ...
def size(obj: ArrayLike, axis: SupportsIndex | None = None) -> int: ...
def diff(a, /, n=1, axis=-1, prepend=..., append=...): ...
def where(condition, x=..., y=...): ...
def choose(indices, choices, out=None, mode="raise"): ...
def round_(a, decimals=0, out=None): ...
round = round_

def inner(a, b): ...
innerproduct = inner

def outer(a, b): ...
outerproduct = outer

def correlate(a, v, mode="valid", propagate_mask=True): ...
def convolve(a, v, mode="full", propagate_mask=True): ...

def allequal(a: ArrayLike, b: ArrayLike, fill_value: bool = True) -> bool: ...

def allclose(a: ArrayLike, b: ArrayLike, masked_equal: bool = True, rtol: float = 1e-5, atol: float = 1e-8) -> bool: ...

def fromflex(fxarray): ...

def append(a, b, axis=None): ...
def dot(a, b, strict=False, out=None): ...

# internal wrapper functions for the functions below
def _convert2ma(
    funcname: str,
    np_ret: str,
    np_ma_ret: str,
    params: dict[str, Any] | None = None,
) -> Callable[..., Any]: ...

# keep in sync with `_core.multiarray.arange`
@overload  # dtype=<known>
def arange[ScalarT: _ArangeScalar](
    start_or_stop: _ArangeScalar | float,
    /,
    stop: _ArangeScalar | float | None = None,
    step: _ArangeScalar | float | None = 1,
    *,
    dtype: _DTypeLike[ScalarT],
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[ScalarT]: ...
@overload  # (int-like, int-like?, int-like?)
def arange(
    start_or_stop: _IntLike_co,
    /,
    stop: _IntLike_co | None = None,
    step: _IntLike_co | None = 1,
    *,
    dtype: type[int] | _DTypeLike[np.int_] | None = None,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[np.int_]: ...
@overload  # (float, float-like?, float-like?)
def arange(
    start_or_stop: float | floating,
    /,
    stop: _FloatLike_co | None = None,
    step: _FloatLike_co | None = 1,
    *,
    dtype: type[float] | _DTypeLike[np.float64] | None = None,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[np.float64 | Any]: ...
@overload  # (float-like, float, float-like?)
def arange(
    start_or_stop: _FloatLike_co,
    /,
    stop: float | floating,
    step: _FloatLike_co | None = 1,
    *,
    dtype: type[float] | _DTypeLike[np.float64] | None = None,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[np.float64 | Any]: ...
@overload  # (timedelta, timedelta-like?, timedelta-like?)
def arange(
    start_or_stop: np.timedelta64,
    /,
    stop: _TD64Like_co | None = None,
    step: _TD64Like_co | None = 1,
    *,
    dtype: _DTypeLike[np.timedelta64] | None = None,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[np.timedelta64[Incomplete]]: ...
@overload  # (timedelta-like, timedelta, timedelta-like?)
def arange(
    start_or_stop: _TD64Like_co,
    /,
    stop: np.timedelta64,
    step: _TD64Like_co | None = 1,
    *,
    dtype: _DTypeLike[np.timedelta64] | None = None,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[np.timedelta64[Incomplete]]: ...
@overload  # (datetime, datetime, timedelta-like) (requires both start and stop)
def arange(
    start_or_stop: np.datetime64,
    /,
    stop: np.datetime64,
    step: _TD64Like_co | None = 1,
    *,
    dtype: _DTypeLike[np.datetime64] | None = None,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[np.datetime64[Incomplete]]: ...
@overload  # dtype=<unknown>
def arange(
    start_or_stop: _ArangeScalar | float,
    /,
    stop: _ArangeScalar | float | None = None,
    step: _ArangeScalar | float | None = 1,
    *,
    dtype: DTypeLike | None = None,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[Incomplete]: ...

# based on `_core.fromnumeric.clip`
@overload
def clip[ScalarT: np.generic](
    a: ScalarT,
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
    dtype: None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> ScalarT: ...
@overload
def clip[ScalarT: np.generic](
    a: NDArray[ScalarT],
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
    dtype: None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> _MaskedArray[ScalarT]: ...
@overload
def clip[MArrayT: MaskedArray](
    a: ArrayLike,
    a_min: ArrayLike | None,
    a_max: ArrayLike | None,
    out: MArrayT,
    *,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
    dtype: DTypeLike | None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> MArrayT: ...
@overload
def clip[MArrayT: MaskedArray](
    a: ArrayLike,
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    *,
    out: MArrayT,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
    dtype: DTypeLike | None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> MArrayT: ...
@overload
def clip(
    a: ArrayLike,
    a_min: ArrayLike | _NoValueType | None = ...,
    a_max: ArrayLike | _NoValueType | None = ...,
    out: None = None,
    *,
    min: ArrayLike | _NoValueType | None = ...,
    max: ArrayLike | _NoValueType | None = ...,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
    dtype: DTypeLike | None = None,
    **kwargs: Unpack[_UFuncKwargs],
) -> Incomplete: ...

# keep in sync with `_core.multiarray.ones`
@overload
def empty(
    shape: SupportsIndex,
    dtype: None = None,
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[np.float64]: ...
@overload
def empty[DTypeT: np.dtype](
    shape: SupportsIndex,
    dtype: DTypeT | _SupportsDType[DTypeT],
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> MaskedArray[tuple[int], DTypeT]: ...
@overload
def empty[ScalarT: np.generic](
    shape: SupportsIndex,
    dtype: type[ScalarT],
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[ScalarT]: ...
@overload
def empty(
    shape: SupportsIndex,
    dtype: DTypeLike | None = None,
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _Masked1D[Any]: ...
@overload  # known shape
def empty[ShapeT: _Shape](
    shape: ShapeT,
    dtype: None = None,
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> MaskedArray[ShapeT, np.dtype[np.float64]]: ...
@overload
def empty[ShapeT: _Shape, DTypeT: np.dtype](
    shape: ShapeT,
    dtype: DTypeT | _SupportsDType[DTypeT],
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> MaskedArray[ShapeT, DTypeT]: ...
@overload
def empty[ShapeT: _Shape, ScalarT: np.generic](
    shape: ShapeT,
    dtype: type[ScalarT],
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> MaskedArray[ShapeT, np.dtype[ScalarT]]: ...
@overload
def empty[ShapeT: _Shape](
    shape: ShapeT,
    dtype: DTypeLike | None = None,
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> MaskedArray[ShapeT]: ...
@overload  # unknown shape
def empty[ShapeT: _Shape](
    shape: _ShapeLike,
    dtype: None = None,
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _MaskedArray[np.float64]: ...
@overload
def empty[DTypeT: np.dtype](
    shape: _ShapeLike,
    dtype: DTypeT | _SupportsDType[DTypeT],
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> MaskedArray[_AnyShape, DTypeT]: ...
@overload
def empty[ScalarT: np.generic](
    shape: _ShapeLike,
    dtype: type[ScalarT],
    order: _OrderCF = "C",
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _MaskedArray[ScalarT]: ...
@overload
def empty(
    shape: _ShapeLike,
    dtype: DTypeLike | None = None,
    *,
    device: Literal["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> MaskedArray: ...

# keep in sync with `_core.multiarray.empty_like`
@overload
def empty_like[MArrayT: MaskedArray](
    a: MArrayT,
    /,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: Literal["cpu"] | None = None,
) -> MArrayT: ...
@overload
def empty_like[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    /,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: Literal["cpu"] | None = None,
) -> _MaskedArray[ScalarT]: ...
@overload
def empty_like[ScalarT: np.generic](
    a: Incomplete,
    /,
    dtype: _DTypeLike[ScalarT],
    order: _OrderKACF = "K",
    subok: bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: Literal["cpu"] | None = None,
) -> _MaskedArray[ScalarT]: ...
@overload
def empty_like(
    a: Incomplete,
    /,
    dtype: DTypeLike | None = None,
    order: _OrderKACF = "K",
    subok: bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: Literal["cpu"] | None = None,
) -> _MaskedArray[Incomplete]: ...

# This is a bit of a hack to avoid having to duplicate all those `empty` overloads for
# `ones` and `zeros`, that relies on the fact that empty/zeros/ones have identical
# type signatures, but may cause some type-checkers to report incorrect names in case
# of user errors. Mypy and Pyright seem to handle this just fine.
ones = empty
ones_like = empty_like
zeros = empty
zeros_like = empty_like

# keep in sync with `_core.multiarray.frombuffer`
@overload
def frombuffer(
    buffer: Buffer,
    *,
    count: SupportsIndex = -1,
    offset: SupportsIndex = 0,
    like: _SupportsArrayFunc | None = None,
) -> _MaskedArray[np.float64]: ...
@overload
def frombuffer[ScalarT: np.generic](
    buffer: Buffer,
    dtype: _DTypeLike[ScalarT],
    count: SupportsIndex = -1,
    offset: SupportsIndex = 0,
    *,
    like: _SupportsArrayFunc | None = None,
) -> _MaskedArray[ScalarT]: ...
@overload
def frombuffer(
    buffer: Buffer,
    dtype: DTypeLike | None = float,
    count: SupportsIndex = -1,
    offset: SupportsIndex = 0,
    *,
    like: _SupportsArrayFunc | None = None,
) -> _MaskedArray[Incomplete]: ...

# keep roughly in sync with `_core.numeric.fromfunction`
def fromfunction[ShapeT: _Shape, DTypeT: np.dtype](
    function: Callable[..., np.ndarray[ShapeT, DTypeT]],
    shape: Sequence[int],
    *,
    dtype: DTypeLike | None = float,
    like: _SupportsArrayFunc | None = None,
    **kwargs: object,
) -> MaskedArray[ShapeT, DTypeT]: ...

# keep roughly in sync with `_core.numeric.identity`
@overload
def identity(
    n: int,
    dtype: None = None,
    *,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> MaskedArray[tuple[int, int], np.dtype[np.float64]]: ...
@overload
def identity[ScalarT: np.generic](
    n: int,
    dtype: _DTypeLike[ScalarT],
    *,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> MaskedArray[tuple[int, int], np.dtype[ScalarT]]: ...
@overload
def identity(
    n: int,
    dtype: DTypeLike | None = None,
    *,
    like: _SupportsArrayFunc | None = None,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> MaskedArray[tuple[int, int], np.dtype[Incomplete]]: ...

# keep roughly in sync with `_core.numeric.indices`
@overload
def indices(
    dimensions: Sequence[int],
    dtype: type[int] = int,
    sparse: Literal[False] = False,
    *,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _MaskedArray[np.intp]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: type[int],
    sparse: Literal[True],
    *,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> tuple[_MaskedArray[np.intp], ...]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: type[int] = int,
    *,
    sparse: Literal[True],
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> tuple[_MaskedArray[np.intp], ...]: ...
@overload
def indices[ScalarT: np.generic](
    dimensions: Sequence[int],
    dtype: _DTypeLike[ScalarT],
    sparse: Literal[False] = False,
    *,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _MaskedArray[ScalarT]: ...
@overload
def indices[ScalarT: np.generic](
    dimensions: Sequence[int],
    dtype: _DTypeLike[ScalarT],
    sparse: Literal[True],
    *,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> tuple[_MaskedArray[ScalarT], ...]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None = int,
    sparse: Literal[False] = False,
    *,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _MaskedArray[Incomplete]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None,
    sparse: Literal[True],
    *,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> tuple[_MaskedArray[Incomplete], ...]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None = int,
    *,
    sparse: Literal[True],
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> tuple[_MaskedArray[Incomplete], ...]: ...

# keep roughly in sync with `_core.fromnumeric.squeeze`
@overload
def squeeze[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    axis: _ShapeLike | None = None,
    *,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _MaskedArray[ScalarT]: ...
@overload
def squeeze(
    a: ArrayLike,
    axis: _ShapeLike | None = None,
    *,
    fill_value: _FillValue | None = None,
    hardmask: bool = False,
) -> _MaskedArray[Incomplete]: ...
