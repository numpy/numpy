# TODO: Sort out any and all missing functions in this namespace
import datetime as dt
from _typeshed import Incomplete, StrOrBytesPath, SupportsLenAndGetItem
from collections.abc import Buffer, Callable, Iterable, Sequence
from typing import (
    Any,
    ClassVar,
    Final,
    Literal as L,
    Protocol,
    SupportsIndex,
    final,
    overload,
    type_check_only,
)
from typing_extensions import CapsuleType

import numpy as np
from numpy import (  # type: ignore[attr-defined]  # Python >=3.12
    _CastingKind,
    _CopyMode,
    _ModeKind,
    _NDIterFlagsKind,
    _NDIterFlagsOp,
    _OrderCF,
    _OrderKACF,
    _SupportsFileMethods,
    broadcast,
    busdaycalendar,
    complexfloating,
    correlate,
    count_nonzero,
    datetime64,
    dtype,
    einsum as c_einsum,
    flatiter,
    float64,
    floating,
    from_dlpack,
    int_,
    interp,
    intp,
    matmul,
    ndarray,
    nditer,
    signedinteger,
    str_,
    timedelta64,
    ufunc,
    uint8,
    unsignedinteger,
    vecdot,
)
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _AnyShape,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeBytes_co,
    _ArrayLikeComplex_co,
    _ArrayLikeDT64_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeStr_co,
    _ArrayLikeTD64_co,
    _ArrayLikeUInt_co,
    _DTypeLike,
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
from numpy._typing._ufunc import (
    _2PTuple,
    _PyFunc_Nin1_Nout1,
    _PyFunc_Nin1P_Nout2P,
    _PyFunc_Nin2_Nout1,
    _PyFunc_Nin3P_Nout1,
)

__all__ = [
    "_ARRAY_API",
    "ALLOW_THREADS",
    "BUFSIZE",
    "CLIP",
    "DATETIMEUNITS",
    "ITEM_HASOBJECT",
    "ITEM_IS_POINTER",
    "LIST_PICKLE",
    "MAXDIMS",
    "MAY_SHARE_BOUNDS",
    "MAY_SHARE_EXACT",
    "NEEDS_INIT",
    "NEEDS_PYAPI",
    "RAISE",
    "USE_GETITEM",
    "USE_SETITEM",
    "WRAP",
    "_flagdict",
    "from_dlpack",
    "_place",
    "_reconstruct",
    "_vec_string",
    "_monotonicity",
    "add_docstring",
    "arange",
    "array",
    "asarray",
    "asanyarray",
    "ascontiguousarray",
    "asfortranarray",
    "bincount",
    "broadcast",
    "busday_count",
    "busday_offset",
    "busdaycalendar",
    "can_cast",
    "compare_chararrays",
    "concatenate",
    "copyto",
    "correlate",
    "correlate2",
    "count_nonzero",
    "c_einsum",
    "datetime_as_string",
    "datetime_data",
    "dot",
    "dragon4_positional",
    "dragon4_scientific",
    "dtype",
    "empty",
    "empty_like",
    "error",
    "flagsobj",
    "flatiter",
    "format_longfloat",
    "frombuffer",
    "fromfile",
    "fromiter",
    "fromstring",
    "get_handler_name",
    "get_handler_version",
    "inner",
    "interp",
    "interp_complex",
    "is_busday",
    "lexsort",
    "matmul",
    "vecdot",
    "may_share_memory",
    "min_scalar_type",
    "ndarray",
    "nditer",
    "nested_iters",
    "normalize_axis_index",
    "packbits",
    "promote_types",
    "putmask",
    "ravel_multi_index",
    "result_type",
    "scalar",
    "set_datetimeparse_function",
    "set_typeDict",
    "shares_memory",
    "typeinfo",
    "unpackbits",
    "unravel_index",
    "vdot",
    "where",
    "zeros",
]

type _Array[ShapeT: _Shape, ScalarT: np.generic] = ndarray[ShapeT, dtype[ScalarT]]
type _Array1D[ScalarT: np.generic] = ndarray[tuple[int], dtype[ScalarT]]

# Valid time units
type _UnitKind = L[
    "Y",
    "M",
    "D",
    "h",
    "m",
    "s",
    "ms",
    "us", "μs",
    "ns",
    "ps",
    "fs",
    "as",
]
type _RollKind = L[  # `raise` is deliberately excluded
    "nat",
    "forward",
    "following",
    "backward",
    "preceding",
    "modifiedfollowing",
    "modifiedpreceding",
]

type _ArangeScalar = np.integer | np.floating | np.datetime64 | np.timedelta64

# The datetime functions perform unsafe casts to `datetime64[D]`,
# so a lot of different argument types are allowed here
type _ToDates = dt.date | _NestedSequence[dt.date]
type _ToDeltas = dt.timedelta | _NestedSequence[dt.timedelta]

@type_check_only
class _SupportsArray[ArrayT_co: np.ndarray](Protocol):
    def __array__(self, /) -> ArrayT_co: ...

@type_check_only
class _ConstructorEmpty(Protocol):
    # 1-D shape
    @overload
    def __call__(
        self,
        /,
        shape: SupportsIndex,
        dtype: None = None,
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> _Array1D[float64]: ...
    @overload
    def __call__[DTypeT: np.dtype](
        self,
        /,
        shape: SupportsIndex,
        dtype: DTypeT | _SupportsDType[DTypeT],
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> ndarray[tuple[int], DTypeT]: ...
    @overload
    def __call__[ScalarT: np.generic](
        self,
        /,
        shape: SupportsIndex,
        dtype: type[ScalarT],
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> _Array1D[ScalarT]: ...
    @overload
    def __call__(
        self,
        /,
        shape: SupportsIndex,
        dtype: DTypeLike | None = None,
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> _Array1D[Incomplete]: ...

    # known shape
    @overload
    def __call__[ShapeT: _Shape](
        self,
        /,
        shape: ShapeT,
        dtype: None = None,
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> _Array[ShapeT, float64]: ...
    @overload
    def __call__[ShapeT: _Shape, DTypeT: np.dtype](
        self,
        /,
        shape: ShapeT,
        dtype: DTypeT | _SupportsDType[DTypeT],
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> ndarray[ShapeT, DTypeT]: ...
    @overload
    def __call__[ShapeT: _Shape, ScalarT: np.generic](
        self,
        /,
        shape: ShapeT,
        dtype: type[ScalarT],
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> _Array[ShapeT, ScalarT]: ...
    @overload
    def __call__[ShapeT: _Shape](
        self,
        /,
        shape: ShapeT,
        dtype: DTypeLike | None = None,
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> _Array[ShapeT, Incomplete]: ...

    # unknown shape
    @overload
    def __call__(
        self, /,
        shape: _ShapeLike,
        dtype: None = None,
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> NDArray[float64]: ...
    @overload
    def __call__[DTypeT: np.dtype](
        self, /,
        shape: _ShapeLike,
        dtype: DTypeT | _SupportsDType[DTypeT],
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> ndarray[_AnyShape, DTypeT]: ...
    @overload
    def __call__[ScalarT: np.generic](
        self, /,
        shape: _ShapeLike,
        dtype: type[ScalarT],
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> NDArray[ScalarT]: ...
    @overload
    def __call__(
        self,
        /,
        shape: _ShapeLike,
        dtype: DTypeLike | None = None,
        order: _OrderCF = "C",
        *,
        device: L["cpu"] | None = None,
        like: _SupportsArrayFunc | None = None,
    ) -> NDArray[Incomplete]: ...

# using `Final` or `TypeAlias` will break stubtest
error = Exception

# from ._multiarray_umath
ITEM_HASOBJECT: Final = 1
LIST_PICKLE: Final = 2
ITEM_IS_POINTER: Final = 4
NEEDS_INIT: Final = 8
NEEDS_PYAPI: Final = 16
USE_GETITEM: Final = 32
USE_SETITEM: Final = 64
DATETIMEUNITS: Final[CapsuleType] = ...
_ARRAY_API: Final[CapsuleType] = ...

_flagdict: Final[dict[str, int]] = ...
_monotonicity: Final[Callable[..., object]] = ...
_place: Final[Callable[..., object]] = ...
_reconstruct: Final[Callable[..., object]] = ...
_vec_string: Final[Callable[..., object]] = ...
correlate2: Final[Callable[..., object]] = ...
dragon4_positional: Final[Callable[..., object]] = ...
dragon4_scientific: Final[Callable[..., object]] = ...
interp_complex: Final[Callable[..., object]] = ...
set_datetimeparse_function: Final[Callable[..., object]] = ...

def get_handler_name(a: NDArray[Any] = ..., /) -> str | None: ...
def get_handler_version(a: NDArray[Any] = ..., /) -> int | None: ...
def format_longfloat(x: np.longdouble, precision: int) -> str: ...
def scalar[DTypeT: np.dtype](dtype: DTypeT, object: bytes | object = ...) -> ndarray[tuple[()], DTypeT]: ...
def set_typeDict(dict_: dict[str, np.dtype], /) -> None: ...

typeinfo: Final[dict[str, np.dtype[np.generic]]] = ...

ALLOW_THREADS: Final[int]  # 0 or 1 (system-specific)
BUFSIZE: Final = 8_192
CLIP: Final = 0
WRAP: Final = 1
RAISE: Final = 2
MAXDIMS: Final = 64
MAY_SHARE_BOUNDS: Final = 0
MAY_SHARE_EXACT: Final = -1
tracemalloc_domain: Final = 389_047

zeros: Final[_ConstructorEmpty] = ...
empty: Final[_ConstructorEmpty] = ...

@overload
def empty_like[ArrayT: np.ndarray](
    prototype: ArrayT,
    /,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> ArrayT: ...
@overload
def empty_like[ScalarT: np.generic](
    prototype: _ArrayLike[ScalarT],
    /,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[ScalarT]: ...
@overload
def empty_like[ScalarT: np.generic](
    prototype: Incomplete,
    /,
    dtype: _DTypeLike[ScalarT],
    order: _OrderKACF = "K",
    subok: bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[ScalarT]: ...
@overload
def empty_like(
    prototype: Incomplete,
    /,
    dtype: DTypeLike | None = None,
    order: _OrderKACF = "K",
    subok: bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[Incomplete]: ...

@overload
def array[ArrayT: np.ndarray](
    object: ArrayT,
    dtype: None = None,
    *,
    copy: bool | _CopyMode | None = True,
    order: _OrderKACF = "K",
    subok: L[True],
    ndmin: int = 0,
    ndmax: int = 0,
    like: _SupportsArrayFunc | None = None,
) -> ArrayT: ...
@overload
def array[ArrayT: np.ndarray](
    object: _SupportsArray[ArrayT],
    dtype: None = None,
    *,
    copy: bool | _CopyMode | None = True,
    order: _OrderKACF = "K",
    subok: L[True],
    ndmin: L[0] = 0,
    ndmax: int = 0,
    like: _SupportsArrayFunc | None = None,
) -> ArrayT: ...
@overload
def array[ScalarT: np.generic](
    object: _ArrayLike[ScalarT],
    dtype: None = None,
    *,
    copy: bool | _CopyMode | None = True,
    order: _OrderKACF = "K",
    subok: bool = False,
    ndmin: int = 0,
    ndmax: int = 0,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[ScalarT]: ...
@overload
def array[ScalarT: np.generic](
    object: Any,
    dtype: _DTypeLike[ScalarT],
    *,
    copy: bool | _CopyMode | None = True,
    order: _OrderKACF = "K",
    subok: bool = False,
    ndmin: int = 0,
    ndmax: int = 0,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[ScalarT]: ...
@overload
def array(
    object: Any,
    dtype: DTypeLike | None = None,
    *,
    copy: bool | _CopyMode | None = True,
    order: _OrderKACF = "K",
    subok: bool = False,
    ndmin: int = 0,
    ndmax: int = 0,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[Any]: ...

#
@overload
def ravel_multi_index(
    multi_index: SupportsLenAndGetItem[_IntLike_co],
    dims: _ShapeLike,
    mode: _ModeKind | tuple[_ModeKind, ...] = "raise",
    order: _OrderCF = "C",
) -> intp: ...
@overload
def ravel_multi_index(
    multi_index: SupportsLenAndGetItem[_ArrayLikeInt_co],
    dims: _ShapeLike,
    mode: _ModeKind | tuple[_ModeKind, ...] = "raise",
    order: _OrderCF = "C",
) -> NDArray[intp]: ...

#
@overload
def unravel_index(indices: _IntLike_co, shape: _ShapeLike, order: _OrderCF = "C") -> tuple[intp, ...]: ...
@overload
def unravel_index(indices: _ArrayLikeInt_co, shape: _ShapeLike, order: _OrderCF = "C") -> tuple[NDArray[intp], ...]: ...

#
def normalize_axis_index(axis: int, ndim: int, msg_prefix: str | None = None) -> int: ...

# NOTE: Allow any sequence of array-like objects
@overload
def concatenate[ScalarT: np.generic](
    arrays: _ArrayLike[ScalarT],
    /,
    axis: SupportsIndex | None = 0,
    out: None = None,
    *,
    dtype: None = None,
    casting: _CastingKind | None = "same_kind",
) -> NDArray[ScalarT]: ...
@overload
def concatenate[ScalarT: np.generic](
    arrays: SupportsLenAndGetItem[ArrayLike],
    /,
    axis: SupportsIndex | None = 0,
    out: None = None,
    *,
    dtype: _DTypeLike[ScalarT],
    casting: _CastingKind | None = "same_kind",
) -> NDArray[ScalarT]: ...
@overload
def concatenate(
    arrays: SupportsLenAndGetItem[ArrayLike],
    /,
    axis: SupportsIndex | None = 0,
    out: None = None,
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind | None = "same_kind",
) -> NDArray[Incomplete]: ...
@overload
def concatenate[OutT: np.ndarray](
    arrays: SupportsLenAndGetItem[ArrayLike],
    /,
    axis: SupportsIndex | None = 0,
    *,
    out: OutT,
    dtype: DTypeLike | None = None,
    casting: _CastingKind | None = "same_kind",
) -> OutT: ...
@overload
def concatenate[OutT: np.ndarray](
    arrays: SupportsLenAndGetItem[ArrayLike],
    /,
    axis: SupportsIndex | None,
    out: OutT,
    *,
    dtype: DTypeLike | None = None,
    casting: _CastingKind | None = "same_kind",
) -> OutT: ...

def inner(a: ArrayLike, b: ArrayLike, /) -> Incomplete: ...

@overload
def where(condition: ArrayLike, x: None = None, y: None = None, /) -> tuple[NDArray[intp], ...]: ...
@overload
def where(condition: ArrayLike, x: ArrayLike, y: ArrayLike, /) -> NDArray[Incomplete]: ...

def lexsort(keys: ArrayLike, axis: SupportsIndex = -1) -> NDArray[intp]: ...

def can_cast(from_: ArrayLike | DTypeLike, to: DTypeLike, casting: _CastingKind = "safe") -> bool: ...

def min_scalar_type(a: ArrayLike, /) -> dtype: ...
def result_type(*arrays_and_dtypes: ArrayLike | DTypeLike | None) -> dtype: ...

@overload
def dot(a: ArrayLike, b: ArrayLike, out: None = None) -> Incomplete: ...
@overload
def dot[OutT: np.ndarray](a: ArrayLike, b: ArrayLike, out: OutT) -> OutT: ...

@overload
def vdot(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co, /) -> np.bool: ...
@overload
def vdot(a: _ArrayLikeUInt_co, b: _ArrayLikeUInt_co, /) -> unsignedinteger: ...
@overload
def vdot(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, /) -> signedinteger: ...
@overload
def vdot(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, /) -> floating: ...
@overload
def vdot(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, /) -> complexfloating: ...
@overload
def vdot(a: _ArrayLikeTD64_co, b: _ArrayLikeTD64_co, /) -> timedelta64: ...
@overload
def vdot(a: _ArrayLikeObject_co, b: object, /) -> Any: ...
@overload
def vdot(a: object, b: _ArrayLikeObject_co, /) -> Any: ...

def bincount(x: ArrayLike, /, weights: ArrayLike | None = None, minlength: SupportsIndex = 0) -> NDArray[intp]: ...

def copyto(dst: ndarray, src: ArrayLike, casting: _CastingKind = "same_kind", where: object = True) -> None: ...
def putmask(a: ndarray, /, mask: _ArrayLikeBool_co, values: ArrayLike) -> None: ...

type _BitOrder = L["big", "little"]

@overload
def packbits(a: _ArrayLikeInt_co, /, axis: None = None, bitorder: _BitOrder = "big") -> ndarray[tuple[int], dtype[uint8]]: ...
@overload
def packbits(a: _ArrayLikeInt_co, /, axis: SupportsIndex, bitorder: _BitOrder = "big") -> NDArray[uint8]: ...

@overload
def unpackbits(
    a: _ArrayLike[uint8],
    /,
    axis: None = None,
    count: SupportsIndex | None = None,
    bitorder: _BitOrder = "big",
) -> ndarray[tuple[int], dtype[uint8]]: ...
@overload
def unpackbits(
    a: _ArrayLike[uint8],
    /,
    axis: SupportsIndex,
    count: SupportsIndex | None = None,
    bitorder: _BitOrder = "big",
) -> NDArray[uint8]: ...

type _MaxWork = L[-1, 0]

# any two python objects will be accepted, not just `ndarray`s
def shares_memory(a: object, b: object, /, max_work: _MaxWork = -1) -> bool: ...
def may_share_memory(a: object, b: object, /, max_work: _MaxWork = 0) -> bool: ...

@overload
def asarray[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    dtype: None = None,
    order: _OrderKACF = ...,
    *,
    device: L["cpu"] | None = ...,
    copy: bool | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def asarray[ScalarT: np.generic](
    a: Any,
    dtype: _DTypeLike[ScalarT],
    order: _OrderKACF = ...,
    *,
    device: L["cpu"] | None = ...,
    copy: bool | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def asarray(
    a: Any,
    dtype: DTypeLike | None = ...,
    order: _OrderKACF = ...,
    *,
    device: L["cpu"] | None = ...,
    copy: bool | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[Any]: ...

@overload
def asanyarray[ArrayT: np.ndarray](
    a: ArrayT,  # Preserve subclass-information
    dtype: None = None,
    order: _OrderKACF = ...,
    *,
    device: L["cpu"] | None = ...,
    copy: bool | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> ArrayT: ...
@overload
def asanyarray[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    dtype: None = None,
    order: _OrderKACF = ...,
    *,
    device: L["cpu"] | None = ...,
    copy: bool | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def asanyarray[ScalarT: np.generic](
    a: Any,
    dtype: _DTypeLike[ScalarT],
    order: _OrderKACF = ...,
    *,
    device: L["cpu"] | None = ...,
    copy: bool | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def asanyarray(
    a: Any,
    dtype: DTypeLike | None = ...,
    order: _OrderKACF = ...,
    *,
    device: L["cpu"] | None = ...,
    copy: bool | None = ...,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[Any]: ...

@overload
def ascontiguousarray[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    dtype: None = None,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def ascontiguousarray[ScalarT: np.generic](
    a: Any,
    dtype: _DTypeLike[ScalarT],
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def ascontiguousarray(
    a: Any,
    dtype: DTypeLike | None = ...,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[Any]: ...

@overload
def asfortranarray[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    dtype: None = None,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def asfortranarray[ScalarT: np.generic](
    a: Any,
    dtype: _DTypeLike[ScalarT],
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def asfortranarray(
    a: Any,
    dtype: DTypeLike | None = ...,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[Any]: ...

def promote_types(__type1: DTypeLike, __type2: DTypeLike) -> dtype: ...

# `sep` is a de facto mandatory argument, as its default value is deprecated
@overload
def fromstring(
    string: str | bytes,
    dtype: None = None,
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[float64]: ...
@overload
def fromstring[ScalarT: np.generic](
    string: str | bytes,
    dtype: _DTypeLike[ScalarT],
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def fromstring(
    string: str | bytes,
    dtype: DTypeLike | None = ...,
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[Any]: ...

@overload
def frompyfunc[ReturnT](
    func: Callable[[Any], ReturnT], /,
    nin: L[1],
    nout: L[1],
    *,
    identity: None = None,
) -> _PyFunc_Nin1_Nout1[ReturnT, None]: ...
@overload
def frompyfunc[ReturnT, IdentityT](
    func: Callable[[Any], ReturnT], /,
    nin: L[1],
    nout: L[1],
    *,
    identity: IdentityT,
) -> _PyFunc_Nin1_Nout1[ReturnT, IdentityT]: ...
@overload
def frompyfunc[ReturnT](
    func: Callable[[Any, Any], ReturnT], /,
    nin: L[2],
    nout: L[1],
    *,
    identity: None = None,
) -> _PyFunc_Nin2_Nout1[ReturnT, None]: ...
@overload
def frompyfunc[ReturnT, IdentityT](
    func: Callable[[Any, Any], ReturnT], /,
    nin: L[2],
    nout: L[1],
    *,
    identity: IdentityT,
) -> _PyFunc_Nin2_Nout1[ReturnT, IdentityT]: ...
@overload
def frompyfunc[ReturnT, NInT: int](
    func: Callable[..., ReturnT], /,
    nin: NInT,
    nout: L[1],
    *,
    identity: None = None,
) -> _PyFunc_Nin3P_Nout1[ReturnT, None, NInT]: ...
@overload
def frompyfunc[ReturnT, NInT: int, IdentityT](
    func: Callable[..., ReturnT], /,
    nin: NInT,
    nout: L[1],
    *,
    identity: IdentityT,
) -> _PyFunc_Nin3P_Nout1[ReturnT, IdentityT, NInT]: ...
@overload
def frompyfunc[ReturnT, NInT: int, NOutT: int](
    func: Callable[..., _2PTuple[ReturnT]], /,
    nin: NInT,
    nout: NOutT,
    *,
    identity: None = None,
) -> _PyFunc_Nin1P_Nout2P[ReturnT, None, NInT, NOutT]: ...
@overload
def frompyfunc[ReturnT, NInT: int, NOutT: int, IdentityT](
    func: Callable[..., _2PTuple[ReturnT]], /,
    nin: NInT,
    nout: NOutT,
    *,
    identity: IdentityT,
) -> _PyFunc_Nin1P_Nout2P[ReturnT, IdentityT, NInT, NOutT]: ...
@overload
def frompyfunc(
    func: Callable[..., Any], /,
    nin: SupportsIndex,
    nout: SupportsIndex,
    *,
    identity: object | None = ...,
) -> ufunc: ...

@overload
def fromfile(
    file: StrOrBytesPath | _SupportsFileMethods,
    dtype: None = None,
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[float64]: ...
@overload
def fromfile[ScalarT: np.generic](
    file: StrOrBytesPath | _SupportsFileMethods,
    dtype: _DTypeLike[ScalarT],
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def fromfile(
    file: StrOrBytesPath | _SupportsFileMethods,
    dtype: DTypeLike | None = ...,
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[Any]: ...

@overload
def fromiter[ScalarT: np.generic](
    iter: Iterable[Any],
    dtype: _DTypeLike[ScalarT],
    count: SupportsIndex = ...,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def fromiter(
    iter: Iterable[Any],
    dtype: DTypeLike | None,
    count: SupportsIndex = ...,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[Any]: ...

@overload
def frombuffer(
    buffer: Buffer,
    dtype: None = None,
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[float64]: ...
@overload
def frombuffer[ScalarT: np.generic](
    buffer: Buffer,
    dtype: _DTypeLike[ScalarT],
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[ScalarT]: ...
@overload
def frombuffer(
    buffer: Buffer,
    dtype: DTypeLike | None = ...,
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: _SupportsArrayFunc | None = ...,
) -> NDArray[Any]: ...

# keep in sync with ma.core.arange
# NOTE: The `float64 | Any` return types needed to avoid incompatible overlapping overloads
@overload  # dtype=<known>
def arange[ScalarT: _ArangeScalar](
    start_or_stop: _ArangeScalar | float,
    /,
    stop: _ArangeScalar | float | None = None,
    step: _ArangeScalar | float | None = 1,
    *,
    dtype: _DTypeLike[ScalarT],
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array1D[ScalarT]: ...
@overload  # (int-like, int-like?, int-like?)
def arange(
    start_or_stop: _IntLike_co,
    /,
    stop: _IntLike_co | None = None,
    step: _IntLike_co | None = 1,
    *,
    dtype: type[int] | _DTypeLike[np.int_] | None = None,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array1D[np.int_]: ...
@overload  # (float, float-like?, float-like?)
def arange(
    start_or_stop: float | floating,
    /,
    stop: _FloatLike_co | None = None,
    step: _FloatLike_co | None = 1,
    *,
    dtype: type[float] | _DTypeLike[np.float64] | None = None,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array1D[np.float64 | Any]: ...
@overload  # (float-like, float, float-like?)
def arange(
    start_or_stop: _FloatLike_co,
    /,
    stop: float | floating,
    step: _FloatLike_co | None = 1,
    *,
    dtype: type[float] | _DTypeLike[np.float64] | None = None,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array1D[np.float64 | Any]: ...
@overload  # (timedelta, timedelta-like?, timedelta-like?)
def arange(
    start_or_stop: np.timedelta64,
    /,
    stop: _TD64Like_co | None = None,
    step: _TD64Like_co | None = 1,
    *,
    dtype: _DTypeLike[np.timedelta64] | None = None,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array1D[np.timedelta64[Incomplete]]: ...
@overload  # (timedelta-like, timedelta, timedelta-like?)
def arange(
    start_or_stop: _TD64Like_co,
    /,
    stop: np.timedelta64,
    step: _TD64Like_co | None = 1,
    *,
    dtype: _DTypeLike[np.timedelta64] | None = None,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array1D[np.timedelta64[Incomplete]]: ...
@overload  # (datetime, datetime, timedelta-like) (requires both start and stop)
def arange(
    start_or_stop: np.datetime64,
    /,
    stop: np.datetime64,
    step: _TD64Like_co | None = 1,
    *,
    dtype: _DTypeLike[np.datetime64] | None = None,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array1D[np.datetime64[Incomplete]]: ...
@overload  # dtype=<unknown>
def arange(
    start_or_stop: _ArangeScalar | float,
    /,
    stop: _ArangeScalar | float | None = None,
    step: _ArangeScalar | float | None = 1,
    *,
    dtype: DTypeLike | None = None,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array1D[Incomplete]: ...

#
def datetime_data(dtype: str | _DTypeLike[datetime64 | timedelta64], /) -> tuple[str, int]: ...

@overload
def busday_count(
    begindates: _ScalarLike_co | dt.date,
    enddates: _ScalarLike_co | dt.date,
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates = (),
    busdaycal: busdaycalendar | None = None,
    out: None = None,
) -> int_: ...
@overload
def busday_count(
    begindates: ArrayLike | _ToDates,
    enddates: ArrayLike | _ToDates,
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates = (),
    busdaycal: busdaycalendar | None = None,
    out: None = None,
) -> NDArray[int_]: ...
@overload
def busday_count[OutT: np.ndarray](
    begindates: ArrayLike | _ToDates,
    enddates: ArrayLike | _ToDates,
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates = (),
    busdaycal: busdaycalendar | None = None,
    *,
    out: OutT,
) -> OutT: ...
@overload
def busday_count[OutT: np.ndarray](
    begindates: ArrayLike | _ToDates,
    enddates: ArrayLike | _ToDates,
    weekmask: ArrayLike,
    holidays: ArrayLike | _ToDates,
    busdaycal: busdaycalendar | None,
    out: OutT,
) -> OutT: ...

# `roll="raise"` is (more or less?) equivalent to `casting="safe"`
@overload
def busday_offset(
    dates: datetime64 | dt.date,
    offsets: _TD64Like_co | dt.timedelta,
    roll: L["raise"] = "raise",
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates | None = None,
    busdaycal: busdaycalendar | None = None,
    out: None = None,
) -> datetime64: ...
@overload
def busday_offset(
    dates: _ArrayLike[datetime64] | _NestedSequence[dt.date],
    offsets: _ArrayLikeTD64_co | _ToDeltas,
    roll: L["raise"] = "raise",
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates | None = None,
    busdaycal: busdaycalendar | None = None,
    out: None = None,
) -> NDArray[datetime64]: ...
@overload
def busday_offset[OutT: np.ndarray](
    dates: _ArrayLike[datetime64] | _ToDates,
    offsets: _ArrayLikeTD64_co | _ToDeltas,
    roll: L["raise"] = "raise",
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates | None = None,
    busdaycal: busdaycalendar | None = None,
    *,
    out: OutT,
) -> OutT: ...
@overload
def busday_offset[OutT: np.ndarray](
    dates: _ArrayLike[datetime64] | _ToDates,
    offsets: _ArrayLikeTD64_co | _ToDeltas,
    roll: L["raise"],
    weekmask: ArrayLike,
    holidays: ArrayLike | _ToDates | None,
    busdaycal: busdaycalendar | None,
    out: OutT,
) -> OutT: ...
@overload
def busday_offset(
    dates: _ScalarLike_co | dt.date,
    offsets: _ScalarLike_co | dt.timedelta,
    roll: _RollKind,
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates | None = None,
    busdaycal: busdaycalendar | None = None,
    out: None = None,
) -> datetime64: ...
@overload
def busday_offset(
    dates: ArrayLike | _NestedSequence[dt.date],
    offsets: ArrayLike | _ToDeltas,
    roll: _RollKind,
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates | None = None,
    busdaycal: busdaycalendar | None = None,
    out: None = None,
) -> NDArray[datetime64]: ...
@overload
def busday_offset[OutT: np.ndarray](
    dates: ArrayLike | _ToDates,
    offsets: ArrayLike | _ToDeltas,
    roll: _RollKind,
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates | None = None,
    busdaycal: busdaycalendar | None = None,
    *,
    out: OutT,
) -> OutT: ...
@overload
def busday_offset[OutT: np.ndarray](
    dates: ArrayLike | _ToDates,
    offsets: ArrayLike | _ToDeltas,
    roll: _RollKind,
    weekmask: ArrayLike,
    holidays: ArrayLike | _ToDates | None,
    busdaycal: busdaycalendar | None,
    out: OutT,
) -> OutT: ...

@overload
def is_busday(
    dates: _ScalarLike_co | dt.date,
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates | None = None,
    busdaycal: busdaycalendar | None = None,
    out: None = None,
) -> np.bool: ...
@overload
def is_busday(
    dates: ArrayLike | _NestedSequence[dt.date],
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates | None = None,
    busdaycal: busdaycalendar | None = None,
    out: None = None,
) -> NDArray[np.bool]: ...
@overload
def is_busday[OutT: np.ndarray](
    dates: ArrayLike | _ToDates,
    weekmask: ArrayLike = "1111100",
    holidays: ArrayLike | _ToDates | None = None,
    busdaycal: busdaycalendar | None = None,
    *,
    out: OutT,
) -> OutT: ...
@overload
def is_busday[OutT: np.ndarray](
    dates: ArrayLike | _ToDates,
    weekmask: ArrayLike,
    holidays: ArrayLike | _ToDates | None,
    busdaycal: busdaycalendar | None,
    out: OutT,
) -> OutT: ...

type _TimezoneContext = L["naive", "UTC", "local"] | dt.tzinfo

@overload
def datetime_as_string(
    arr: datetime64 | dt.date,
    unit: L["auto"] | _UnitKind | None = None,
    timezone: _TimezoneContext = "naive",
    casting: _CastingKind = "same_kind",
) -> str_: ...
@overload
def datetime_as_string(
    arr: _ArrayLikeDT64_co | _NestedSequence[dt.date],
    unit: L["auto"] | _UnitKind | None = None,
    timezone: _TimezoneContext = "naive",
    casting: _CastingKind = "same_kind",
) -> NDArray[str_]: ...

@overload
def compare_chararrays(
    a1: _ArrayLikeStr_co,
    a2: _ArrayLikeStr_co,
    cmp: L["<", "<=", "==", ">=", ">", "!="],
    rstrip: bool,
) -> NDArray[np.bool]: ...
@overload
def compare_chararrays(
    a1: _ArrayLikeBytes_co,
    a2: _ArrayLikeBytes_co,
    cmp: L["<", "<=", "==", ">=", ">", "!="],
    rstrip: bool,
) -> NDArray[np.bool]: ...

def add_docstring(obj: Callable[..., Any], docstring: str, /) -> None: ...

type _GetItemKeys = L[
    "C", "CONTIGUOUS", "C_CONTIGUOUS",
    "F", "FORTRAN", "F_CONTIGUOUS",
    "W", "WRITEABLE",
    "B", "BEHAVED",
    "O", "OWNDATA",
    "A", "ALIGNED",
    "X", "WRITEBACKIFCOPY",
    "CA", "CARRAY",
    "FA", "FARRAY",
    "FNC",
    "FORC",
]
type _SetItemKeys = L[
    "A", "ALIGNED",
    "W", "WRITEABLE",
    "X", "WRITEBACKIFCOPY",
]

@final
class flagsobj:
    __hash__: ClassVar[None]  # type: ignore[assignment]
    aligned: bool
    # NOTE: deprecated
    # updateifcopy: bool
    writeable: bool
    writebackifcopy: bool
    @property
    def behaved(self) -> bool: ...
    @property
    def c_contiguous(self) -> bool: ...
    @property
    def carray(self) -> bool: ...
    @property
    def contiguous(self) -> bool: ...
    @property
    def f_contiguous(self) -> bool: ...
    @property
    def farray(self) -> bool: ...
    @property
    def fnc(self) -> bool: ...
    @property
    def forc(self) -> bool: ...
    @property
    def fortran(self) -> bool: ...
    @property
    def num(self) -> int: ...
    @property
    def owndata(self) -> bool: ...
    def __getitem__(self, key: _GetItemKeys) -> bool: ...
    def __setitem__(self, key: _SetItemKeys, value: bool) -> None: ...

def nested_iters(
    op: ArrayLike | Sequence[ArrayLike],
    axes: Sequence[Sequence[SupportsIndex]],
    flags: Sequence[_NDIterFlagsKind] | None = ...,
    op_flags: Sequence[Sequence[_NDIterFlagsOp]] | None = ...,
    op_dtypes: DTypeLike | Sequence[DTypeLike | None] | None = ...,
    order: _OrderKACF = ...,
    casting: _CastingKind = ...,
    buffersize: SupportsIndex = ...,
) -> tuple[nditer, ...]: ...
