import datetime as dt
import os
import sys
from collections.abc import Mapping, Sequence, Callable, Iterable
from typing import (
    Literal as L,
    Any,
    NoReturn,
    TypeAlias,
    TypedDict,
    overload,
    TypeVar,
    SupportsIndex,
    final,
    Final,
    Protocol,
    ClassVar,
)

import numpy as np
from numpy import (
    # Re-exports
    busdaycalendar as busdaycalendar,
    broadcast as broadcast,
    dtype as dtype,
    flatiter as flatiter,
    interp as interp,
    interp as interp_complex,
    matmul as matmul,
    ndarray as ndarray,
    nditer as nditer,
    vecdot as vecdot,

    # The rest
    ufunc,
    str_,
    uint8,
    intp,
    int_,
    float64,
    timedelta64,
    datetime64,
    generic,
    unsignedinteger,
    signedinteger,
    floating,
    complexfloating,
    _OrderKACF,
    _OrderCF,
    _CastingKind,
    _ModeKind,
    _SupportsBuffer,
    _IOProtocol,
    _CopyMode,
    _NDIterFlagsKind,
    _NDIterOpFlagsKind,
)
from numpy._typing import (
    # Shapes
    _ShapeLike,

    # DTypes
    DTypeLike,
    _DTypeLike,

    # Arrays
    NDArray,
    ArrayLike,
    _ArrayLike,
    _SupportsArrayFunc,
    _NestedSequence,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
    _ArrayLikeObject_co,
    _ArrayLikeStr_co,
    _ArrayLikeBytes_co,
    _ScalarLike_co,
    _IntLike_co,
    _FloatLike_co,
    _TD64Like_co,
)
from numpy.lib.array_utils import (
    normalize_axis_index as normalize_axis_index,
)

from .einsumfunc import (
    einsum as c_einsum,
)
from .numeric import (
    correlate as correlate,
    correlate as correlate2,
    count_nonzero as count_nonzero,
)

if sys.version_info >= (3, 11):
    from types import LiteralString
else:
    LiteralString: TypeAlias = str

if sys.version_info >= (3, 13):
    from types import CapsuleType
else:
    CapsuleType: TypeAlias = Any

__all__ = [
    '_ARRAY_API',
    'ALLOW_THREADS',
    'BUFSIZE',
    'CLIP',
    'DATETIMEUNITS',
    'ITEM_HASOBJECT',
    'ITEM_IS_POINTER',
    'LIST_PICKLE',
    'MAXDIMS',
    'MAY_SHARE_BOUNDS',
    'MAY_SHARE_EXACT',
    'NEEDS_INIT',
    'NEEDS_PYAPI',
    'RAISE',
    'USE_GETITEM',
    'USE_SETITEM',
    'WRAP',
    '_flagdict',
    'from_dlpack',
    '_place',
    '_reconstruct',
    '_vec_string',
    '_monotonicity',
    'add_docstring',
    'arange',
    'array',
    'asarray',
    'asanyarray',
    'ascontiguousarray',
    'asfortranarray',
    'bincount',
    'broadcast',
    'busday_count',
    'busday_offset',
    'busdaycalendar',
    'can_cast',
    'compare_chararrays',
    'concatenate',
    'copyto',
    'correlate',
    'correlate2',
    'count_nonzero',
    'c_einsum',
    'datetime_as_string',
    'datetime_data',
    'dot',
    'dragon4_positional',
    'dragon4_scientific',
    'dtype',
    'empty',
    'empty_like',
    'error',
    'flagsobj',
    'flatiter',
    'format_longfloat',
    'frombuffer',
    'fromfile',
    'fromiter',
    'fromstring',
    'get_handler_name',
    'get_handler_version',
    'inner',
    'interp',
    'interp_complex',
    'is_busday',
    'lexsort',
    'matmul',
    'vecdot',
    'may_share_memory',
    'min_scalar_type',
    'ndarray',
    'nditer',
    'nested_iters',
    'normalize_axis_index',
    'packbits',
    'promote_types',
    'putmask',
    'ravel_multi_index',
    'result_type',
    'scalar',
    'set_datetimeparse_function',
    'set_legacy_print_mode',
    'set_typeDict',
    'shares_memory',
    'typeinfo',
    'unpackbits',
    'unravel_index',
    'vdot',
    'where',
    'zeros',
    '_get_promotion_state',
    '_set_promotion_state',
]

_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)
_SCT = TypeVar("_SCT", bound=generic)
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])
_Self = TypeVar("_Self", bound=object)

# Valid time units
_UnitKind: TypeAlias = L[
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
_RollKind: TypeAlias = L[  # `raise` is deliberately excluded
    "nat",
    "forward",
    "following",
    "backward",
    "preceding",
    "modifiedfollowing",
    "modifiedpreceding",
]

@final
class _SupportsLenAndGetItem(Protocol[_T_contra, _T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, key: _T_contra, /) -> _T_co: ...

@final
class _SupportsPartialOrder(Protocol):
    def __lt__(self: _Self, key: _Self, /) -> bool | np.bool: ...

@final
class _SupportsDLPack(Protocol[_T_contra]):
    def __dlpack__(
        self,
        /, *,
        stream: None | _T_contra = ...,
        max_version: tuple[int, int] | None = ...,
        dl_device: tuple[int, int] | None = ...,
        copy: None | bool = ...,
    ) -> CapsuleType: ...
    # def __dlpack_device__(self, /) -> tuple[int, int]: ...

class _FlagDict(TypedDict):
    C: L[1]
    CONTIGUOUS: L[1]
    C_CONTIGUOUS: L[1]
    F: L[2]
    FORTRAN: L[2]
    F_CONTIGUOUS: L[2]
    O: L[4]
    OWNDATA: L[4]
    A: L[256]
    ALIGNED: L[256]
    W: L[1024]
    WRITEABLE: L[1024]
    X: L[8192]
    WRITEBACKIFCOPY: L[8192]


class _TypeInfo(TypedDict):
    bool: np.dtype[np.bool]
    NPY_BOOL: np.dtype[np.bool]

    float16: np.dtype[np.float16]
    NPY_HALF: np.dtype[np.half]
    float32: np.dtype[np.float32]
    NPY_FLOAT: np.dtype[np.single]
    float64: np.dtype[np.float64]
    NPY_DOUBLE: np.dtype[np.double]
    longdouble: np.dtype[np.longdouble]
    NPY_LONGDOUBLE: np.dtype[np.longdouble]

    complex64: np.dtype[np.complex64]
    NPY_CFLOAT: np.dtype[np.csingle]
    complex128: np.dtype[np.complex128]
    NPY_CDOUBLE: np.dtype[np.cdouble]
    clongdouble: np.dtype[np.clongdouble]
    NPY_CLONGDOUBLE: np.dtype[np.clongdouble]

    bytes_: np.dtype[np.bytes_]
    NPY_STRING: np.dtype[np.bytes_]
    str_: np.dtype[np.str_]
    NPY_UNICODE: np.dtype[np.str_]
    void: np.dtype[np.void]
    NPY_VOID: np.dtype[np.void]

    object_: np.dtype[np.object_]
    NPY_OBJECT: np.dtype[np.object_]

    datetime64: np.dtype[np.datetime64]
    NPY_DATETIME: np.dtype[np.datetime64]
    timedelta64: np.dtype[np.timedelta64]
    NPY_TIMEDELTA: np.dtype[np.timedelta64]

    int8: np.dtype[np.int8]
    byte: np.dtype[np.int8]
    NPY_BYTE: np.dtype[np.byte]
    uint8: np.dtype[np.uint8]
    ubyte: np.dtype[np.ubyte]
    NPY_UBYTE: np.dtype[np.ubyte]
    int16: np.dtype[np.int16]
    short: np.dtype[np.short]
    NPY_SHORT: np.dtype[np.short]
    uint16: np.dtype[np.uint16]
    ushort: np.dtype[np.ushort]
    NPY_USHORT: np.dtype[np.ushort]
    int32: np.dtype[np.int32]
    intc: np.dtype[np.intc]
    NPY_INT: np.dtype[np.intc]
    uint32: np.dtype[np.uint32]
    uintc: np.dtype[np.uintc]
    NPY_UINT: np.dtype[np.uintc]
    int64: np.dtype[np.int64]
    long: np.dtype[np.long]
    NPY_LONG: np.dtype[np.long]
    uint64: np.dtype[np.uint64]
    ulong: np.dtype[np.ulong]
    NPY_ULONG: np.dtype[np.ulong]
    longlong: np.dtype[np.longlong]
    NPY_LONGLONG: np.dtype[np.longlong]
    ulonglong: np.dtype[np.ulonglong]
    NPY_ULONGLONG: np.dtype[np.ulonglong]
    intp: np.dtype[np.intp]
    uintp: np.dtype[np.uintp]

error: Final[type[Exception]] = Exception

_ARRAY_API: Final[CapsuleType]
DATETIMEUNITS: Final[CapsuleType]

ALLOW_THREADS: Final[L[0, 1]]  # system-specific
BUFSIZE: Final[L[8192]]
MAXDIMS: Final[L[64]]

CLIP: Final[L[0]]
WRAP: Final[L[1]]
RAISE: Final[L[2]]

MAY_SHARE_EXACT: Final[L[-1]]
MAY_SHARE_BOUNDS: Final[L[0]]

ITEM_HASOBJECT: Final[L[1]]
LIST_PICKLE: Final[L[2]]
ITEM_IS_POINTER: Final[L[4]]
NEEDS_INIT: Final[L[8]]
NEEDS_PYAPI: Final[L[16]]
USE_GETITEM: Final[L[32]]
USE_SETITEM: Final[L[64]]

_flagdict: Final[_FlagDict]

typeinfo: Final[_TypeInfo]

tracemalloc_domain: Final[L[389047]]

@overload
def empty_like(
    prototype: _ArrayType,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> _ArrayType: ...
@overload
def empty_like(
    prototype: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[_SCT]: ...
@overload
def empty_like(
    prototype: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[Any]: ...
@overload
def empty_like(
    prototype: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[_SCT]: ...
@overload
def empty_like(
    prototype: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: None | _ShapeLike = ...,
    *,
    device: None | L["cpu"] = ...,
) -> NDArray[Any]: ...

@overload
def array(
    object: _ArrayType,
    dtype: None = ...,
    *,
    copy: None | bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: L[True],
    ndmin: int = ...,
    like: None | _SupportsArrayFunc = ...,
) -> _ArrayType: ...
@overload
def array(
    object: _ArrayLike[_SCT],
    dtype: None = ...,
    *,
    copy: None | bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def array(
    object: object,
    dtype: None = ...,
    *,
    copy: None | bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...
@overload
def array(
    object: Any,
    dtype: _DTypeLike[_SCT],
    *,
    copy: None | bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def array(
    object: Any,
    dtype: DTypeLike,
    *,
    copy: None | bool | _CopyMode = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

@overload
def zeros(
    shape: _ShapeLike,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...
@overload
def zeros(
    shape: _ShapeLike,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def zeros(
    shape: _ShapeLike,
    dtype: DTypeLike,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

@overload
def empty(
    shape: _ShapeLike,
    dtype: None = ...,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...
@overload
def empty(
    shape: _ShapeLike,
    dtype: _DTypeLike[_SCT],
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def empty(
    shape: _ShapeLike,
    dtype: DTypeLike,
    order: _OrderCF = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

@overload
def unravel_index(  # type: ignore[misc]
    indices: _IntLike_co,
    shape: _ShapeLike,
    order: _OrderCF = ...,
) -> tuple[intp, ...]: ...
@overload
def unravel_index(
    indices: _ArrayLikeInt_co,
    shape: _ShapeLike,
    order: _OrderCF = ...,
) -> tuple[NDArray[intp], ...]: ...

@overload
def ravel_multi_index(  # type: ignore[misc]
    multi_index: Sequence[_IntLike_co],
    dims: Sequence[SupportsIndex],
    mode: _ModeKind | tuple[_ModeKind, ...] = ...,
    order: _OrderCF = ...,
) -> intp: ...
@overload
def ravel_multi_index(
    multi_index: Sequence[_ArrayLikeInt_co],
    dims: Sequence[SupportsIndex],
    mode: _ModeKind | tuple[_ModeKind, ...] = ...,
    order: _OrderCF = ...,
) -> NDArray[intp]: ...

# NOTE: Allow any sequence of array-like objects
@overload
def concatenate(  # type: ignore[misc]
    arrays: _ArrayLike[_SCT],
    /,
    axis: None | SupportsIndex = ...,
    out: None = ...,
    *,
    dtype: None = ...,
    casting: None | _CastingKind = ...
) -> NDArray[_SCT]: ...
@overload
def concatenate(  # type: ignore[misc]
    arrays: _SupportsLenAndGetItem[int, ArrayLike],
    /,
    axis: None | SupportsIndex = ...,
    out: None = ...,
    *,
    dtype: None = ...,
    casting: None | _CastingKind = ...
) -> NDArray[Any]: ...
@overload
def concatenate(  # type: ignore[misc]
    arrays: _SupportsLenAndGetItem[int, ArrayLike],
    /,
    axis: None | SupportsIndex = ...,
    out: None = ...,
    *,
    dtype: _DTypeLike[_SCT],
    casting: None | _CastingKind = ...
) -> NDArray[_SCT]: ...
@overload
def concatenate(  # type: ignore[misc]
    arrays: _SupportsLenAndGetItem[int, ArrayLike],
    /,
    axis: None | SupportsIndex = ...,
    out: None = ...,
    *,
    dtype: DTypeLike,
    casting: None | _CastingKind = ...
) -> NDArray[Any]: ...
@overload
def concatenate(
    arrays: _SupportsLenAndGetItem[int, ArrayLike],
    axis: None | SupportsIndex,
    out: _ArrayType,
    /, *,
    dtype: DTypeLike = ...,
    casting: None | _CastingKind = ...
) -> _ArrayType: ...
@overload
def concatenate(
    arrays: _SupportsLenAndGetItem[int, ArrayLike],
    /,
    axis: None | SupportsIndex = ...,
    *,
    out: _ArrayType,
    dtype: DTypeLike = ...,
    casting: None | _CastingKind = ...
) -> _ArrayType: ...

def inner(a: ArrayLike, b: ArrayLike, /) -> Any: ...

@overload
def where(cond: ArrayLike, /) -> tuple[NDArray[intp], ...]: ...
@overload
def where(cond: ArrayLike, x: ArrayLike, y: ArrayLike, /) -> NDArray[Any]: ...

@overload
def lexsort(
    keys: Sequence[NDArray[Any] | Sequence[_ScalarLike_co]],
    axis: SupportsIndex = ...,
) -> NDArray[np.intp]: ...
@overload
def lexsort(
    keys: Sequence[_ScalarLike_co],
    axis: SupportsIndex = ...,
) -> np.intp: ...
@overload
def lexsort(
    keys: Sequence[Sequence[_SupportsPartialOrder]],
    axis: SupportsIndex = ...,
) -> NDArray[np.intp]: ...
@overload
def lexsort(
    keys: Sequence[_SupportsPartialOrder],
    axis: SupportsIndex = ...,
) -> np.intp: ...
@overload
def lexsort(
    keys: NDArray[Any] | Sequence[Any],
    axis: SupportsIndex = ...,
) -> NDArray[np.intp] | np.intp: ...

def can_cast(
    from_: ArrayLike | DTypeLike,
    to: DTypeLike,
    casting: None | _CastingKind = ...,
) -> bool: ...

def min_scalar_type(a: ArrayLike, /) -> dtype[Any]: ...

def result_type(*arrays_and_dtypes: ArrayLike | DTypeLike) -> dtype[Any]: ...

@overload
def dot(a: ArrayLike, b: ArrayLike, out: None = ...) -> Any: ...
@overload
def dot(a: ArrayLike, b: ArrayLike, out: _ArrayType) -> _ArrayType: ...

@overload
def vdot(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co, /) -> np.bool: ...  # type: ignore[misc]
@overload
def vdot(a: _ArrayLikeUInt_co, b: _ArrayLikeUInt_co, /) -> unsignedinteger[Any]: ...  # type: ignore[misc]
@overload
def vdot(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, /) -> signedinteger[Any]: ... # type: ignore[misc]
@overload
def vdot(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, /) -> floating[Any]: ...  # type: ignore[misc]
@overload
def vdot(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, /) -> complexfloating[Any, Any]: ...  # type: ignore[misc]
@overload
def vdot(a: _ArrayLikeTD64_co, b: _ArrayLikeTD64_co, /) -> timedelta64: ...
@overload
def vdot(a: _ArrayLikeObject_co, b: Any, /) -> Any: ...
@overload
def vdot(a: Any, b: _ArrayLikeObject_co, /) -> Any: ...

def bincount(
    x: ArrayLike,
    /,
    weights: None | ArrayLike = ...,
    minlength: SupportsIndex = ...,
) -> NDArray[intp]: ...

def copyto(
    dst: NDArray[Any],
    src: ArrayLike,
    casting: None | _CastingKind = ...,
    where: None | _ArrayLikeBool_co = ...,
) -> None: ...

def putmask(
    a: NDArray[Any],
    /,
    mask: _ArrayLikeBool_co,
    values: ArrayLike,
) -> None: ...

def packbits(
    a: _ArrayLikeInt_co,
    /,
    axis: None | SupportsIndex = ...,
    bitorder: L["big", "little"] = ...,
) -> NDArray[uint8]: ...

def unpackbits(
    a: _ArrayLike[uint8],
    /,
    axis: None | SupportsIndex = ...,
    count: None | SupportsIndex = ...,
    bitorder: L["big", "little"] = ...,
) -> NDArray[uint8]: ...

def shares_memory(
    a: object,
    b: object,
    /,
    max_work: None | int = ...,
) -> bool: ...

def may_share_memory(
    a: object,
    b: object,
    /,
    max_work: None | int = ...,
) -> bool: ...

@overload
def asarray(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def asarray(
    a: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...
@overload
def asarray(
    a: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def asarray(
    a: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

@overload
def asanyarray(
    a: _ArrayType,  # Preserve subclass-information
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> _ArrayType: ...
@overload
def asanyarray(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def asanyarray(
    a: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...
@overload
def asanyarray(
    a: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def asanyarray(
    a: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
    *,
    device: None | L["cpu"] = ...,
    copy: None | bool = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

@overload
def ascontiguousarray(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def ascontiguousarray(
    a: object,
    dtype: None = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...
@overload
def ascontiguousarray(
    a: Any,
    dtype: _DTypeLike[_SCT],
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def ascontiguousarray(
    a: Any,
    dtype: DTypeLike,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

@overload
def asfortranarray(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def asfortranarray(
    a: object,
    dtype: None = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...
@overload
def asfortranarray(
    a: Any,
    dtype: _DTypeLike[_SCT],
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def asfortranarray(
    a: Any,
    dtype: DTypeLike,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

def promote_types(__type1: DTypeLike, __type2: DTypeLike) -> dtype[Any]: ...

# `sep` is a de facto mandatory argument, as its default value is deprecated
@overload
def fromstring(
    string: str | bytes,
    dtype: None = ...,
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...
@overload
def fromstring(
    string: str | bytes,
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def fromstring(
    string: str | bytes,
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    *,
    sep: str,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

def frompyfunc(
    func: Callable[..., Any],
    /,
    nin: SupportsIndex,
    nout: SupportsIndex,
    *,
    identity: Any = ...,
) -> ufunc: ...

@overload
def fromfile(
    file: str | bytes | os.PathLike[Any] | _IOProtocol,
    dtype: None = ...,
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...
@overload
def fromfile(
    file: str | bytes | os.PathLike[Any] | _IOProtocol,
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def fromfile(
    file: str | bytes | os.PathLike[Any] | _IOProtocol,
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    sep: str = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

@overload
def fromiter(
    iter: Iterable[Any],
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def fromiter(
    iter: Iterable[Any],
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

@overload
def frombuffer(
    buffer: _SupportsBuffer,
    dtype: None = ...,
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[float64]: ...
@overload
def frombuffer(
    buffer: _SupportsBuffer,
    dtype: _DTypeLike[_SCT],
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def frombuffer(
    buffer: _SupportsBuffer,
    dtype: DTypeLike,
    count: SupportsIndex = ...,
    offset: SupportsIndex = ...,
    *,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

@overload
def arange(  # type: ignore[misc]
    stop: _IntLike_co,
    /, *,
    dtype: None = ...,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def arange(  # type: ignore[misc]
    start: _IntLike_co,
    stop: _IntLike_co,
    step: _IntLike_co = ...,
    dtype: None = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[signedinteger[Any]]: ...
@overload
def arange(  # type: ignore[misc]
    stop: _FloatLike_co,
    /, *,
    dtype: None = ...,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[floating[Any]]: ...
@overload
def arange(  # type: ignore[misc]
    start: _FloatLike_co,
    stop: _FloatLike_co,
    step: _FloatLike_co = ...,
    dtype: None = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[floating[Any]]: ...
@overload
def arange(
    stop: _TD64Like_co,
    /, *,
    dtype: None = ...,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[timedelta64]: ...
@overload
def arange(
    start: _TD64Like_co,
    stop: _TD64Like_co,
    step: _TD64Like_co = ...,
    dtype: None = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[timedelta64]: ...
@overload
def arange(  # both start and stop must always be specified for datetime64
    start: datetime64,
    stop: datetime64,
    step: datetime64 = ...,
    dtype: None = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[datetime64]: ...
@overload
def arange(
    stop: Any,
    /, *,
    dtype: _DTypeLike[_SCT],
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def arange(
    start: Any,
    stop: Any,
    step: Any = ...,
    *,
    dtype: _DTypeLike[_SCT],
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def arange(
    start: Any,
    stop: Any,
    step: Any,
    dtype: _DTypeLike[_SCT],
    /, *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[_SCT]: ...
@overload
def arange(
    stop: Any,
    /, *,
    dtype: DTypeLike,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...
@overload
def arange(
    start: Any,
    stop: Any,
    step: Any = ...,
    dtype: DTypeLike = ...,
    *,
    device: None | L["cpu"] = ...,
    like: None | _SupportsArrayFunc = ...,
) -> NDArray[Any]: ...

def datetime_data(
    dtype: str | _DTypeLike[datetime64] | _DTypeLike[timedelta64],
    /,
) -> tuple[str, int]: ...

# The datetime functions perform unsafe casts to `datetime64[D]`,
# so a lot of different argument types are allowed here

@overload
def busday_count(  # type: ignore[misc]
    begindates: _ScalarLike_co | dt.date,
    enddates: _ScalarLike_co | dt.date,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> int_: ...
@overload
def busday_count(  # type: ignore[misc]
    begindates: ArrayLike | dt.date | _NestedSequence[dt.date],
    enddates: ArrayLike | dt.date | _NestedSequence[dt.date],
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> NDArray[int_]: ...
@overload
def busday_count(
    begindates: ArrayLike | dt.date | _NestedSequence[dt.date],
    enddates: ArrayLike | dt.date | _NestedSequence[dt.date],
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    *,
    out: _ArrayType,
) -> _ArrayType: ...
@overload
def busday_count(
    begindates: ArrayLike | dt.date | _NestedSequence[dt.date],
    enddates: ArrayLike | dt.date | _NestedSequence[dt.date],
    weekmask: ArrayLike,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date],
    busdaycal: None | busdaycalendar,
    out: _ArrayType,
    /,
) -> _ArrayType: ...

# `roll="raise"` is (more or less?) equivalent to `casting="safe"`
@overload
def busday_offset(  # type: ignore[misc]
    dates: datetime64 | dt.date,
    offsets: _TD64Like_co | dt.timedelta,
    roll: L["raise"] = ...,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> datetime64: ...
@overload
def busday_offset(  # type: ignore[misc]
    dates: _ArrayLike[datetime64] | dt.date | _NestedSequence[dt.date],
    offsets: _ArrayLikeTD64_co | dt.timedelta | _NestedSequence[dt.timedelta],
    roll: L["raise"] = ...,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> NDArray[datetime64]: ...
@overload
def busday_offset(  # type: ignore[misc]
    dates: _ArrayLike[datetime64] | dt.date | _NestedSequence[dt.date],
    offsets: _ArrayLikeTD64_co | dt.timedelta | _NestedSequence[dt.timedelta],
    roll: L["raise"] = ...,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    *,
    out: _ArrayType,
) -> _ArrayType: ...
@overload
def busday_offset(  # type: ignore[misc]
    dates: _ArrayLike[datetime64] | dt.date | _NestedSequence[dt.date],
    offsets: _ArrayLikeTD64_co | dt.timedelta | _NestedSequence[dt.timedelta],
    roll: L["raise"],
    weekmask: ArrayLike,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date],
    busdaycal: None | busdaycalendar,
    out: _ArrayType,
    /,
) -> _ArrayType: ...
@overload
def busday_offset(  # type: ignore[misc]
    dates: _ScalarLike_co | dt.date,
    offsets: _ScalarLike_co | dt.timedelta,
    roll: _RollKind,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> datetime64: ...
@overload
def busday_offset(  # type: ignore[misc]
    dates: ArrayLike | dt.date | _NestedSequence[dt.date],
    offsets: ArrayLike | dt.timedelta | _NestedSequence[dt.timedelta],
    roll: _RollKind,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> NDArray[datetime64]: ...
@overload
def busday_offset(
    dates: ArrayLike | dt.date | _NestedSequence[dt.date],
    offsets: ArrayLike | dt.timedelta | _NestedSequence[dt.timedelta],
    roll: _RollKind,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    *,
    out: _ArrayType,
) -> _ArrayType: ...
@overload
def busday_offset(
    dates: ArrayLike | dt.date | _NestedSequence[dt.date],
    offsets: ArrayLike | dt.timedelta | _NestedSequence[dt.timedelta],
    roll: _RollKind,
    weekmask: ArrayLike,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date],
    busdaycal: None | busdaycalendar,
    out: _ArrayType,
    /,
) -> _ArrayType: ...

@overload
def is_busday(  # type: ignore[misc]
    dates: _ScalarLike_co | dt.date,
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> np.bool: ...
@overload
def is_busday(  # type: ignore[misc]
    dates: ArrayLike | _NestedSequence[dt.date],
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    out: None = ...,
) -> NDArray[np.bool]: ...
@overload
def is_busday(
    dates: ArrayLike | _NestedSequence[dt.date],
    weekmask: ArrayLike = ...,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    busdaycal: None | busdaycalendar = ...,
    *,
    out: _ArrayType,
) -> _ArrayType: ...
@overload
def is_busday(
    dates: ArrayLike | _NestedSequence[dt.date],
    weekmask: ArrayLike,
    holidays: None | ArrayLike | dt.date | _NestedSequence[dt.date],
    busdaycal: None | busdaycalendar,
    out: _ArrayType,
    /,
) -> _ArrayType: ...

@overload
def datetime_as_string(  # type: ignore[misc]
    arr: datetime64 | dt.date,
    unit: None | L["auto"] | _UnitKind = ...,
    timezone: L["naive", "UTC", "local"] | dt.tzinfo = ...,
    casting: _CastingKind = ...,
) -> str_: ...
@overload
def datetime_as_string(
    arr: _ArrayLikeDT64_co | _NestedSequence[dt.date],
    unit: None | L["auto"] | _UnitKind = ...,
    timezone: L["naive", "UTC", "local"] | dt.tzinfo = ...,
    casting: _CastingKind = ...,
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

_GetItemKeys: TypeAlias = L[
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
_SetItemKeys: TypeAlias = L[
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
    flags: None | Sequence[_NDIterFlagsKind] = ...,
    op_flags: None | Sequence[Sequence[_NDIterOpFlagsKind]] = ...,
    op_dtypes: DTypeLike | Sequence[DTypeLike] = ...,
    order: _OrderKACF = ...,
    casting: _CastingKind = ...,
    buffersize: SupportsIndex = ...,
) -> tuple[nditer, ...]: ...

def from_dlpack(x: _SupportsDLPack[Any], /) -> NDArray[Any]: ...

def _place(
    input: NDArray[Any],
    mask: NDArray[Any],
    vals: NDArray[Any],
) -> None: ...

def _reconstruct(
    subtype: type[_ArrayType],
    shape: _ShapeLike,
    dtype: DTypeLike,
) -> _ArrayType: ...

# TODO: figure out the signature (it takes least 3 arguments)
_vec_string: Callable[..., Any]

def _monotonicity(x: ArrayLike) -> int: ...
def _get_promotion_state() -> LiteralString: ...
def _set_promotion_state(state: LiteralString, /) -> None: ...

def dragon4_positional(
    x: _FloatLike_co,
    precision: _IntLike_co = ...,
    min_digits: _IntLike_co = ...,
    unique: bool = ...,
    fractional: bool = ...,
    trim: None | L['k', '.', '0', '-'] = ...,
    sign: bool = ...,
    pad_left: _IntLike_co = ...,
    pad_right: _IntLike_co = ...,
) -> str: ...

def dragon4_scientific(
    x: _FloatLike_co,
    precision: _IntLike_co = ...,
    min_digits: _IntLike_co = ...,
    unique: bool = ...,
    fractional: bool = ...,
    trim: None | L['k', '.', '0', '-'] = ...,
    sign: bool = ...,
    pad_left: _IntLike_co = ...,
    pad_right: _IntLike_co = ...,
    exp_digits: _IntLike_co = ...,
) -> str: ...

def format_longfloat(
    x: np.longdouble,
    precision: _IntLike_co = ...,
) -> str: ...

def get_handler_name(a: None | NDArray[Any] = ...) -> LiteralString | None: ...
def get_handler_version(a: None | NDArray[Any] = ...) -> int | None: ...

def scalar(dtype: DTypeLike, obj: _ScalarLike_co) -> NDArray[Any]: ...

# NOTE: this function has been removed: raises a `RuntimeError` when called
set_datetimeparse_function: Callable[..., NoReturn]

def set_legacy_print_mode(mode: _IntLike_co, /) -> None: ...
def set_typeDict(dict: Mapping[str, type[np.generic]]) -> None: ...
