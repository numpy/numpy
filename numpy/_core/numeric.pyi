from builtins import bool as py_bool
from collections.abc import Callable, Sequence
from typing import (
    Any,
    Final,
    Literal as L,
    Never,
    NoReturn,
    SupportsAbs,
    SupportsIndex,
    TypeAlias,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np
from numpy import (
    False_,
    True_,
    _OrderCF,
    _OrderKACF,
    bitwise_not,
    inf,
    little_endian,
    nan,
    newaxis,
    ufunc,
)
from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeComplex_co,
    _ArrayLikeFloat_co,
    _ArrayLikeInt_co,
    _ArrayLikeObject_co,
    _ArrayLikeTD64_co,
    _ArrayLikeUInt_co,
    _DTypeLike,
    _NestedSequence,
    _ScalarLike_co,
    _Shape,
    _ShapeLike,
    _SupportsArrayFunc,
    _SupportsDType,
)
from numpy.lib._array_utils_impl import normalize_axis_tuple as normalize_axis_tuple

from ._asarray import require
from ._ufunc_config import (
    errstate,
    getbufsize,
    geterr,
    geterrcall,
    setbufsize,
    seterr,
    seterrcall,
)
from .arrayprint import (
    array2string,
    array_repr,
    array_str,
    format_float_positional,
    format_float_scientific,
    get_printoptions,
    printoptions,
    set_printoptions,
)
from .fromnumeric import (
    all,
    amax,
    amin,
    any,
    argmax,
    argmin,
    argpartition,
    argsort,
    around,
    choose,
    clip,
    compress,
    cumprod,
    cumsum,
    cumulative_prod,
    cumulative_sum,
    diagonal,
    matrix_transpose,
    max,
    mean,
    min,
    ndim,
    nonzero,
    partition,
    prod,
    ptp,
    put,
    ravel,
    repeat,
    reshape,
    resize,
    round,
    searchsorted,
    shape,
    size,
    sort,
    squeeze,
    std,
    sum,
    swapaxes,
    take,
    trace,
    transpose,
    var,
)
from .multiarray import (
    ALLOW_THREADS as ALLOW_THREADS,
    BUFSIZE as BUFSIZE,
    CLIP as CLIP,
    MAXDIMS as MAXDIMS,
    MAY_SHARE_BOUNDS as MAY_SHARE_BOUNDS,
    MAY_SHARE_EXACT as MAY_SHARE_EXACT,
    RAISE as RAISE,
    WRAP as WRAP,
    _Array,
    _ConstructorEmpty,
    arange,
    array,
    asanyarray,
    asarray,
    ascontiguousarray,
    asfortranarray,
    broadcast,
    can_cast,
    concatenate,
    copyto,
    dot,
    dtype,
    empty,
    empty_like,
    flatiter,
    from_dlpack,
    frombuffer,
    fromfile,
    fromiter,
    fromstring,
    inner,
    lexsort,
    matmul,
    may_share_memory,
    min_scalar_type,
    ndarray,
    nditer,
    nested_iters,
    normalize_axis_index as normalize_axis_index,
    promote_types,
    putmask,
    result_type,
    shares_memory,
    vdot,
    where,
    zeros,
)
from .numerictypes import (
    ScalarType,
    bool,
    bool_,
    busday_count,
    busday_offset,
    busdaycalendar,
    byte,
    bytes_,
    cdouble,
    character,
    clongdouble,
    complex64,
    complex128,
    complex192,
    complex256,
    complexfloating,
    csingle,
    datetime64,
    datetime_as_string,
    datetime_data,
    double,
    flexible,
    float16,
    float32,
    float64,
    float96,
    float128,
    floating,
    generic,
    half,
    inexact,
    int8,
    int16,
    int32,
    int64,
    int_,
    intc,
    integer,
    intp,
    is_busday,
    isdtype,
    issubdtype,
    long,
    longdouble,
    longlong,
    number,
    object_,
    short,
    signedinteger,
    single,
    str_,
    timedelta64,
    typecodes,
    ubyte,
    uint,
    uint8,
    uint16,
    uint32,
    uint64,
    uintc,
    uintp,
    ulong,
    ulonglong,
    unsignedinteger,
    ushort,
    void,
)
from .umath import (
    absolute,
    add,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    bitwise_and,
    bitwise_count,
    bitwise_or,
    bitwise_xor,
    cbrt,
    ceil,
    conj,
    conjugate,
    copysign,
    cos,
    cosh,
    deg2rad,
    degrees,
    divide,
    divmod,
    e,
    equal,
    euler_gamma,
    exp,
    exp2,
    expm1,
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
    log,
    log1p,
    log2,
    log10,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
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
    rint,
    sign,
    signbit,
    sin,
    sinh,
    spacing,
    sqrt,
    square,
    subtract,
    tan,
    tanh,
    true_divide,
    trunc,
    vecdot,
    vecmat,
)

__all__ = [
    "False_",
    "ScalarType",
    "True_",
    "absolute",
    "add",
    "all",
    "allclose",
    "amax",
    "amin",
    "any",
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
    "argpartition",
    "argsort",
    "argwhere",
    "around",
    "array",
    "array2string",
    "array_equal",
    "array_equiv",
    "array_repr",
    "array_str",
    "asanyarray",
    "asarray",
    "ascontiguousarray",
    "asfortranarray",
    "astype",
    "base_repr",
    "binary_repr",
    "bitwise_and",
    "bitwise_count",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "bool",
    "bool_",
    "broadcast",
    "busday_count",
    "busday_offset",
    "busdaycalendar",
    "byte",
    "bytes_",
    "can_cast",
    "cbrt",
    "cdouble",
    "ceil",
    "character",
    "choose",
    "clip",
    "clongdouble",
    "complex64",
    "complex128",
    "complex192",
    "complex256",
    "complexfloating",
    "compress",
    "concatenate",
    "conj",
    "conjugate",
    "convolve",
    "copysign",
    "copyto",
    "correlate",
    "cos",
    "cosh",
    "count_nonzero",
    "cross",
    "csingle",
    "cumprod",
    "cumsum",
    "cumulative_prod",
    "cumulative_sum",
    "datetime64",
    "datetime_as_string",
    "datetime_data",
    "deg2rad",
    "degrees",
    "diagonal",
    "divide",
    "divmod",
    "dot",
    "double",
    "dtype",
    "e",
    "empty",
    "empty_like",
    "equal",
    "errstate",
    "euler_gamma",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "flatiter",
    "flatnonzero",
    "flexible",
    "float16",
    "float32",
    "float64",
    "float96",
    "float128",
    "float_power",
    "floating",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "format_float_positional",
    "format_float_scientific",
    "frexp",
    "from_dlpack",
    "frombuffer",
    "fromfile",
    "fromfunction",
    "fromiter",
    "frompyfunc",
    "fromstring",
    "full",
    "full_like",
    "gcd",
    "generic",
    "get_printoptions",
    "getbufsize",
    "geterr",
    "geterrcall",
    "greater",
    "greater_equal",
    "half",
    "heaviside",
    "hypot",
    "identity",
    "indices",
    "inexact",
    "inf",
    "inner",
    "int8",
    "int16",
    "int32",
    "int64",
    "int_",
    "intc",
    "integer",
    "intp",
    "invert",
    "is_busday",
    "isclose",
    "isdtype",
    "isfinite",
    "isfortran",
    "isinf",
    "isnan",
    "isnat",
    "isscalar",
    "issubdtype",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "lexsort",
    "little_endian",
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
    "long",
    "longdouble",
    "longlong",
    "matmul",
    "matrix_transpose",
    "matvec",
    "max",
    "maximum",
    "may_share_memory",
    "mean",
    "min",
    "min_scalar_type",
    "minimum",
    "mod",
    "modf",
    "moveaxis",
    "multiply",
    "nan",
    "ndarray",
    "ndim",
    "nditer",
    "negative",
    "nested_iters",
    "newaxis",
    "nextafter",
    "nonzero",
    "not_equal",
    "number",
    "object_",
    "ones",
    "ones_like",
    "outer",
    "partition",
    "pi",
    "positive",
    "power",
    "printoptions",
    "prod",
    "promote_types",
    "ptp",
    "put",
    "putmask",
    "rad2deg",
    "radians",
    "ravel",
    "reciprocal",
    "remainder",
    "repeat",
    "require",
    "reshape",
    "resize",
    "result_type",
    "right_shift",
    "rint",
    "roll",
    "rollaxis",
    "round",
    "searchsorted",
    "set_printoptions",
    "setbufsize",
    "seterr",
    "seterrcall",
    "shape",
    "shares_memory",
    "short",
    "sign",
    "signbit",
    "signedinteger",
    "sin",
    "single",
    "sinh",
    "size",
    "sort",
    "spacing",
    "sqrt",
    "square",
    "squeeze",
    "std",
    "str_",
    "subtract",
    "sum",
    "swapaxes",
    "take",
    "tan",
    "tanh",
    "tensordot",
    "timedelta64",
    "trace",
    "transpose",
    "true_divide",
    "trunc",
    "typecodes",
    "ubyte",
    "ufunc",
    "uint",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "uintc",
    "uintp",
    "ulong",
    "ulonglong",
    "unsignedinteger",
    "ushort",
    "var",
    "vdot",
    "vecdot",
    "vecmat",
    "void",
    "where",
    "zeros",
    "zeros_like",
]

_T = TypeVar("_T")
_ScalarT = TypeVar("_ScalarT", bound=generic)
_NumericScalarT = TypeVar("_NumericScalarT", bound=number | timedelta64 | object_)
_DTypeT = TypeVar("_DTypeT", bound=dtype)
_ArrayT = TypeVar("_ArrayT", bound=np.ndarray[Any, Any])
_ShapeT = TypeVar("_ShapeT", bound=_Shape)
_AnyShapeT = TypeVar(
    "_AnyShapeT",
    tuple[()],
    tuple[int],
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, ...],
)

_CorrelateMode: TypeAlias = L["valid", "same", "full"]

# keep in sync with `ones_like`
@overload
def zeros_like(
    a: _ArrayT,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: L[True] = True,
    shape: None = None,
    *,
    device: L["cpu"] | None = None,
) -> _ArrayT: ...
@overload
def zeros_like(
    a: _ArrayLike[_ScalarT],
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def zeros_like(
    a: object,
    dtype: _DTypeLike[_ScalarT],
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def zeros_like(
    a: object,
    dtype: DTypeLike | None = None,
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[Any]: ...

ones: Final[_ConstructorEmpty]

# keep in sync with `zeros_like`
@overload
def ones_like(
    a: _ArrayT,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: L[True] = True,
    shape: None = None,
    *,
    device: L["cpu"] | None = None,
) -> _ArrayT: ...
@overload
def ones_like(
    a: _ArrayLike[_ScalarT],
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def ones_like(
    a: object,
    dtype: _DTypeLike[_ScalarT],
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def ones_like(
    a: object,
    dtype: DTypeLike | None = None,
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[Any]: ...

# TODO: Add overloads for bool, int, float, complex, str, bytes, and memoryview
# 1-D shape
@overload
def full(
    shape: SupportsIndex,
    fill_value: _ScalarT,
    dtype: None = None,
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[tuple[int], _ScalarT]: ...
@overload
def full(
    shape: SupportsIndex,
    fill_value: Any,
    dtype: _DTypeT | _SupportsDType[_DTypeT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> np.ndarray[tuple[int], _DTypeT]: ...
@overload
def full(
    shape: SupportsIndex,
    fill_value: Any,
    dtype: type[_ScalarT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[tuple[int], _ScalarT]: ...
@overload
def full(
    shape: SupportsIndex,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[tuple[int], Any]: ...
# known shape
@overload
def full(
    shape: _AnyShapeT,
    fill_value: _ScalarT,
    dtype: None = None,
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[_AnyShapeT, _ScalarT]: ...
@overload
def full(
    shape: _AnyShapeT,
    fill_value: Any,
    dtype: _DTypeT | _SupportsDType[_DTypeT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> np.ndarray[_AnyShapeT, _DTypeT]: ...
@overload
def full(
    shape: _AnyShapeT,
    fill_value: Any,
    dtype: type[_ScalarT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[_AnyShapeT, _ScalarT]: ...
@overload
def full(
    shape: _AnyShapeT,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[_AnyShapeT, Any]: ...
# unknown shape
@overload
def full(
    shape: _ShapeLike,
    fill_value: _ScalarT,
    dtype: None = None,
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: _DTypeT | _SupportsDType[_DTypeT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> np.ndarray[Any, _DTypeT]: ...
@overload
def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: type[_ScalarT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[Any]: ...

@overload
def full_like(
    a: _ArrayT,
    fill_value: object,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: L[True] = True,
    shape: None = None,
    *,
    device: L["cpu"] | None = None,
) -> _ArrayT: ...
@overload
def full_like(
    a: _ArrayLike[_ScalarT],
    fill_value: object,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def full_like(
    a: object,
    fill_value: object,
    dtype: _DTypeLike[_ScalarT],
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def full_like(
    a: object,
    fill_value: object,
    dtype: DTypeLike | None = None,
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[Any]: ...

#
@overload
def count_nonzero(a: ArrayLike, axis: None = None, *, keepdims: L[False] = False) -> np.intp: ...
@overload
def count_nonzero(a: _ScalarLike_co, axis: _ShapeLike | None = None, *, keepdims: L[True]) -> np.intp: ...
@overload
def count_nonzero(
    a: NDArray[Any] | _NestedSequence[ArrayLike], axis: _ShapeLike | None = None, *, keepdims: L[True]
) -> NDArray[np.intp]: ...
@overload
def count_nonzero(a: ArrayLike, axis: _ShapeLike | None = None, *, keepdims: py_bool = False) -> Any: ...

#
def isfortran(a: NDArray[Any] | generic) -> py_bool: ...

def argwhere(a: ArrayLike) -> NDArray[intp]: ...

def flatnonzero(a: ArrayLike) -> NDArray[intp]: ...

@overload
def correlate(
    a: _ArrayLike[Never],
    v: _ArrayLike[Never],
    mode: _CorrelateMode = "valid",
) -> NDArray[Any]: ...
@overload
def correlate(
    a: _ArrayLikeBool_co,
    v: _ArrayLikeBool_co,
    mode: _CorrelateMode = "valid",
) -> NDArray[np.bool]: ...
@overload
def correlate(
    a: _ArrayLikeUInt_co,
    v: _ArrayLikeUInt_co,
    mode: _CorrelateMode = "valid",
) -> NDArray[unsignedinteger]: ...
@overload
def correlate(
    a: _ArrayLikeInt_co,
    v: _ArrayLikeInt_co,
    mode: _CorrelateMode = "valid",
) -> NDArray[signedinteger]: ...
@overload
def correlate(
    a: _ArrayLikeFloat_co,
    v: _ArrayLikeFloat_co,
    mode: _CorrelateMode = "valid",
) -> NDArray[floating]: ...
@overload
def correlate(
    a: _ArrayLikeComplex_co,
    v: _ArrayLikeComplex_co,
    mode: _CorrelateMode = "valid",
) -> NDArray[complexfloating]: ...
@overload
def correlate(
    a: _ArrayLikeTD64_co,
    v: _ArrayLikeTD64_co,
    mode: _CorrelateMode = "valid",
) -> NDArray[timedelta64]: ...
@overload
def correlate(
    a: _ArrayLikeObject_co,
    v: _ArrayLikeObject_co,
    mode: _CorrelateMode = "valid",
) -> NDArray[object_]: ...

@overload
def convolve(
    a: _ArrayLike[Never],
    v: _ArrayLike[Never],
    mode: _CorrelateMode = "full",
) -> NDArray[Any]: ...
@overload
def convolve(
    a: _ArrayLikeBool_co,
    v: _ArrayLikeBool_co,
    mode: _CorrelateMode = "full",
) -> NDArray[np.bool]: ...
@overload
def convolve(
    a: _ArrayLikeUInt_co,
    v: _ArrayLikeUInt_co,
    mode: _CorrelateMode = "full",
) -> NDArray[unsignedinteger]: ...
@overload
def convolve(
    a: _ArrayLikeInt_co,
    v: _ArrayLikeInt_co,
    mode: _CorrelateMode = "full",
) -> NDArray[signedinteger]: ...
@overload
def convolve(
    a: _ArrayLikeFloat_co,
    v: _ArrayLikeFloat_co,
    mode: _CorrelateMode = "full",
) -> NDArray[floating]: ...
@overload
def convolve(
    a: _ArrayLikeComplex_co,
    v: _ArrayLikeComplex_co,
    mode: _CorrelateMode = "full",
) -> NDArray[complexfloating]: ...
@overload
def convolve(
    a: _ArrayLikeTD64_co,
    v: _ArrayLikeTD64_co,
    mode: _CorrelateMode = "full",
) -> NDArray[timedelta64]: ...
@overload
def convolve(
    a: _ArrayLikeObject_co,
    v: _ArrayLikeObject_co,
    mode: _CorrelateMode = "full",
) -> NDArray[object_]: ...

@overload
def outer(
    a: _ArrayLike[Never],
    b: _ArrayLike[Never],
    out: None = None,
) -> NDArray[Any]: ...
@overload
def outer(
    a: _ArrayLikeBool_co,
    b: _ArrayLikeBool_co,
    out: None = None,
) -> NDArray[np.bool]: ...
@overload
def outer(
    a: _ArrayLikeUInt_co,
    b: _ArrayLikeUInt_co,
    out: None = None,
) -> NDArray[unsignedinteger]: ...
@overload
def outer(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    out: None = None,
) -> NDArray[signedinteger]: ...
@overload
def outer(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    out: None = None,
) -> NDArray[floating]: ...
@overload
def outer(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    out: None = None,
) -> NDArray[complexfloating]: ...
@overload
def outer(
    a: _ArrayLikeTD64_co,
    b: _ArrayLikeTD64_co,
    out: None = None,
) -> NDArray[timedelta64]: ...
@overload
def outer(
    a: _ArrayLikeObject_co,
    b: _ArrayLikeObject_co,
    out: None = None,
) -> NDArray[object_]: ...
@overload
def outer(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    b: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    out: _ArrayT,
) -> _ArrayT: ...

# keep in sync with numpy.linalg._linalg.tensordot (ignoring `/, *`)
@overload
def tensordot(
    a: _ArrayLike[_NumericScalarT],
    b: _ArrayLike[_NumericScalarT],
    axes: int | tuple[_ShapeLike, _ShapeLike] = 2,
) -> NDArray[_NumericScalarT]: ...
@overload
def tensordot(
    a: _ArrayLikeBool_co,
    b: _ArrayLikeBool_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = 2,
) -> NDArray[bool_]: ...
@overload
def tensordot(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = 2,
) -> NDArray[int_ | Any]: ...
@overload
def tensordot(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = 2,
) -> NDArray[float64 | Any]: ...
@overload
def tensordot(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axes: int | tuple[_ShapeLike, _ShapeLike] = 2,
) -> NDArray[complex128 | Any]: ...

@overload
def roll(
    a: _ArrayLike[_ScalarT],
    shift: _ShapeLike,
    axis: _ShapeLike | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def roll(
    a: ArrayLike,
    shift: _ShapeLike,
    axis: _ShapeLike | None = None,
) -> NDArray[Any]: ...

def rollaxis(
    a: NDArray[_ScalarT],
    axis: int,
    start: int = 0,
) -> NDArray[_ScalarT]: ...

def moveaxis(
    a: NDArray[_ScalarT],
    source: _ShapeLike,
    destination: _ShapeLike,
) -> NDArray[_ScalarT]: ...

@overload
def cross(
    a: _ArrayLike[Never],
    b: _ArrayLike[Never],
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NDArray[Any]: ...
@overload
def cross(
    a: _ArrayLikeBool_co,
    b: _ArrayLikeBool_co,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NoReturn: ...
@overload
def cross(
    a: _ArrayLikeUInt_co,
    b: _ArrayLikeUInt_co,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NDArray[unsignedinteger]: ...
@overload
def cross(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NDArray[signedinteger]: ...
@overload
def cross(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NDArray[floating]: ...
@overload
def cross(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NDArray[complexfloating]: ...
@overload
def cross(
    a: _ArrayLikeObject_co,
    b: _ArrayLikeObject_co,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NDArray[object_]: ...

@overload
def indices(
    dimensions: Sequence[int],
    dtype: type[int] = ...,
    sparse: L[False] = False,
) -> NDArray[int_]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: type[int],
    sparse: L[True],
) -> tuple[NDArray[int_], ...]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: type[int] = ...,
    *,
    sparse: L[True],
) -> tuple[NDArray[int_], ...]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: _DTypeLike[_ScalarT],
    sparse: L[False] = False,
) -> NDArray[_ScalarT]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: _DTypeLike[_ScalarT],
    sparse: L[True],
) -> tuple[NDArray[_ScalarT], ...]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None = ...,
    sparse: L[False] = False,
) -> NDArray[Any]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None,
    sparse: L[True],
) -> tuple[NDArray[Any], ...]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None = ...,
    *,
    sparse: L[True],
) -> tuple[NDArray[Any], ...]: ...

def fromfunction(
    function: Callable[..., _T],
    shape: Sequence[int],
    *,
    dtype: DTypeLike | None = ...,
    like: _SupportsArrayFunc | None = None,
    **kwargs: Any,
) -> _T: ...

def isscalar(element: object) -> TypeGuard[generic | complex | str | bytes | memoryview]: ...

def binary_repr(num: SupportsIndex, width: int | None = None) -> str: ...

def base_repr(
    number: SupportsAbs[float],
    base: float = 2,
    padding: SupportsIndex | None = 0,
) -> str: ...

@overload
def identity(
    n: int,
    dtype: None = None,
    *,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[float64]: ...
@overload
def identity(
    n: int,
    dtype: _DTypeLike[_ScalarT],
    *,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[_ScalarT]: ...
@overload
def identity(
    n: int,
    dtype: DTypeLike | None = None,
    *,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[Any]: ...

def allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> py_bool: ...

@overload
def isclose(
    a: _ScalarLike_co,
    b: _ScalarLike_co,
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> np.bool: ...
@overload
def isclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> NDArray[np.bool]: ...

def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan: py_bool = False) -> py_bool: ...

def array_equiv(a1: ArrayLike, a2: ArrayLike) -> py_bool: ...

@overload
def astype(
    x: ndarray[_ShapeT, dtype],
    dtype: _DTypeLike[_ScalarT],
    /,
    *,
    copy: py_bool = True,
    device: L["cpu"] | None = None,
) -> ndarray[_ShapeT, dtype[_ScalarT]]: ...
@overload
def astype(
    x: ndarray[_ShapeT, dtype],
    dtype: DTypeLike | None,
    /,
    *,
    copy: py_bool = True,
    device: L["cpu"] | None = None,
) -> ndarray[_ShapeT, dtype]: ...
