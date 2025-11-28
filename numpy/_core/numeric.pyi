from _typeshed import Incomplete
from builtins import bool as py_bool
from collections.abc import Callable, Iterable, Sequence
from typing import (
    Any,
    Final,
    Literal as L,
    SupportsAbs,
    SupportsIndex,
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
    _ArrayLikeNumber_co,
    _ArrayLikeTD64_co,
    _CDoubleCodes,
    _Complex128Codes,
    _DoubleCodes,
    _DTypeLike,
    _DTypeLikeBool,
    _Float64Codes,
    _IntCodes,
    _NestedSequence,
    _NumberLike_co,
    _ScalarLike_co,
    _Shape,
    _ShapeLike,
    _SupportsArray,
    _SupportsArrayFunc,
    _SupportsDType,
)

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

_AnyNumericScalarT = TypeVar(
    "_AnyNumericScalarT",
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64, np.longdouble,
    np.complex64, np.complex128, np.clongdouble,
    np.timedelta64,
    np.object_,
)

type _CorrelateMode = L["valid", "same", "full"]

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]
type _Array4D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int, int], np.dtype[ScalarT]]

type _Int_co = np.integer | np.bool
type _Float_co = np.floating | _Int_co
type _Number_co = np.number | np.bool
type _TD64_co = np.timedelta64 | _Int_co

type _ArrayLike1D[ScalarT: np.generic] = _SupportsArray[np.dtype[ScalarT]] | Sequence[ScalarT]
type _ArrayLike1DBool_co = _SupportsArray[np.dtype[np.bool]] | Sequence[py_bool | np.bool]
type _ArrayLike1DInt_co = _SupportsArray[np.dtype[_Int_co]] | Sequence[int | _Int_co]
type _ArrayLike1DFloat_co = _SupportsArray[np.dtype[_Float_co]] | Sequence[float | _Float_co]
type _ArrayLike1DNumber_co = _SupportsArray[np.dtype[_Number_co]] | Sequence[complex | _Number_co]
type _ArrayLike1DTD64_co = _ArrayLike1D[_TD64_co]
type _ArrayLike1DObject_co = _ArrayLike1D[np.object_]

type _DTypeLikeInt = type[int] | _IntCodes
type _DTypeLikeFloat64 = type[float] | _Float64Codes | _DoubleCodes
type _DTypeLikeComplex128 = type[complex] | _Complex128Codes | _CDoubleCodes

###

# keep in sync with `ones_like`
@overload
def zeros_like[ArrayT: np.ndarray](
    a: ArrayT,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: L[True] = True,
    shape: None = None,
    *,
    device: L["cpu"] | None = None,
) -> ArrayT: ...
@overload
def zeros_like[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[ScalarT]: ...
@overload
def zeros_like[ScalarT: np.generic](
    a: object,
    dtype: _DTypeLike[ScalarT],
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[ScalarT]: ...
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
def ones_like[ArrayT: np.ndarray](
    a: ArrayT,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: L[True] = True,
    shape: None = None,
    *,
    device: L["cpu"] | None = None,
) -> ArrayT: ...
@overload
def ones_like[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[ScalarT]: ...
@overload
def ones_like[ScalarT: np.generic](
    a: object,
    dtype: _DTypeLike[ScalarT],
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[ScalarT]: ...
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
def full[ScalarT: np.generic](
    shape: SupportsIndex,
    fill_value: ScalarT,
    dtype: None = None,
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[tuple[int], ScalarT]: ...
@overload
def full[DTypeT: np.dtype](
    shape: SupportsIndex,
    fill_value: Any,
    dtype: DTypeT | _SupportsDType[DTypeT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> np.ndarray[tuple[int], DTypeT]: ...
@overload
def full[ScalarT: np.generic](
    shape: SupportsIndex,
    fill_value: Any,
    dtype: type[ScalarT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[tuple[int], ScalarT]: ...
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
def full[ShapeT: _Shape, ScalarT: np.generic](
    shape: ShapeT,
    fill_value: ScalarT,
    dtype: None = None,
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[ShapeT, ScalarT]: ...
@overload
def full[ShapeT: _Shape, DTypeT: np.dtype](
    shape: ShapeT,
    fill_value: Any,
    dtype: DTypeT | _SupportsDType[DTypeT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> np.ndarray[ShapeT, DTypeT]: ...
@overload
def full[ShapeT: _Shape, ScalarT: np.generic](
    shape: ShapeT,
    fill_value: Any,
    dtype: type[ScalarT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[ShapeT, ScalarT]: ...
@overload
def full[ShapeT: _Shape](
    shape: ShapeT,
    fill_value: Any,
    dtype: DTypeLike | None = None,
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> _Array[ShapeT, Any]: ...
# unknown shape
@overload
def full[ScalarT: np.generic](
    shape: _ShapeLike,
    fill_value: ScalarT,
    dtype: None = None,
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[ScalarT]: ...
@overload
def full[DTypeT: np.dtype](
    shape: _ShapeLike,
    fill_value: Any,
    dtype: DTypeT | _SupportsDType[DTypeT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> np.ndarray[Any, DTypeT]: ...
@overload
def full[ScalarT: np.generic](
    shape: _ShapeLike,
    fill_value: Any,
    dtype: type[ScalarT],
    order: _OrderCF = "C",
    *,
    device: L["cpu"] | None = None,
    like: _SupportsArrayFunc | None = None,
) -> NDArray[ScalarT]: ...
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
def full_like[ArrayT: np.ndarray](
    a: ArrayT,
    fill_value: object,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: L[True] = True,
    shape: None = None,
    *,
    device: L["cpu"] | None = None,
) -> ArrayT: ...
@overload
def full_like[ScalarT: np.generic](
    a: _ArrayLike[ScalarT],
    fill_value: object,
    dtype: None = None,
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[ScalarT]: ...
@overload
def full_like[ScalarT: np.generic](
    a: object,
    fill_value: object,
    dtype: _DTypeLike[ScalarT],
    order: _OrderKACF = "K",
    subok: py_bool = True,
    shape: _ShapeLike | None = None,
    *,
    device: L["cpu"] | None = None,
) -> NDArray[ScalarT]: ...
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
def isfortran(a: ndarray | generic) -> py_bool: ...

#
def argwhere(a: ArrayLike) -> _Array2D[np.intp]: ...
def flatnonzero(a: ArrayLike) -> _Array1D[np.intp]: ...

# keep in sync with `convolve`
@overload
def correlate(
    a: _ArrayLike1D[_AnyNumericScalarT], v: _ArrayLike1D[_AnyNumericScalarT], mode: _CorrelateMode = "valid"
) -> _Array1D[_AnyNumericScalarT]: ...
@overload
def correlate(a: _ArrayLike1DBool_co, v: _ArrayLike1DBool_co, mode: _CorrelateMode = "valid") -> _Array1D[np.bool]: ...
@overload
def correlate(a: _ArrayLike1DInt_co, v: _ArrayLike1DInt_co, mode: _CorrelateMode = "valid") -> _Array1D[np.int_ | Any]: ...
@overload
def correlate(a: _ArrayLike1DFloat_co, v: _ArrayLike1DFloat_co, mode: _CorrelateMode = "valid") -> _Array1D[np.float64 | Any]: ...
@overload
def correlate(
    a: _ArrayLike1DNumber_co, v: _ArrayLike1DNumber_co, mode: _CorrelateMode = "valid"
) -> _Array1D[np.complex128 | Any]: ...
@overload
def correlate(
    a: _ArrayLike1DTD64_co, v: _ArrayLike1DTD64_co, mode: _CorrelateMode = "valid"
) -> _Array1D[np.timedelta64 | Any]: ...

# keep in sync with `correlate`
@overload
def convolve(
    a: _ArrayLike1D[_AnyNumericScalarT], v: _ArrayLike1D[_AnyNumericScalarT], mode: _CorrelateMode = "valid"
) -> _Array1D[_AnyNumericScalarT]: ...
@overload
def convolve(a: _ArrayLike1DBool_co, v: _ArrayLike1DBool_co, mode: _CorrelateMode = "valid") -> _Array1D[np.bool]: ...
@overload
def convolve(a: _ArrayLike1DInt_co, v: _ArrayLike1DInt_co, mode: _CorrelateMode = "valid") -> _Array1D[np.int_ | Any]: ...
@overload
def convolve(a: _ArrayLike1DFloat_co, v: _ArrayLike1DFloat_co, mode: _CorrelateMode = "valid") -> _Array1D[np.float64 | Any]: ...
@overload
def convolve(
    a: _ArrayLike1DNumber_co, v: _ArrayLike1DNumber_co, mode: _CorrelateMode = "valid"
) -> _Array1D[np.complex128 | Any]: ...
@overload
def convolve(
    a: _ArrayLike1DTD64_co, v: _ArrayLike1DTD64_co, mode: _CorrelateMode = "valid"
) -> _Array1D[np.timedelta64 | Any]: ...

# keep roughly in sync with `convolve` and `correlate`, but for 2-D output and an additional `out` overload
@overload
def outer(
    a: _ArrayLike[_AnyNumericScalarT], b: _ArrayLike[_AnyNumericScalarT], out: None = None
) -> _Array2D[_AnyNumericScalarT]: ...
@overload
def outer(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co, out: None = None) -> _Array2D[np.bool]: ...
@overload
def outer(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, out: None = None) -> _Array2D[np.int_ | Any]: ...
@overload
def outer(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, out: None = None) -> _Array2D[np.float64 | Any]: ...
@overload
def outer(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, out: None = None) -> _Array2D[np.complex128 | Any]: ...
@overload
def outer(a: _ArrayLikeTD64_co, b: _ArrayLikeTD64_co, out: None = None) -> _Array2D[np.timedelta64 | Any]: ...
@overload
def outer[ArrayT: np.ndarray](a: _ArrayLikeNumber_co | _ArrayLikeTD64_co, b: _ArrayLikeNumber_co | _ArrayLikeTD64_co, out: ArrayT) -> ArrayT: ...

# keep in sync with numpy.linalg._linalg.tensordot (ignoring `/, *`)
@overload
def tensordot(
    a: _ArrayLike[_AnyNumericScalarT], b: _ArrayLike[_AnyNumericScalarT], axes: int | tuple[_ShapeLike, _ShapeLike] = 2
) -> NDArray[_AnyNumericScalarT]: ...
@overload
def tensordot(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co, axes: int | tuple[_ShapeLike, _ShapeLike] = 2) -> NDArray[np.bool]: ...
@overload
def tensordot(
    a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, axes: int | tuple[_ShapeLike, _ShapeLike] = 2
) -> NDArray[np.int_ | Any]: ...
@overload
def tensordot(
    a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, axes: int | tuple[_ShapeLike, _ShapeLike] = 2
) -> NDArray[np.float64 | Any]: ...
@overload
def tensordot(
    a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, axes: int | tuple[_ShapeLike, _ShapeLike] = 2
) -> NDArray[np.complex128 | Any]: ...

#
@overload
def cross(
    a: _ArrayLike[_AnyNumericScalarT],
    b: _ArrayLike[_AnyNumericScalarT],
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NDArray[_AnyNumericScalarT]: ...
@overload
def cross(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NDArray[np.int_ | Any]: ...
@overload
def cross(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NDArray[np.float64 | Any]: ...
@overload
def cross(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = None,
) -> NDArray[np.complex128 | Any]: ...

#
@overload
def roll[ArrayT: np.ndarray](a: ArrayT, shift: _ShapeLike, axis: _ShapeLike | None = None) -> ArrayT: ...
@overload
def roll[ScalarT: np.generic](a: _ArrayLike[ScalarT], shift: _ShapeLike, axis: _ShapeLike | None = None) -> NDArray[ScalarT]: ...
@overload
def roll(a: ArrayLike, shift: _ShapeLike, axis: _ShapeLike | None = None) -> NDArray[Any]: ...

#
def rollaxis[ArrayT: np.ndarray](a: ArrayT, axis: int, start: int = 0) -> ArrayT: ...
def moveaxis[ArrayT: np.ndarray](a: ArrayT, source: _ShapeLike, destination: _ShapeLike) -> ArrayT: ...
def normalize_axis_tuple(
    axis: int | Iterable[int],
    ndim: int,
    argname: str | None = None,
    allow_duplicate: py_bool | None = False,
) -> tuple[int, ...]: ...

#
@overload  # 0d, dtype=int (default), sparse=False (default)
def indices(dimensions: tuple[()], dtype: type[int] = int, sparse: L[False] = False) -> _Array1D[np.intp]: ...
@overload  # 0d, dtype=<irrelevant>, sparse=True
def indices(dimensions: tuple[()], dtype: DTypeLike | None = int, *, sparse: L[True]) -> tuple[()]: ...
@overload  # 0d, dtype=<known>, sparse=False (default)
def indices[ScalarT: np.generic](dimensions: tuple[()], dtype: _DTypeLike[ScalarT], sparse: L[False] = False) -> _Array1D[ScalarT]: ...
@overload  # 0d, dtype=<unknown>, sparse=False (default)
def indices(dimensions: tuple[()], dtype: DTypeLike, sparse: L[False] = False) -> _Array1D[Any]: ...
@overload  # 1d, dtype=int (default), sparse=False (default)
def indices(dimensions: tuple[int], dtype: type[int] = int, sparse: L[False] = False) -> _Array2D[np.intp]: ...
@overload  # 1d, dtype=int (default), sparse=True
def indices(dimensions: tuple[int], dtype: type[int] = int, *, sparse: L[True]) -> tuple[_Array1D[np.intp]]: ...
@overload  # 1d, dtype=<known>, sparse=False (default)
def indices[ScalarT: np.generic](dimensions: tuple[int], dtype: _DTypeLike[ScalarT], sparse: L[False] = False) -> _Array2D[ScalarT]: ...
@overload  # 1d, dtype=<known>, sparse=True
def indices[ScalarT: np.generic](dimensions: tuple[int], dtype: _DTypeLike[ScalarT], sparse: L[True]) -> tuple[_Array1D[ScalarT]]: ...
@overload  # 1d, dtype=<unknown>, sparse=False (default)
def indices(dimensions: tuple[int], dtype: DTypeLike, sparse: L[False] = False) -> _Array2D[Any]: ...
@overload  # 1d, dtype=<unknown>, sparse=True
def indices(dimensions: tuple[int], dtype: DTypeLike, sparse: L[True]) -> tuple[_Array1D[Any]]: ...
@overload  # 2d, dtype=int (default), sparse=False (default)
def indices(dimensions: tuple[int, int], dtype: type[int] = int, sparse: L[False] = False) -> _Array3D[np.intp]: ...
@overload  # 2d, dtype=int (default), sparse=True
def indices(
    dimensions: tuple[int, int], dtype: type[int] = int, *, sparse: L[True]
) -> tuple[_Array2D[np.intp], _Array2D[np.intp]]: ...
@overload  # 2d, dtype=<known>, sparse=False (default)
def indices[ScalarT: np.generic](dimensions: tuple[int, int], dtype: _DTypeLike[ScalarT], sparse: L[False] = False) -> _Array3D[ScalarT]: ...
@overload  # 2d, dtype=<known>, sparse=True
def indices[ScalarT: np.generic](
    dimensions: tuple[int, int], dtype: _DTypeLike[ScalarT], sparse: L[True]
) -> tuple[_Array2D[ScalarT], _Array2D[ScalarT]]: ...
@overload  # 2d, dtype=<unknown>, sparse=False (default)
def indices(dimensions: tuple[int, int], dtype: DTypeLike, sparse: L[False] = False) -> _Array3D[Any]: ...
@overload  # 2d, dtype=<unknown>, sparse=True
def indices(dimensions: tuple[int, int], dtype: DTypeLike, sparse: L[True]) -> tuple[_Array2D[Any], _Array2D[Any]]: ...
@overload  # ?d, dtype=int (default), sparse=False (default)
def indices(dimensions: Sequence[int], dtype: type[int] = int, sparse: L[False] = False) -> NDArray[np.intp]: ...
@overload  # ?d, dtype=int (default), sparse=True
def indices(dimensions: Sequence[int], dtype: type[int] = int, *, sparse: L[True]) -> tuple[NDArray[np.intp], ...]: ...
@overload  # ?d, dtype=<known>, sparse=False (default)
def indices[ScalarT: np.generic](dimensions: Sequence[int], dtype: _DTypeLike[ScalarT], sparse: L[False] = False) -> NDArray[ScalarT]: ...
@overload  # ?d, dtype=<known>, sparse=True
def indices[ScalarT: np.generic](dimensions: Sequence[int], dtype: _DTypeLike[ScalarT], sparse: L[True]) -> tuple[NDArray[ScalarT], ...]: ...
@overload  # ?d, dtype=<unknown>, sparse=False (default)
def indices(dimensions: Sequence[int], dtype: DTypeLike, sparse: L[False] = False) -> ndarray: ...
@overload  # ?d, dtype=<unknown>, sparse=True
def indices(dimensions: Sequence[int], dtype: DTypeLike, sparse: L[True]) -> tuple[ndarray, ...]: ...

#
def fromfunction[ReturnT](
    function: Callable[..., ReturnT],
    shape: Sequence[int],
    *,
    dtype: DTypeLike | None = float,
    like: _SupportsArrayFunc | None = None,
    **kwargs: object,
) -> ReturnT: ...

#
def isscalar(element: object) -> TypeGuard[generic | complex | str | bytes | memoryview]: ...

#
def binary_repr(num: SupportsIndex, width: int | None = None) -> str: ...
def base_repr(number: SupportsAbs[float], base: float = 2, padding: SupportsIndex | None = 0) -> str: ...

#
@overload  # dtype: None (default)
def identity(n: int, dtype: None = None, *, like: _SupportsArrayFunc | None = None) -> _Array2D[np.float64]: ...
@overload  # dtype: known scalar type
def identity[ScalarT: np.generic](n: int, dtype: _DTypeLike[ScalarT], *, like: _SupportsArrayFunc | None = None) -> _Array2D[ScalarT]: ...
@overload  # dtype: like bool
def identity(n: int, dtype: _DTypeLikeBool, *, like: _SupportsArrayFunc | None = None) -> _Array2D[np.bool]: ...
@overload  # dtype: like int_
def identity(n: int, dtype: _DTypeLikeInt, *, like: _SupportsArrayFunc | None = None) -> _Array2D[np.int_ | Any]: ...
@overload  # dtype: like float64
def identity(n: int, dtype: _DTypeLikeFloat64, *, like: _SupportsArrayFunc | None = None) -> _Array2D[np.float64 | Any]: ...
@overload  # dtype: like complex128
def identity(n: int, dtype: _DTypeLikeComplex128, *, like: _SupportsArrayFunc | None = None) -> _Array2D[np.complex128 | Any]: ...
@overload  # dtype: unknown
def identity(n: int, dtype: DTypeLike, *, like: _SupportsArrayFunc | None = None) -> _Array2D[Incomplete]: ...

#
def allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> py_bool: ...

#
@overload  # scalar, scalar
def isclose(
    a: _NumberLike_co,
    b: _NumberLike_co,
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> np.bool: ...
@overload  # known shape, same shape or scalar
def isclose[ShapeT: _Shape](
    a: np.ndarray[ShapeT],
    b: np.ndarray[ShapeT] | _NumberLike_co,
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # same shape or scalar, known shape
def isclose[ShapeT: _Shape](
    a: np.ndarray[ShapeT] | _NumberLike_co,
    b: np.ndarray[ShapeT],
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> np.ndarray[ShapeT, np.dtype[np.bool]]: ...
@overload  # 1d sequence, <=1d array-like
def isclose(
    a: Sequence[_NumberLike_co],
    b: Sequence[_NumberLike_co] | _NumberLike_co | np.ndarray[tuple[int]],
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.bool]]: ...
@overload  # <=1d array-like, 1d sequence
def isclose(
    a: Sequence[_NumberLike_co] | _NumberLike_co | np.ndarray[tuple[int]],
    b: Sequence[_NumberLike_co],
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.bool]]: ...
@overload  # 2d sequence, <=2d array-like
def isclose(
    a: Sequence[Sequence[_NumberLike_co]],
    b: Sequence[Sequence[_NumberLike_co]] | Sequence[_NumberLike_co] | _NumberLike_co | np.ndarray[tuple[int] | tuple[int, int]],
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.bool]]: ...
@overload  # <=2d array-like, 2d sequence
def isclose(
    b: Sequence[Sequence[_NumberLike_co]] | Sequence[_NumberLike_co] | _NumberLike_co | np.ndarray[tuple[int] | tuple[int, int]],
    a: Sequence[Sequence[_NumberLike_co]],
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> np.ndarray[tuple[int], np.dtype[np.bool]]: ...
@overload  # unknown shape, unknown shape
def isclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = 1e-5,
    atol: ArrayLike = 1e-8,
    equal_nan: py_bool = False,
) -> NDArray[np.bool] | Any: ...

#
def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan: py_bool = False) -> py_bool: ...
def array_equiv(a1: ArrayLike, a2: ArrayLike) -> py_bool: ...

#
@overload
def astype[ShapeT: _Shape, ScalarT: np.generic](
    x: ndarray[ShapeT],
    dtype: _DTypeLike[ScalarT],
    /,
    *,
    copy: py_bool = True,
    device: L["cpu"] | None = None,
) -> ndarray[ShapeT, dtype[ScalarT]]: ...
@overload
def astype[ShapeT: _Shape](
    x: ndarray[ShapeT],
    dtype: DTypeLike | None,
    /,
    *,
    copy: py_bool = True,
    device: L["cpu"] | None = None,
) -> ndarray[ShapeT]: ...
