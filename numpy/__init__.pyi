import builtins
import os
import sys
import datetime as dt
from abc import abstractmethod
from types import TracebackType
from contextlib import ContextDecorator

from numpy.core._internal import _ctypes
from numpy.typing import (
    # Arrays
    ArrayLike,
    _ArrayND,
    _ArrayOrScalar,
    _SupportsArray,
    _NestedSequence,
    _RecursiveSequence,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeNumber_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
    _ArrayLikeObject_co,

    # DTypes
    DTypeLike,
    _SupportsDType,
    _VoidDTypeLike,

    # Shapes
    _Shape,
    _ShapeLike,

    # Scalars
    _CharLike_co,
    _BoolLike_co,
    _IntLike_co,
    _FloatLike_co,
    _ComplexLike_co,
    _TD64Like_co,
    _NumberLike_co,

    # `number` precision
    NBitBase,
    _256Bit,
    _128Bit,
    _96Bit,
    _80Bit,
    _64Bit,
    _32Bit,
    _16Bit,
    _8Bit,
    _NBitByte,
    _NBitShort,
    _NBitIntC,
    _NBitIntP,
    _NBitInt,
    _NBitLongLong,
    _NBitHalf,
    _NBitSingle,
    _NBitDouble,
    _NBitLongDouble,

    # Character codes
    _BoolCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _Complex64Codes,
    _Complex128Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _IntPCodes,
    _IntCodes,
    _LongLongCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntPCodes,
    _UIntCodes,
    _ULongLongCodes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
    _DT64Codes,
    _TD64Codes,
    _StrCodes,
    _BytesCodes,
    _VoidCodes,
    _ObjectCodes,
)
from numpy.typing._callable import (
    _BoolOp,
    _BoolBitOp,
    _BoolSub,
    _BoolTrueDiv,
    _BoolMod,
    _BoolDivMod,
    _TD64Div,
    _IntTrueDiv,
    _UnsignedIntOp,
    _UnsignedIntBitOp,
    _UnsignedIntMod,
    _UnsignedIntDivMod,
    _SignedIntOp,
    _SignedIntBitOp,
    _SignedIntMod,
    _SignedIntDivMod,
    _FloatOp,
    _FloatMod,
    _FloatDivMod,
    _ComplexOp,
    _NumberOp,
    _ComparisonOp,
)

# NOTE: Numpy's mypy plugin is used for removing the types unavailable
# to the specific platform
from numpy.typing._extended_precision import (
    uint128 as uint128,
    uint256 as uint256,
    int128 as int128,
    int256 as int256,
    float80 as float80,
    float96 as float96,
    float128 as float128,
    float256 as float256,
    complex160 as complex160,
    complex192 as complex192,
    complex256 as complex256,
    complex512 as complex512,
)

from typing import (
    Any,
    ByteString,
    Callable,
    Container,
    Callable,
    Dict,
    Generic,
    IO,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    overload,
    Sequence,
    Sized,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    Text,
    Tuple,
    Type,
    TypeVar,
    Union,
)

if sys.version_info >= (3, 8):
    from typing import Literal, Protocol, SupportsIndex, Final
else:
    from typing_extensions import Literal, Protocol, SupportsIndex, Final

# Ensures that the stubs are picked up
from numpy import (
    char as char,
    ctypeslib as ctypeslib,
    emath as emath,
    fft as fft,
    lib as lib,
    linalg as linalg,
    ma as ma,
    matrixlib as matrixlib,
    polynomial as polynomial,
    random as random,
    rec as rec,
    testing as testing,
    version as version,
)

from numpy.core.function_base import (
    linspace as linspace,
    logspace as logspace,
    geomspace as geomspace,
)

from numpy.core.fromnumeric import (
    take as take,
    reshape as reshape,
    choose as choose,
    repeat as repeat,
    put as put,
    swapaxes as swapaxes,
    transpose as transpose,
    partition as partition,
    argpartition as argpartition,
    sort as sort,
    argsort as argsort,
    argmax as argmax,
    argmin as argmin,
    searchsorted as searchsorted,
    resize as resize,
    squeeze as squeeze,
    diagonal as diagonal,
    trace as trace,
    ravel as ravel,
    nonzero as nonzero,
    shape as shape,
    compress as compress,
    clip as clip,
    sum as sum,
    all as all,
    any as any,
    cumsum as cumsum,
    ptp as ptp,
    amax as amax,
    amin as amin,
    prod as prod,
    cumprod as cumprod,
    ndim as ndim,
    size as size,
    around as around,
    mean as mean,
    std as std,
    var as var,
)

from numpy.core._asarray import (
    asarray as asarray,
    asanyarray as asanyarray,
    ascontiguousarray as ascontiguousarray,
    asfortranarray as asfortranarray,
    require as require,
)

from numpy.core._type_aliases import (
    sctypes as sctypes,
    sctypeDict as sctypeDict,
)

from numpy.core._ufunc_config import (
    seterr as seterr,
    geterr as geterr,
    setbufsize as setbufsize,
    getbufsize as getbufsize,
    seterrcall as seterrcall,
    geterrcall as geterrcall,
    _SupportsWrite,
    _ErrKind,
    _ErrFunc,
    _ErrDictOptional,
)

from numpy.core.arrayprint import (
    set_printoptions as set_printoptions,
    get_printoptions as get_printoptions,
    array2string as array2string,
    format_float_scientific as format_float_scientific,
    format_float_positional as format_float_positional,
    array_repr as array_repr,
    array_str as array_str,
    set_string_function as set_string_function,
    printoptions as printoptions,
)

from numpy.core.einsumfunc import (
    einsum as einsum,
    einsum_path as einsum_path,
)

from numpy.core.numeric import (
    zeros_like as zeros_like,
    ones as ones,
    ones_like as ones_like,
    empty_like as empty_like,
    full as full,
    full_like as full_like,
    count_nonzero as count_nonzero,
    isfortran as isfortran,
    argwhere as argwhere,
    flatnonzero as flatnonzero,
    correlate as correlate,
    convolve as convolve,
    outer as outer,
    tensordot as tensordot,
    roll as roll,
    rollaxis as rollaxis,
    moveaxis as moveaxis,
    cross as cross,
    indices as indices,
    fromfunction as fromfunction,
    isscalar as isscalar,
    binary_repr as binary_repr,
    base_repr as base_repr,
    identity as identity,
    allclose as allclose,
    isclose as isclose,
    array_equal as array_equal,
    array_equiv as array_equiv,
)

from numpy.core.numerictypes import (
    maximum_sctype as maximum_sctype,
    issctype as issctype,
    obj2sctype as obj2sctype,
    issubclass_ as issubclass_,
    issubsctype as issubsctype,
    issubdtype as issubdtype,
    sctype2char as sctype2char,
    find_common_type as find_common_type,
)

from numpy.core.shape_base import (
    atleast_1d as atleast_1d,
    atleast_2d as atleast_2d,
    atleast_3d as atleast_3d,
    block as block,
    hstack as hstack,
    stack as stack,
    vstack as vstack,
)

from numpy.lib.index_tricks import (
    ravel_multi_index as ravel_multi_index,
    unravel_index as unravel_index,
    mgrid as mgrid,
    ogrid as ogrid,
    r_ as r_,
    c_ as c_,
    s_ as s_,
    index_exp as index_exp,
    ix_ as ix_,
    fill_diagonal as fill_diagonal,
    diag_indices as diag_indices,
    diag_indices_from as diag_indices_from,
)

from numpy.lib.ufunclike import (
    fix as fix,
    isposinf as isposinf,
    isneginf as isneginf,
)

__all__: List[str]
__path__: List[str]
__version__: str
__git_version__: str

DataSource: Any
MachAr: Any
ScalarType: Any
angle: Any
append: Any
apply_along_axis: Any
apply_over_axes: Any
arange: Any
array_split: Any
asarray_chkfinite: Any
asfarray: Any
asmatrix: Any
asscalar: Any
average: Any
bartlett: Any
bincount: Any
bitwise_not: Any
blackman: Any
bmat: Any
broadcast: Any
broadcast_arrays: Any
broadcast_to: Any
busday_count: Any
busday_offset: Any
busdaycalendar: Any
byte_bounds: Any
can_cast: Any
cast: Any
chararray: Any
column_stack: Any
common_type: Any
compare_chararrays: Any
concatenate: Any
conj: Any
copy: Any
copyto: Any
corrcoef: Any
cov: Any
cumproduct: Any
datetime_as_string: Any
datetime_data: Any
delete: Any
deprecate: Any
deprecate_with_doc: Any
diag: Any
diagflat: Any
diff: Any
digitize: Any
disp: Any
divide: Any
dot: Any
dsplit: Any
dstack: Any
ediff1d: Any
expand_dims: Any
extract: Any
def eye(
    N: int,
    M: Optional[int] = ...,
    k: int = ...,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: Optional[ArrayLike] = ...
) -> ndarray[Any, Any]: ...
finfo: Any
flip: Any
fliplr: Any
flipud: Any
format_parser: Any
frombuffer: Any
fromfile: Any
fromiter: Any
frompyfunc: Any
fromregex: Any
fromstring: Any
genfromtxt: Any
get_include: Any
geterrobj: Any
gradient: Any
hamming: Any
hanning: Any
histogram: Any
histogram2d: Any
histogram_bin_edges: Any
histogramdd: Any
hsplit: Any
i0: Any
iinfo: Any
imag: Any
in1d: Any
info: Any
inner: Any
insert: Any
interp: Any
intersect1d: Any
is_busday: Any
iscomplex: Any
iscomplexobj: Any
isin: Any
isreal: Any
isrealobj: Any
iterable: Any
kaiser: Any
kron: Any
lexsort: Any
load: Any
loads: Any
loadtxt: Any
lookfor: Any
mafromtxt: Any
mask_indices: Any
mat: Any
matrix: Any
max: Any
may_share_memory: Any
median: Any
memmap: Any
meshgrid: Any
min: Any
min_scalar_type: Any
mintypecode: Any
mod: Any
msort: Any
nan_to_num: Any
nanargmax: Any
nanargmin: Any
nancumprod: Any
nancumsum: Any
nanmax: Any
nanmean: Any
nanmedian: Any
nanmin: Any
nanpercentile: Any
nanprod: Any
nanquantile: Any
nanstd: Any
nansum: Any
nanvar: Any
nbytes: Any
ndfromtxt: Any
nditer: Any
nested_iters: Any
newaxis: Any
numarray: Any
packbits: Any
pad: Any
percentile: Any
piecewise: Any
place: Any
poly: Any
poly1d: Any
polyadd: Any
polyder: Any
polydiv: Any
polyfit: Any
polyint: Any
polymul: Any
polysub: Any
polyval: Any
product: Any
promote_types: Any
put_along_axis: Any
putmask: Any
quantile: Any
real: Any
real_if_close: Any
recarray: Any
recfromcsv: Any
recfromtxt: Any
record: Any
result_type: Any
roots: Any
rot90: Any
round: Any
round_: Any
row_stack: Any
save: Any
savetxt: Any
savez: Any
savez_compressed: Any
select: Any
setdiff1d: Any
seterrobj: Any
setxor1d: Any
shares_memory: Any
show_config: Any
sinc: Any
sort_complex: Any
source: Any
split: Any
take_along_axis: Any
tile: Any
trapz: Any
tri: Any
tril: Any
tril_indices: Any
tril_indices_from: Any
trim_zeros: Any
triu: Any
triu_indices: Any
triu_indices_from: Any
typeDict: Any
typecodes: Any
typename: Any
union1d: Any
unique: Any
unpackbits: Any
unwrap: Any
vander: Any
vdot: Any
vectorize: Any
vsplit: Any
where: Any
who: Any

_NdArraySubClass = TypeVar("_NdArraySubClass", bound=ndarray)
_DTypeScalar_co = TypeVar("_DTypeScalar_co", covariant=True, bound=generic)
_ByteOrder = Literal["S", "<", ">", "=", "|", "L", "B", "N", "I"]

class dtype(Generic[_DTypeScalar_co]):
    names: Optional[Tuple[str, ...]]
    # Overload for subclass of generic
    @overload
    def __new__(
        cls,
        dtype: Type[_DTypeScalar_co],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    # Overloads for string aliases, Python types, and some assorted
    # other special cases. Order is sometimes important because of the
    # subtype relationships
    #
    # bool < int < float < complex
    #
    # so we have to make sure the overloads for the narrowest type is
    # first.
    # Builtin types
    @overload
    def __new__(cls, dtype: Type[bool], align: bool = ..., copy: bool = ...) -> dtype[bool_]: ...
    @overload
    def __new__(cls, dtype: Type[int], align: bool = ..., copy: bool = ...) -> dtype[int_]: ...
    @overload
    def __new__(cls, dtype: Optional[Type[float]], align: bool = ..., copy: bool = ...) -> dtype[float_]: ...
    @overload
    def __new__(cls, dtype: Type[complex], align: bool = ..., copy: bool = ...) -> dtype[complex_]: ...
    @overload
    def __new__(cls, dtype: Type[str], align: bool = ..., copy: bool = ...) -> dtype[str_]: ...
    @overload
    def __new__(cls, dtype: Type[bytes], align: bool = ..., copy: bool = ...) -> dtype[bytes_]: ...

    # `unsignedinteger` string-based representations
    @overload
    def __new__(cls, dtype: _UInt8Codes, align: bool = ..., copy: bool = ...) -> dtype[uint8]: ...
    @overload
    def __new__(cls, dtype: _UInt16Codes, align: bool = ..., copy: bool = ...) -> dtype[uint16]: ...
    @overload
    def __new__(cls, dtype: _UInt32Codes, align: bool = ..., copy: bool = ...) -> dtype[uint32]: ...
    @overload
    def __new__(cls, dtype: _UInt64Codes, align: bool = ..., copy: bool = ...) -> dtype[uint64]: ...
    @overload
    def __new__(cls, dtype: _UByteCodes, align: bool = ..., copy: bool = ...) -> dtype[ubyte]: ...
    @overload
    def __new__(cls, dtype: _UShortCodes, align: bool = ..., copy: bool = ...) -> dtype[ushort]: ...
    @overload
    def __new__(cls, dtype: _UIntCCodes, align: bool = ..., copy: bool = ...) -> dtype[uintc]: ...
    @overload
    def __new__(cls, dtype: _UIntPCodes, align: bool = ..., copy: bool = ...) -> dtype[uintp]: ...
    @overload
    def __new__(cls, dtype: _UIntCodes, align: bool = ..., copy: bool = ...) -> dtype[uint]: ...
    @overload
    def __new__(cls, dtype: _ULongLongCodes, align: bool = ..., copy: bool = ...) -> dtype[ulonglong]: ...

    # `signedinteger` string-based representations
    @overload
    def __new__(cls, dtype: _Int8Codes, align: bool = ..., copy: bool = ...) -> dtype[int8]: ...
    @overload
    def __new__(cls, dtype: _Int16Codes, align: bool = ..., copy: bool = ...) -> dtype[int16]: ...
    @overload
    def __new__(cls, dtype: _Int32Codes, align: bool = ..., copy: bool = ...) -> dtype[int32]: ...
    @overload
    def __new__(cls, dtype: _Int64Codes, align: bool = ..., copy: bool = ...) -> dtype[int64]: ...
    @overload
    def __new__(cls, dtype: _ByteCodes, align: bool = ..., copy: bool = ...) -> dtype[byte]: ...
    @overload
    def __new__(cls, dtype: _ShortCodes, align: bool = ..., copy: bool = ...) -> dtype[short]: ...
    @overload
    def __new__(cls, dtype: _IntCCodes, align: bool = ..., copy: bool = ...) -> dtype[intc]: ...
    @overload
    def __new__(cls, dtype: _IntPCodes, align: bool = ..., copy: bool = ...) -> dtype[intp]: ...
    @overload
    def __new__(cls, dtype: _IntCodes, align: bool = ..., copy: bool = ...) -> dtype[int_]: ...
    @overload
    def __new__(cls, dtype: _LongLongCodes, align: bool = ..., copy: bool = ...) -> dtype[longlong]: ...

    # `floating` string-based representations
    @overload
    def __new__(cls, dtype: _Float16Codes, align: bool = ..., copy: bool = ...) -> dtype[float16]: ...
    @overload
    def __new__(cls, dtype: _Float32Codes, align: bool = ..., copy: bool = ...) -> dtype[float32]: ...
    @overload
    def __new__(cls, dtype: _Float64Codes, align: bool = ..., copy: bool = ...) -> dtype[float64]: ...
    @overload
    def __new__(cls, dtype: _HalfCodes, align: bool = ..., copy: bool = ...) -> dtype[half]: ...
    @overload
    def __new__(cls, dtype: _SingleCodes, align: bool = ..., copy: bool = ...) -> dtype[single]: ...
    @overload
    def __new__(cls, dtype: _DoubleCodes, align: bool = ..., copy: bool = ...) -> dtype[double]: ...
    @overload
    def __new__(cls, dtype: _LongDoubleCodes, align: bool = ..., copy: bool = ...) -> dtype[longdouble]: ...

    # `complexfloating` string-based representations
    @overload
    def __new__(cls, dtype: _Complex64Codes, align: bool = ..., copy: bool = ...) -> dtype[complex64]: ...
    @overload
    def __new__(cls, dtype: _Complex128Codes, align: bool = ..., copy: bool = ...) -> dtype[complex128]: ...
    @overload
    def __new__(cls, dtype: _CSingleCodes, align: bool = ..., copy: bool = ...) -> dtype[csingle]: ...
    @overload
    def __new__(cls, dtype: _CDoubleCodes, align: bool = ..., copy: bool = ...) -> dtype[cdouble]: ...
    @overload
    def __new__(cls, dtype: _CLongDoubleCodes, align: bool = ..., copy: bool = ...) -> dtype[clongdouble]: ...

    # Miscellaneous string-based representations
    @overload
    def __new__(cls, dtype: _BoolCodes, align: bool = ..., copy: bool = ...) -> dtype[bool_]: ...
    @overload
    def __new__(cls, dtype: _TD64Codes, align: bool = ..., copy: bool = ...) -> dtype[timedelta64]: ...
    @overload
    def __new__(cls, dtype: _DT64Codes, align: bool = ..., copy: bool = ...) -> dtype[datetime64]: ...
    @overload
    def __new__(cls, dtype: _StrCodes, align: bool = ..., copy: bool = ...) -> dtype[str_]: ...
    @overload
    def __new__(cls, dtype: _BytesCodes, align: bool = ..., copy: bool = ...) -> dtype[bytes_]: ...
    @overload
    def __new__(cls, dtype: _VoidCodes, align: bool = ..., copy: bool = ...) -> dtype[void]: ...
    @overload
    def __new__(cls, dtype: _ObjectCodes, align: bool = ..., copy: bool = ...) -> dtype[object_]: ...

    # dtype of a dtype is the same dtype
    @overload
    def __new__(
        cls,
        dtype: dtype[_DTypeScalar_co],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    @overload
    def __new__(
        cls,
        dtype: _SupportsDType[dtype[_DTypeScalar_co]],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    # Handle strings that can't be expressed as literals; i.e. s1, s2, ...
    @overload
    def __new__(
        cls,
        dtype: str,
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[Any]: ...
    # Catchall overload
    @overload
    def __new__(
        cls,
        dtype: _VoidDTypeLike,
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[void]: ...
    def __eq__(self, other: DTypeLike) -> bool: ...
    def __ne__(self, other: DTypeLike) -> bool: ...
    def __gt__(self, other: DTypeLike) -> bool: ...
    def __ge__(self, other: DTypeLike) -> bool: ...
    def __lt__(self, other: DTypeLike) -> bool: ...
    def __le__(self, other: DTypeLike) -> bool: ...
    @property
    def alignment(self) -> int: ...
    @property
    def base(self: _DType) -> _DType: ...
    @property
    def byteorder(self) -> str: ...
    @property
    def char(self) -> str: ...
    @property
    def descr(self) -> List[Union[Tuple[str, str], Tuple[str, str, _Shape]]]: ...
    @property
    def fields(
        self,
    ) -> Optional[Mapping[str, Union[Tuple[dtype[Any], int], Tuple[dtype[Any], int, Any]]]]: ...
    @property
    def flags(self) -> int: ...
    @property
    def hasobject(self) -> bool: ...
    @property
    def isbuiltin(self) -> int: ...
    @property
    def isnative(self) -> bool: ...
    @property
    def isalignedstruct(self) -> bool: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def kind(self) -> str: ...
    @property
    def metadata(self) -> Optional[Mapping[str, Any]]: ...
    @property
    def name(self) -> str: ...
    @property
    def num(self) -> int: ...
    @property
    def shape(self) -> _Shape: ...
    @property
    def ndim(self) -> int: ...
    @property
    def subdtype(self: _DType) -> Optional[Tuple[_DType, _Shape]]: ...
    def newbyteorder(self: _DType, __new_order: _ByteOrder = ...) -> _DType: ...
    # Leave str and type for end to avoid having to use `builtins.str`
    # everywhere. See https://github.com/python/mypy/issues/3775
    @property
    def str(self) -> builtins.str: ...
    @property
    def type(self) -> Type[_DTypeScalar_co]: ...

class _flagsobj:
    aligned: bool
    updateifcopy: bool
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
    def __getitem__(self, key: str) -> bool: ...
    def __setitem__(self, key: str, value: bool) -> None: ...

_ArrayLikeInt = Union[
    int,
    integer,
    Sequence[Union[int, integer]],
    Sequence[Sequence[Any]],  # TODO: wait for support for recursive types
    ndarray
]

_FlatIterSelf = TypeVar("_FlatIterSelf", bound=flatiter)

class flatiter(Generic[_NdArraySubClass]):
    @property
    def base(self) -> _NdArraySubClass: ...
    @property
    def coords(self) -> _Shape: ...
    @property
    def index(self) -> int: ...
    def copy(self) -> _NdArraySubClass: ...
    def __iter__(self: _FlatIterSelf) -> _FlatIterSelf: ...
    def __next__(self: flatiter[ndarray[Any, dtype[_ScalarType]]]) -> _ScalarType: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(
        self: flatiter[ndarray[Any, dtype[_ScalarType]]],
        key: Union[int, integer],
    ) -> _ScalarType: ...
    @overload
    def __getitem__(
        self, key: Union[_ArrayLikeInt, slice, ellipsis],
    ) -> _NdArraySubClass: ...
    @overload
    def __array__(self: flatiter[ndarray[Any, _DType]], __dtype: None = ...) -> ndarray[Any, _DType]: ...
    @overload
    def __array__(self, __dtype: DTypeLike) -> ndarray[Any, dtype[Any]]: ...

_OrderKACF = Optional[Literal["K", "A", "C", "F"]]
_OrderACF = Optional[Literal["A", "C", "F"]]
_OrderCF = Optional[Literal["C", "F"]]

_ModeKind = Literal["raise", "wrap", "clip"]
_PartitionKind = Literal["introselect"]
_SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
_SortSide = Literal["left", "right"]

_ArrayLikeBool = Union[_BoolLike_co, Sequence[_BoolLike_co], ndarray]
_ArrayLikeIntOrBool = Union[
    _IntLike_co,
    ndarray,
    Sequence[_IntLike_co],
    Sequence[Sequence[Any]],  # TODO: wait for support for recursive types
]

_ArraySelf = TypeVar("_ArraySelf", bound=_ArrayOrScalarCommon)

class _ArrayOrScalarCommon:
    @property
    def T(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def data(self) -> memoryview: ...
    @property
    def flags(self) -> _flagsobj: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __copy__(self: _ArraySelf) -> _ArraySelf: ...
    def __deepcopy__(self: _ArraySelf, __memo: Optional[dict] = ...) -> _ArraySelf: ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def astype(
        self: _ArraySelf,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        casting: _Casting = ...,
        subok: bool = ...,
        copy: bool = ...,
    ) -> _ArraySelf: ...
    def copy(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...
    def dump(self, file: str) -> None: ...
    def dumps(self) -> bytes: ...
    def flatten(self, order: _OrderKACF = ...) -> ndarray: ...
    def getfield(
        self: _ArraySelf, dtype: DTypeLike, offset: int = ...
    ) -> _ArraySelf: ...
    def ravel(self, order: _OrderKACF = ...) -> ndarray: ...
    @overload
    def reshape(
        self, __shape: Sequence[int], *, order: _OrderACF = ...
    ) -> ndarray: ...
    @overload
    def reshape(
        self, *shape: int, order: _OrderACF = ...
    ) -> ndarray: ...
    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...
    # NOTE: `tostring()` is deprecated and therefore excluded
    # def tostring(self, order=...): ...
    def tofile(
        self, fid: Union[IO[bytes], str, bytes, os.PathLike[Any]], sep: str = ..., format: str = ...
    ) -> None: ...
    # generics and 0d arrays return builtin scalars
    def tolist(self) -> Any: ...
    @overload
    def view(self, type: Type[_NdArraySubClass]) -> _NdArraySubClass: ...
    @overload
    def view(self: _ArraySelf, dtype: DTypeLike = ...) -> _ArraySelf: ...
    @overload
    def view(
        self, dtype: DTypeLike, type: Type[_NdArraySubClass]
    ) -> _NdArraySubClass: ...

    # TODO: Add proper signatures
    def __getitem__(self, key) -> Any: ...
    @property
    def __array_interface__(self): ...
    @property
    def __array_priority__(self): ...
    @property
    def __array_struct__(self): ...
    def __array_wrap__(array, context=...): ...
    def __setstate__(self, __state): ...
    # a `bool_` is returned when `keepdims=True` and `self` is a 0d array
    @overload
    def all(
        self, axis: None = ..., out: None = ..., keepdims: Literal[False] = ...
    ) -> bool_: ...
    @overload
    def all(
        self, axis: Optional[_ShapeLike] = ..., out: None = ..., keepdims: bool = ...
    ) -> Union[bool_, ndarray]: ...
    @overload
    def all(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def any(
        self, axis: None = ..., out: None = ..., keepdims: Literal[False] = ...
    ) -> bool_: ...
    @overload
    def any(
        self, axis: Optional[_ShapeLike] = ..., out: None = ..., keepdims: bool = ...
    ) -> Union[bool_, ndarray]: ...
    @overload
    def any(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def argmax(self, axis: None = ..., out: None = ...) -> intp: ...
    @overload
    def argmax(
        self, axis: _ShapeLike = ..., out: None = ...
    ) -> Union[ndarray, intp]: ...
    @overload
    def argmax(
        self, axis: Optional[_ShapeLike] = ..., out: _NdArraySubClass = ...
    ) -> _NdArraySubClass: ...
    @overload
    def argmin(self, axis: None = ..., out: None = ...) -> intp: ...
    @overload
    def argmin(
        self, axis: _ShapeLike = ..., out: None = ...
    ) -> Union[ndarray, intp]: ...
    @overload
    def argmin(
        self, axis: Optional[_ShapeLike] = ..., out: _NdArraySubClass = ...
    ) -> _NdArraySubClass: ...
    def argsort(
        self,
        axis: Optional[int] = ...,
        kind: Optional[_SortKind] = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> ndarray: ...
    @overload
    def choose(
        self, choices: ArrayLike, out: None = ..., mode: _ModeKind = ...,
    ) -> ndarray: ...
    @overload
    def choose(
        self, choices: ArrayLike, out: _NdArraySubClass = ..., mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: Optional[ArrayLike] = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> Union[number, ndarray]: ...
    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> Union[number, ndarray]: ...
    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: Optional[ArrayLike] = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass: ...
    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass: ...
    @overload
    def compress(
        self, a: ArrayLike, axis: Optional[int] = ..., out: None = ...,
    ) -> ndarray: ...
    @overload
    def compress(
        self, a: ArrayLike, axis: Optional[int] = ..., out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    def conj(self: _ArraySelf) -> _ArraySelf: ...
    def conjugate(self: _ArraySelf) -> _ArraySelf: ...
    @overload
    def cumprod(
        self, axis: Optional[int] = ..., dtype: DTypeLike = ..., out: None = ...,
    ) -> ndarray: ...
    @overload
    def cumprod(
        self,
        axis: Optional[int] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def cumsum(
        self, axis: Optional[int] = ..., dtype: DTypeLike = ..., out: None = ...,
    ) -> ndarray: ...
    @overload
    def cumsum(
        self,
        axis: Optional[int] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def max(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: Literal[False] = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> number: ...
    @overload
    def max(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def max(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def mean(
        self,
        axis: None = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: Literal[False] = ...,
    ) -> number: ...
    @overload
    def mean(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def mean(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def min(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: Literal[False] = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> number: ...
    @overload
    def min(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def min(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> _NdArraySubClass: ...
    def newbyteorder(self: _ArraySelf, __new_order: _ByteOrder = ...) -> _ArraySelf: ...
    @overload
    def prod(
        self,
        axis: None = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: Literal[False] = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> number: ...
    @overload
    def prod(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def prod(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def ptp(
        self, axis: None = ..., out: None = ..., keepdims: Literal[False] = ...,
    ) -> number: ...
    @overload
    def ptp(
        self, axis: Optional[_ShapeLike] = ..., out: None = ..., keepdims: bool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def ptp(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    def repeat(
        self, repeats: _ArrayLikeIntOrBool, axis: Optional[int] = ...
    ) -> ndarray: ...
    @overload
    def round(self: _ArraySelf, decimals: int = ..., out: None = ...) -> _ArraySelf: ...
    @overload
    def round(
        self, decimals: int = ..., out: _NdArraySubClass = ...
    ) -> _NdArraySubClass: ...
    @overload
    def std(
        self,
        axis: None = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: Literal[False] = ...,
    ) -> number: ...
    @overload
    def std(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def std(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def sum(
        self,
        axis: None = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: Literal[False] = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> number: ...
    @overload
    def sum(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def sum(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def take(
        self,
        indices: _IntLike_co,
        axis: Optional[int] = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> generic: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeIntOrBool,
        axis: Optional[int] = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeIntOrBool,
        axis: Optional[int] = ...,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def var(
        self,
        axis: None = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: Literal[False] = ...,
    ) -> number: ...
    @overload
    def var(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def var(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

_DType = TypeVar("_DType", bound=dtype[Any])
_DType_co = TypeVar("_DType_co", covariant=True, bound=dtype[Any])

# TODO: Set the `bound` to something more suitable once we
# have proper shape support
_ShapeType = TypeVar("_ShapeType", bound=Any)
_NumberType = TypeVar("_NumberType", bound=number[Any])
_BufferType = Union[ndarray, bytes, bytearray, memoryview]

_T = TypeVar("_T")
_2Tuple = Tuple[_T, _T]
_Casting = Literal["no", "equiv", "safe", "same_kind", "unsafe"]

_ArrayUInt_co = _ArrayND[Union[bool_, unsignedinteger[Any]]]
_ArrayInt_co = _ArrayND[Union[bool_, integer[Any]]]
_ArrayFloat_co = _ArrayND[Union[bool_, integer[Any], floating[Any]]]
_ArrayComplex_co = _ArrayND[Union[bool_, integer[Any], floating[Any], complexfloating[Any, Any]]]
_ArrayNumber_co = _ArrayND[Union[bool_, number[Any]]]
_ArrayTD64_co = _ArrayND[Union[bool_, integer[Any], timedelta64]]

class ndarray(_ArrayOrScalarCommon, Generic[_ShapeType, _DType_co]):
    @property
    def base(self) -> Optional[ndarray]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @real.setter
    def real(self, value: ArrayLike) -> None: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    @imag.setter
    def imag(self, value: ArrayLike) -> None: ...
    def __new__(
        cls: Type[_ArraySelf],
        shape: Sequence[int],
        dtype: DTypeLike = ...,
        buffer: _BufferType = ...,
        offset: int = ...,
        strides: _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> _ArraySelf: ...
    @overload
    def __array__(self, __dtype: None = ...) -> ndarray[Any, _DType_co]: ...
    @overload
    def __array__(self, __dtype: DTypeLike) -> ndarray[Any, dtype[Any]]: ...
    @property
    def ctypes(self) -> _ctypes: ...
    @property
    def shape(self) -> _Shape: ...
    @shape.setter
    def shape(self, value: _ShapeLike): ...
    @property
    def strides(self) -> _Shape: ...
    @strides.setter
    def strides(self, value: _ShapeLike): ...
    def byteswap(self: _ArraySelf, inplace: bool = ...) -> _ArraySelf: ...
    def fill(self, value: Any) -> None: ...
    @property
    def flat(self: _NdArraySubClass) -> flatiter[_NdArraySubClass]: ...
    @overload
    def item(self, *args: int) -> Any: ...
    @overload
    def item(self, __args: Tuple[int, ...]) -> Any: ...
    @overload
    def itemset(self, __value: Any) -> None: ...
    @overload
    def itemset(self, __item: _ShapeLike, __value: Any) -> None: ...
    @overload
    def resize(self, __new_shape: Sequence[int], *, refcheck: bool = ...) -> None: ...
    @overload
    def resize(self, *new_shape: int, refcheck: bool = ...) -> None: ...
    def setflags(
        self, write: bool = ..., align: bool = ..., uic: bool = ...
    ) -> None: ...
    def squeeze(
        self: _ArraySelf, axis: Union[int, Tuple[int, ...]] = ...
    ) -> _ArraySelf: ...
    def swapaxes(self: _ArraySelf, axis1: int, axis2: int) -> _ArraySelf: ...
    @overload
    def transpose(self: _ArraySelf, __axes: Sequence[int]) -> _ArraySelf: ...
    @overload
    def transpose(self: _ArraySelf, *axes: int) -> _ArraySelf: ...
    def argpartition(
        self,
        kth: _ArrayLikeIntOrBool,
        axis: Optional[int] = ...,
        kind: _PartitionKind = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> ndarray: ...
    def diagonal(
        self: _ArraySelf, offset: int = ..., axis1: int = ..., axis2: int = ...
    ) -> _ArraySelf: ...
    @overload
    def dot(self, b: ArrayLike, out: None = ...) -> Union[number, ndarray]: ...
    @overload
    def dot(self, b: ArrayLike, out: _NdArraySubClass = ...) -> _NdArraySubClass: ...
    # `nonzero()` is deprecated for 0d arrays/generics
    def nonzero(self) -> Tuple[ndarray, ...]: ...
    def partition(
        self,
        kth: _ArrayLikeIntOrBool,
        axis: int = ...,
        kind: _PartitionKind = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> None: ...
    # `put` is technically available to `generic`,
    # but is pointless as `generic`s are immutable
    def put(
        self, ind: _ArrayLikeIntOrBool, v: ArrayLike, mode: _ModeKind = ...
    ) -> None: ...
    def searchsorted(
        self,  # >= 1D array
        v: ArrayLike,
        side: _SortSide = ...,
        sorter: Optional[_ArrayLikeIntOrBool] = ...,  # 1D int array
    ) -> ndarray: ...
    def setfield(
        self, val: ArrayLike, dtype: DTypeLike, offset: int = ...
    ) -> None: ...
    def sort(
        self,
        axis: int = ...,
        kind: Optional[_SortKind] = ...,
        order: Union[None, str, Sequence[str]] = ...,
    ) -> None: ...
    @overload
    def trace(
        self,  # >= 2D array
        offset: int = ...,
        axis1: int = ...,
        axis2: int = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def trace(
        self,  # >= 2D array
        offset: int = ...,
        axis1: int = ...,
        axis2: int = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    # Many of these special methods are irrelevant currently, since protocols
    # aren't supported yet. That said, I'm adding them for completeness.
    # https://docs.python.org/3/reference/datamodel.html
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __len__(self) -> int: ...
    def __setitem__(self, key, value): ...
    def __iter__(self) -> Any: ...
    def __contains__(self, key) -> bool: ...
    def __index__(self) -> int: ...

    # The last overload is for catching recursive objects whose
    # nesting is too deep.
    # The first overload is for catching `bytes` (as they are a subtype of
    # `Sequence[int]`) and `str`. As `str` is a recusive sequence of
    # strings, it will pass through the final overload otherwise

    @overload
    def __lt__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __lt__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __lt__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __lt__(self: _ArrayND[datetime64], other: _ArrayLikeDT64_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __lt__(self: _ArrayND[object_], other: Any) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __lt__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __lt__(
        self: _ArrayND[Union[number[Any], datetime64, timedelta64, bool_]],
        other: _RecursiveSequence,
    ) -> _ArrayOrScalar[bool_]: ...

    @overload
    def __le__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __le__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __le__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __le__(self: _ArrayND[datetime64], other: _ArrayLikeDT64_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __le__(self: _ArrayND[object_], other: Any) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __le__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __le__(
        self: _ArrayND[Union[number[Any], datetime64, timedelta64, bool_]],
        other: _RecursiveSequence,
    ) -> _ArrayOrScalar[bool_]: ...

    @overload
    def __gt__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __gt__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __gt__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __gt__(self: _ArrayND[datetime64], other: _ArrayLikeDT64_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __gt__(self: _ArrayND[object_], other: Any) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __gt__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __gt__(
        self: _ArrayND[Union[number[Any], datetime64, timedelta64, bool_]],
        other: _RecursiveSequence,
    ) -> _ArrayOrScalar[bool_]: ...

    @overload
    def __ge__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __ge__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __ge__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __ge__(self: _ArrayND[datetime64], other: _ArrayLikeDT64_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __ge__(self: _ArrayND[object_], other: Any) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __ge__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __ge__(
        self: _ArrayND[Union[number[Any], datetime64, timedelta64, bool_]],
        other: _RecursiveSequence,
    ) -> _ArrayOrScalar[bool_]: ...

    # Unary ops
    @overload
    def __abs__(self: _ArrayND[bool_]) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __abs__(self: _ArrayND[complexfloating[_NBit1, _NBit1]]) -> _ArrayOrScalar[floating[_NBit1]]: ...
    @overload
    def __abs__(self: _ArrayND[_NumberType]) -> _ArrayOrScalar[_NumberType]: ...
    @overload
    def __abs__(self: _ArrayND[timedelta64]) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __abs__(self: _ArrayND[object_]) -> Any: ...

    @overload
    def __invert__(self: _ArrayND[bool_]) -> _ArrayOrScalar[bool_]: ...
    @overload
    def __invert__(self: _ArrayND[_IntType]) -> _ArrayOrScalar[_IntType]: ...
    @overload
    def __invert__(self: _ArrayND[object_]) -> Any: ...

    @overload
    def __pos__(self: _ArrayND[_NumberType]) -> _ArrayOrScalar[_NumberType]: ...
    @overload
    def __pos__(self: _ArrayND[timedelta64]) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __pos__(self: _ArrayND[object_]) -> Any: ...

    @overload
    def __neg__(self: _ArrayND[_NumberType]) -> _ArrayOrScalar[_NumberType]: ...
    @overload
    def __neg__(self: _ArrayND[timedelta64]) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __neg__(self: _ArrayND[object_]) -> Any: ...

    # Binary ops
    # NOTE: `ndarray` does not implement `__imatmul__`
    @overload
    def __matmul__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __matmul__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...
    @overload
    def __matmul__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __matmul__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __matmul__(
        self: _ArrayNumber_co,
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rmatmul__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rmatmul__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...
    @overload
    def __rmatmul__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rmatmul__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rmatmul__(
        self: _ArrayNumber_co,
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __mod__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __mod__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[int8]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __mod__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __mod__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __mod__(
        self: _ArrayND[Union[bool_, integer[Any], floating[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rmod__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rmod__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[int8]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __rmod__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rmod__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rmod__(
        self: _ArrayND[Union[bool_, integer[Any], floating[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __divmod__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __divmod__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _2Tuple[_ArrayOrScalar[int8]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _2Tuple[_ArrayOrScalar[unsignedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _2Tuple[_ArrayOrScalar[signedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _2Tuple[_ArrayOrScalar[floating[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> Union[Tuple[int64, timedelta64], Tuple[_ArrayND[int64], _ArrayND[timedelta64]]]: ...
    @overload
    def __divmod__(
        self: _ArrayND[Union[bool_, integer[Any], floating[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> _2Tuple[Any]: ...

    @overload
    def __rdivmod__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rdivmod__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _2Tuple[_ArrayOrScalar[int8]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _2Tuple[_ArrayOrScalar[unsignedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _2Tuple[_ArrayOrScalar[signedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _2Tuple[_ArrayOrScalar[floating[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> Union[Tuple[int64, timedelta64], Tuple[_ArrayND[int64], _ArrayND[timedelta64]]]: ...
    @overload
    def __rdivmod__(
        self: _ArrayND[Union[bool_, integer[Any], floating[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> _2Tuple[Any]: ...

    @overload
    def __add__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __add__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> _ArrayOrScalar[datetime64]: ...
    @overload
    def __add__(self: _ArrayND[datetime64], other: _ArrayLikeTD64_co) -> _ArrayOrScalar[datetime64]: ...
    @overload
    def __add__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __add__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __add__(
        self: _ArrayND[Union[bool_, number[Any], timedelta64, datetime64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __radd__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __radd__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> _ArrayOrScalar[datetime64]: ...
    @overload
    def __radd__(self: _ArrayND[datetime64], other: _ArrayLikeTD64_co) -> _ArrayOrScalar[datetime64]: ...
    @overload
    def __radd__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __radd__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __radd__(
        self: _ArrayND[Union[bool_, number[Any], timedelta64, datetime64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __sub__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __sub__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __sub__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayND[datetime64], other: _ArrayLikeTD64_co) -> _ArrayOrScalar[datetime64]: ...
    @overload
    def __sub__(self: _ArrayND[datetime64], other: _ArrayLikeDT64_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __sub__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __sub__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __sub__(
        self: _ArrayND[Union[bool_, number[Any], timedelta64, datetime64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rsub__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rsub__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __rsub__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> _ArrayOrScalar[datetime64]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayND[datetime64], other: _ArrayLikeDT64_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __rsub__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rsub__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rsub__(
        self: _ArrayND[Union[bool_, number[Any], timedelta64, datetime64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __mul__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __mul__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayTD64_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __mul__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __mul__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __mul__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __mul__(
        self: _ArrayND[Union[bool_, number[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rmul__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rmul__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayTD64_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __rmul__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __rmul__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rmul__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rmul__(
        self: _ArrayND[Union[bool_, number[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __floordiv__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __floordiv__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[int8]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayND[timedelta64], other: _NestedSequence[_SupportsArray[dtype[timedelta64]]]) -> _ArrayOrScalar[int64]: ...
    @overload
    def __floordiv__(self: _ArrayND[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __floordiv__(self: _ArrayND[timedelta64], other: _ArrayLikeFloat_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __floordiv__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __floordiv__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __floordiv__(
        self: _ArrayND[Union[bool_, number[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rfloordiv__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rfloordiv__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[int8]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayND[timedelta64], other: _NestedSequence[_SupportsArray[dtype[timedelta64]]]) -> _ArrayOrScalar[int64]: ...
    @overload
    def __rfloordiv__(self: _ArrayND[bool_], other: _ArrayLikeTD64_co) -> NoReturn: ...
    @overload
    def __rfloordiv__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __rfloordiv__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rfloordiv__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rfloordiv__(
        self: _ArrayND[Union[bool_, number[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __pow__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __pow__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[int8]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...
    @overload
    def __pow__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __pow__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __pow__(
        self: _ArrayND[Union[bool_, number[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rpow__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rpow__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[int8]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...
    @overload
    def __rpow__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rpow__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rpow__(
        self: _ArrayND[Union[bool_, number[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __truediv__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __truediv__(self: _ArrayInt_co, other: _ArrayInt_co) -> _ArrayOrScalar[float64]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: _ArrayND[timedelta64], other: _NestedSequence[_SupportsArray[dtype[timedelta64]]]) -> _ArrayOrScalar[float64]: ...
    @overload
    def __truediv__(self: _ArrayND[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __truediv__(self: _ArrayND[timedelta64], other: _ArrayLikeFloat_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __truediv__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __truediv__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __truediv__(
        self: _ArrayND[Union[bool_, number[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rtruediv__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rtruediv__(self: _ArrayInt_co, other: _ArrayInt_co) -> _ArrayOrScalar[float64]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _ArrayOrScalar[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> _ArrayOrScalar[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: _ArrayND[timedelta64], other: _NestedSequence[_SupportsArray[dtype[timedelta64]]]) -> _ArrayOrScalar[float64]: ...
    @overload
    def __rtruediv__(self: _ArrayND[bool_], other: _ArrayLikeTD64_co) -> NoReturn: ...
    @overload
    def __rtruediv__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> _ArrayOrScalar[timedelta64]: ...
    @overload
    def __rtruediv__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rtruediv__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rtruediv__(
        self: _ArrayND[Union[bool_, number[Any], timedelta64]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __lshift__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __lshift__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[int8]: ...  # type: ignore[misc]
    @overload
    def __lshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __lshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...
    @overload
    def __lshift__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __lshift__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __lshift__(
        self: _ArrayND[Union[bool_, integer[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rlshift__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rlshift__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[int8]: ...  # type: ignore[misc]
    @overload
    def __rlshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rlshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...
    @overload
    def __rlshift__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rlshift__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rlshift__(
        self: _ArrayND[Union[bool_, integer[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rshift__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rshift__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[int8]: ...  # type: ignore[misc]
    @overload
    def __rshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...
    @overload
    def __rshift__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rshift__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rshift__(
        self: _ArrayND[Union[bool_, integer[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rrshift__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rrshift__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[int8]: ...  # type: ignore[misc]
    @overload
    def __rrshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rrshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...
    @overload
    def __rrshift__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rrshift__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rrshift__(
        self: _ArrayND[Union[bool_, integer[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __and__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __and__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __and__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __and__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...
    @overload
    def __and__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __and__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __and__(
        self: _ArrayND[Union[bool_, integer[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rand__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rand__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __rand__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rand__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...
    @overload
    def __rand__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rand__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rand__(
        self: _ArrayND[Union[bool_, integer[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __xor__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __xor__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __xor__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __xor__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...
    @overload
    def __xor__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __xor__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __xor__(
        self: _ArrayND[Union[bool_, integer[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __rxor__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __rxor__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __rxor__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rxor__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...
    @overload
    def __rxor__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __rxor__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __rxor__(
        self: _ArrayND[Union[bool_, integer[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __or__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __or__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __or__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __or__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...
    @overload
    def __or__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __or__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __or__(
        self: _ArrayND[Union[bool_, integer[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    @overload
    def __ror__(self: _ArrayND[Any], other: _NestedSequence[Union[str, bytes]]) -> NoReturn: ...
    @overload
    def __ror__(self: _ArrayND[bool_], other: _ArrayLikeBool_co) -> _ArrayOrScalar[bool_]: ...  # type: ignore[misc]
    @overload
    def __ror__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _ArrayOrScalar[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __ror__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _ArrayOrScalar[signedinteger[Any]]: ...
    @overload
    def __ror__(self: _ArrayND[object_], other: Any) -> Any: ...
    @overload
    def __ror__(self: _ArrayND[Any], other: _ArrayLikeObject_co) -> Any: ...
    @overload
    def __ror__(
        self: _ArrayND[Union[bool_, integer[Any]]],
        other: _RecursiveSequence,
    ) -> Any: ...

    # `np.generic` does not support inplace operations
    def __iadd__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __isub__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __imul__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __itruediv__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __ifloordiv__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __ipow__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __imod__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __ilshift__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __irshift__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __iand__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __ixor__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __ior__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...

    # Keep `dtype` at the bottom to avoid name conflicts with `np.dtype`
    @property
    def dtype(self) -> _DType_co: ...

# NOTE: while `np.generic` is not technically an instance of `ABCMeta`,
# the `@abstractmethod` decorator is herein used to (forcefully) deny
# the creation of `np.generic` instances.
# The `# type: ignore` comments are necessary to silence mypy errors regarding
# the missing `ABCMeta` metaclass.

# See https://github.com/numpy/numpy-stubs/pull/80 for more details.

_ScalarType = TypeVar("_ScalarType", bound=generic)
_NBit1 = TypeVar("_NBit1", bound=NBitBase)
_NBit2 = TypeVar("_NBit2", bound=NBitBase)

class generic(_ArrayOrScalarCommon):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def __array__(self: _ScalarType, __dtype: None = ...) -> ndarray[Any, dtype[_ScalarType]]: ...
    @overload
    def __array__(self, __dtype: DTypeLike) -> ndarray[Any, dtype[Any]]: ...
    @property
    def base(self) -> None: ...
    @property
    def ndim(self) -> Literal[0]: ...
    @property
    def size(self) -> Literal[1]: ...
    @property
    def shape(self) -> Tuple[()]: ...
    @property
    def strides(self) -> Tuple[()]: ...
    def byteswap(self: _ScalarType, inplace: Literal[False] = ...) -> _ScalarType: ...
    @property
    def flat(self: _ScalarType) -> flatiter[ndarray[Any, dtype[_ScalarType]]]: ...
    def item(
        self: _ScalarType,
        __args: Union[Literal[0], Tuple[()], Tuple[Literal[0]]] = ...,
    ) -> Any: ...
    def squeeze(
        self: _ScalarType, axis: Union[Literal[0], Tuple[()]] = ...
    ) -> _ScalarType: ...
    def transpose(self: _ScalarType, __axes: Tuple[()] = ...) -> _ScalarType: ...
    # Keep `dtype` at the bottom to avoid name conflicts with `np.dtype`
    @property
    def dtype(self: _ScalarType) -> dtype[_ScalarType]: ...

class number(generic, Generic[_NBit1]):  # type: ignore
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    # Ensure that objects annotated as `number` support arithmetic operations
    __add__: _NumberOp
    __radd__: _NumberOp
    __sub__: _NumberOp
    __rsub__: _NumberOp
    __mul__: _NumberOp
    __rmul__: _NumberOp
    __floordiv__: _NumberOp
    __rfloordiv__: _NumberOp
    __pow__: _NumberOp
    __rpow__: _NumberOp
    __truediv__: _NumberOp
    __rtruediv__: _NumberOp
    __lt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __le__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __gt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __ge__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]

class bool_(generic):
    def __init__(self, __value: object = ...) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    __add__: _BoolOp[bool_]
    __radd__: _BoolOp[bool_]
    __sub__: _BoolSub
    __rsub__: _BoolSub
    __mul__: _BoolOp[bool_]
    __rmul__: _BoolOp[bool_]
    __floordiv__: _BoolOp[int8]
    __rfloordiv__: _BoolOp[int8]
    __pow__: _BoolOp[int8]
    __rpow__: _BoolOp[int8]
    __truediv__: _BoolTrueDiv
    __rtruediv__: _BoolTrueDiv
    def __invert__(self) -> bool_: ...
    __lshift__: _BoolBitOp[int8]
    __rlshift__: _BoolBitOp[int8]
    __rshift__: _BoolBitOp[int8]
    __rrshift__: _BoolBitOp[int8]
    __and__: _BoolBitOp[bool_]
    __rand__: _BoolBitOp[bool_]
    __xor__: _BoolBitOp[bool_]
    __rxor__: _BoolBitOp[bool_]
    __or__: _BoolBitOp[bool_]
    __ror__: _BoolBitOp[bool_]
    __mod__: _BoolMod
    __rmod__: _BoolMod
    __divmod__: _BoolDivMod
    __rdivmod__: _BoolDivMod
    __lt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __le__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __gt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __ge__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]

bool8 = bool_

class object_(generic):
    def __init__(self, __value: object = ...) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...

object0 = object_

class datetime64(generic):
    @overload
    def __init__(
        self,
        __value: Union[None, datetime64, _CharLike_co, dt.datetime] = ...,
        __format: Union[_CharLike_co, Tuple[_CharLike_co, _IntLike_co]] = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        __value: int,
        __format: Union[_CharLike_co, Tuple[_CharLike_co, _IntLike_co]]
    ) -> None: ...
    def __add__(self, other: _TD64Like_co) -> datetime64: ...
    def __radd__(self, other: _TD64Like_co) -> datetime64: ...
    @overload
    def __sub__(self, other: datetime64) -> timedelta64: ...
    @overload
    def __sub__(self, other: _TD64Like_co) -> datetime64: ...
    def __rsub__(self, other: datetime64) -> timedelta64: ...
    __lt__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __le__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __gt__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __ge__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]

# Support for `__index__` was added in python 3.8 (bpo-20092)
if sys.version_info >= (3, 8):
    _IntValue = Union[SupportsInt, _CharLike_co, SupportsIndex]
    _FloatValue = Union[None, _CharLike_co, SupportsFloat, SupportsIndex]
    _ComplexValue = Union[None, _CharLike_co, SupportsFloat, SupportsComplex, SupportsIndex]
else:
    _IntValue = Union[SupportsInt, _CharLike_co]
    _FloatValue = Union[None, _CharLike_co, SupportsFloat]
    _ComplexValue = Union[None, _CharLike_co, SupportsFloat, SupportsComplex]

class integer(number[_NBit1]):  # type: ignore
    # NOTE: `__index__` is technically defined in the bottom-most
    # sub-classes (`int64`, `uint32`, etc)
    def __index__(self) -> int: ...
    __truediv__: _IntTrueDiv[_NBit1]
    __rtruediv__: _IntTrueDiv[_NBit1]
    def __mod__(self, value: _IntLike_co) -> integer: ...
    def __rmod__(self, value: _IntLike_co) -> integer: ...
    def __invert__(self: _IntType) -> _IntType: ...
    # Ensure that objects annotated as `integer` support bit-wise operations
    def __lshift__(self, other: _IntLike_co) -> integer: ...
    def __rlshift__(self, other: _IntLike_co) -> integer: ...
    def __rshift__(self, other: _IntLike_co) -> integer: ...
    def __rrshift__(self, other: _IntLike_co) -> integer: ...
    def __and__(self, other: _IntLike_co) -> integer: ...
    def __rand__(self, other: _IntLike_co) -> integer: ...
    def __or__(self, other: _IntLike_co) -> integer: ...
    def __ror__(self, other: _IntLike_co) -> integer: ...
    def __xor__(self, other: _IntLike_co) -> integer: ...
    def __rxor__(self, other: _IntLike_co) -> integer: ...

class signedinteger(integer[_NBit1]):
    def __init__(self, __value: _IntValue = ...) -> None: ...
    __add__: _SignedIntOp[_NBit1]
    __radd__: _SignedIntOp[_NBit1]
    __sub__: _SignedIntOp[_NBit1]
    __rsub__: _SignedIntOp[_NBit1]
    __mul__: _SignedIntOp[_NBit1]
    __rmul__: _SignedIntOp[_NBit1]
    __floordiv__: _SignedIntOp[_NBit1]
    __rfloordiv__: _SignedIntOp[_NBit1]
    __pow__: _SignedIntOp[_NBit1]
    __rpow__: _SignedIntOp[_NBit1]
    __lshift__: _SignedIntBitOp[_NBit1]
    __rlshift__: _SignedIntBitOp[_NBit1]
    __rshift__: _SignedIntBitOp[_NBit1]
    __rrshift__: _SignedIntBitOp[_NBit1]
    __and__: _SignedIntBitOp[_NBit1]
    __rand__: _SignedIntBitOp[_NBit1]
    __xor__: _SignedIntBitOp[_NBit1]
    __rxor__: _SignedIntBitOp[_NBit1]
    __or__: _SignedIntBitOp[_NBit1]
    __ror__: _SignedIntBitOp[_NBit1]
    __mod__: _SignedIntMod[_NBit1]
    __rmod__: _SignedIntMod[_NBit1]
    __divmod__: _SignedIntDivMod[_NBit1]
    __rdivmod__: _SignedIntDivMod[_NBit1]

int8 = signedinteger[_8Bit]
int16 = signedinteger[_16Bit]
int32 = signedinteger[_32Bit]
int64 = signedinteger[_64Bit]

byte = signedinteger[_NBitByte]
short = signedinteger[_NBitShort]
intc = signedinteger[_NBitIntC]
intp = signedinteger[_NBitIntP]
int0 = signedinteger[_NBitIntP]
int_ = signedinteger[_NBitInt]
longlong = signedinteger[_NBitLongLong]

class timedelta64(generic):
    def __init__(
        self,
        __value: Union[None, int, _CharLike_co, dt.timedelta, timedelta64] = ...,
        __format: Union[_CharLike_co, Tuple[_CharLike_co, _IntLike_co]] = ...,
    ) -> None: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    def __add__(self, other: _TD64Like_co) -> timedelta64: ...
    def __radd__(self, other: _TD64Like_co) -> timedelta64: ...
    def __sub__(self, other: _TD64Like_co) -> timedelta64: ...
    def __rsub__(self, other: _TD64Like_co) -> timedelta64: ...
    def __mul__(self, other: _FloatLike_co) -> timedelta64: ...
    def __rmul__(self, other: _FloatLike_co) -> timedelta64: ...
    __truediv__: _TD64Div[float64]
    __floordiv__: _TD64Div[int64]
    def __rtruediv__(self, other: timedelta64) -> float64: ...
    def __rfloordiv__(self, other: timedelta64) -> int64: ...
    def __mod__(self, other: timedelta64) -> timedelta64: ...
    def __rmod__(self, other: timedelta64) -> timedelta64: ...
    def __divmod__(self, other: timedelta64) -> Tuple[int64, timedelta64]: ...
    def __rdivmod__(self, other: timedelta64) -> Tuple[int64, timedelta64]: ...
    __lt__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __le__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __gt__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __ge__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]

class unsignedinteger(integer[_NBit1]):
    # NOTE: `uint64 + signedinteger -> float64`
    def __init__(self, __value: _IntValue = ...) -> None: ...
    __add__: _UnsignedIntOp[_NBit1]
    __radd__: _UnsignedIntOp[_NBit1]
    __sub__: _UnsignedIntOp[_NBit1]
    __rsub__: _UnsignedIntOp[_NBit1]
    __mul__: _UnsignedIntOp[_NBit1]
    __rmul__: _UnsignedIntOp[_NBit1]
    __floordiv__: _UnsignedIntOp[_NBit1]
    __rfloordiv__: _UnsignedIntOp[_NBit1]
    __pow__: _UnsignedIntOp[_NBit1]
    __rpow__: _UnsignedIntOp[_NBit1]
    __lshift__: _UnsignedIntBitOp[_NBit1]
    __rlshift__: _UnsignedIntBitOp[_NBit1]
    __rshift__: _UnsignedIntBitOp[_NBit1]
    __rrshift__: _UnsignedIntBitOp[_NBit1]
    __and__: _UnsignedIntBitOp[_NBit1]
    __rand__: _UnsignedIntBitOp[_NBit1]
    __xor__: _UnsignedIntBitOp[_NBit1]
    __rxor__: _UnsignedIntBitOp[_NBit1]
    __or__: _UnsignedIntBitOp[_NBit1]
    __ror__: _UnsignedIntBitOp[_NBit1]
    __mod__: _UnsignedIntMod[_NBit1]
    __rmod__: _UnsignedIntMod[_NBit1]
    __divmod__: _UnsignedIntDivMod[_NBit1]
    __rdivmod__: _UnsignedIntDivMod[_NBit1]

uint8 = unsignedinteger[_8Bit]
uint16 = unsignedinteger[_16Bit]
uint32 = unsignedinteger[_32Bit]
uint64 = unsignedinteger[_64Bit]

ubyte = unsignedinteger[_NBitByte]
ushort = unsignedinteger[_NBitShort]
uintc = unsignedinteger[_NBitIntC]
uintp = unsignedinteger[_NBitIntP]
uint0 = unsignedinteger[_NBitIntP]
uint = unsignedinteger[_NBitInt]
ulonglong = unsignedinteger[_NBitLongLong]

class inexact(number[_NBit1]): ...  # type: ignore

_IntType = TypeVar("_IntType", bound=integer)
_FloatType = TypeVar('_FloatType', bound=floating)

class floating(inexact[_NBit1]):
    def __init__(self, __value: _FloatValue = ...) -> None: ...
    __add__: _FloatOp[_NBit1]
    __radd__: _FloatOp[_NBit1]
    __sub__: _FloatOp[_NBit1]
    __rsub__: _FloatOp[_NBit1]
    __mul__: _FloatOp[_NBit1]
    __rmul__: _FloatOp[_NBit1]
    __truediv__: _FloatOp[_NBit1]
    __rtruediv__: _FloatOp[_NBit1]
    __floordiv__: _FloatOp[_NBit1]
    __rfloordiv__: _FloatOp[_NBit1]
    __pow__: _FloatOp[_NBit1]
    __rpow__: _FloatOp[_NBit1]
    __mod__: _FloatMod[_NBit1]
    __rmod__: _FloatMod[_NBit1]
    __divmod__: _FloatDivMod[_NBit1]
    __rdivmod__: _FloatDivMod[_NBit1]

float16 = floating[_16Bit]
float32 = floating[_32Bit]
float64 = floating[_64Bit]

half = floating[_NBitHalf]
single = floating[_NBitSingle]
double = floating[_NBitDouble]
float_ = floating[_NBitDouble]
longdouble = floating[_NBitLongDouble]
longfloat = floating[_NBitLongDouble]

# The main reason for `complexfloating` having two typevars is cosmetic.
# It is used to clarify why `complex128`s precision is `_64Bit`, the latter
# describing the two 64 bit floats representing its real and imaginary component

class complexfloating(inexact[_NBit1], Generic[_NBit1, _NBit2]):
    def __init__(self, __value: _ComplexValue = ...) -> None: ...
    @property
    def real(self) -> floating[_NBit1]: ...  # type: ignore[override]
    @property
    def imag(self) -> floating[_NBit2]: ...  # type: ignore[override]
    def __abs__(self) -> floating[_NBit1]: ...  # type: ignore[override]
    __add__: _ComplexOp[_NBit1]
    __radd__: _ComplexOp[_NBit1]
    __sub__: _ComplexOp[_NBit1]
    __rsub__: _ComplexOp[_NBit1]
    __mul__: _ComplexOp[_NBit1]
    __rmul__: _ComplexOp[_NBit1]
    __truediv__: _ComplexOp[_NBit1]
    __rtruediv__: _ComplexOp[_NBit1]
    __floordiv__: _ComplexOp[_NBit1]
    __rfloordiv__: _ComplexOp[_NBit1]
    __pow__: _ComplexOp[_NBit1]
    __rpow__: _ComplexOp[_NBit1]

complex64 = complexfloating[_32Bit, _32Bit]
complex128 = complexfloating[_64Bit, _64Bit]

csingle = complexfloating[_NBitSingle, _NBitSingle]
singlecomplex = complexfloating[_NBitSingle, _NBitSingle]
cdouble = complexfloating[_NBitDouble, _NBitDouble]
complex_ = complexfloating[_NBitDouble, _NBitDouble]
cfloat = complexfloating[_NBitDouble, _NBitDouble]
clongdouble = complexfloating[_NBitLongDouble, _NBitLongDouble]
clongfloat = complexfloating[_NBitLongDouble, _NBitLongDouble]
longcomplex = complexfloating[_NBitLongDouble, _NBitLongDouble]

class flexible(generic): ...  # type: ignore

class void(flexible):
    def __init__(self, __value: Union[_IntLike_co, bytes]): ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def setfield(
        self, val: ArrayLike, dtype: DTypeLike, offset: int = ...
    ) -> None: ...

void0 = void

class character(flexible):  # type: ignore
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...

# NOTE: Most `np.bytes_` / `np.str_` methods return their
# builtin `bytes` / `str` counterpart

class bytes_(character, bytes):
    @overload
    def __init__(self, __value: object = ...) -> None: ...
    @overload
    def __init__(
        self, __value: str, encoding: str = ..., errors: str = ...
    ) -> None: ...

string_ = bytes_
bytes0 = bytes_

class str_(character, str):
    @overload
    def __init__(self, __value: object = ...) -> None: ...
    @overload
    def __init__(
        self, __value: bytes, encoding: str = ..., errors: str = ...
    ) -> None: ...

unicode_ = str_
str0 = str_

def array(
    object: object,
    dtype: DTypeLike = ...,
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: ArrayLike = ...,
) -> ndarray: ...
def zeros(
    shape: _ShapeLike,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> ndarray: ...
def empty(
    shape: _ShapeLike,
    dtype: DTypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> ndarray: ...

def broadcast_shapes(*args: _ShapeLike) -> _Shape: ...

#
# Constants
#

Inf: Final[float]
Infinity: Final[float]
NAN: Final[float]
NINF: Final[float]
NZERO: Final[float]
NaN: Final[float]
PINF: Final[float]
PZERO: Final[float]
e: Final[float]
euler_gamma: Final[float]
inf: Final[float]
infty: Final[float]
nan: Final[float]
pi: Final[float]
ALLOW_THREADS: Final[int]
BUFSIZE: Final[int]
CLIP: Final[int]
ERR_CALL: Final[int]
ERR_DEFAULT: Final[int]
ERR_IGNORE: Final[int]
ERR_LOG: Final[int]
ERR_PRINT: Final[int]
ERR_RAISE: Final[int]
ERR_WARN: Final[int]
FLOATING_POINT_SUPPORT: Final[int]
FPE_DIVIDEBYZERO: Final[int]
FPE_INVALID: Final[int]
FPE_OVERFLOW: Final[int]
FPE_UNDERFLOW: Final[int]
MAXDIMS: Final[int]
MAY_SHARE_BOUNDS: Final[int]
MAY_SHARE_EXACT: Final[int]
RAISE: Final[int]
SHIFT_DIVIDEBYZERO: Final[int]
SHIFT_INVALID: Final[int]
SHIFT_OVERFLOW: Final[int]
SHIFT_UNDERFLOW: Final[int]
UFUNC_BUFSIZE_DEFAULT: Final[int]
WRAP: Final[int]
tracemalloc_domain: Final[int]

little_endian: Final[bool]
True_: Final[bool_]
False_: Final[bool_]

UFUNC_PYVALS_NAME: Final[str]

class ufunc:
    @property
    def __name__(self) -> str: ...
    def __call__(
        self,
        *args: ArrayLike,
        out: Optional[Union[ndarray, Tuple[ndarray, ...]]] = ...,
        where: Optional[ndarray] = ...,
        # The list should be a list of tuples of ints, but since we
        # don't know the signature it would need to be
        # Tuple[int, ...]. But, since List is invariant something like
        # e.g. List[Tuple[int, int]] isn't a subtype of
        # List[Tuple[int, ...]], so we can't type precisely here.
        axes: List[Any] = ...,
        axis: int = ...,
        keepdims: bool = ...,
        casting: _Casting = ...,
        order: _OrderKACF = ...,
        dtype: DTypeLike = ...,
        subok: bool = ...,
        signature: Union[str, Tuple[str]] = ...,
        # In reality this should be a length of list 3 containing an
        # int, an int, and a callable, but there's no way to express
        # that.
        extobj: List[Union[int, Callable]] = ...,
    ) -> Any: ...
    @property
    def nin(self) -> int: ...
    @property
    def nout(self) -> int: ...
    @property
    def nargs(self) -> int: ...
    @property
    def ntypes(self) -> int: ...
    @property
    def types(self) -> List[str]: ...
    # Broad return type because it has to encompass things like
    #
    # >>> np.logical_and.identity is True
    # True
    # >>> np.add.identity is 0
    # True
    # >>> np.sin.identity is None
    # True
    #
    # and any user-defined ufuncs.
    @property
    def identity(self) -> Any: ...
    # This is None for ufuncs and a string for gufuncs.
    @property
    def signature(self) -> Optional[str]: ...
    # The next four methods will always exist, but they will just
    # raise a ValueError ufuncs with that don't accept two input
    # arguments and return one output argument. Because of that we
    # can't type them very precisely.
    @property
    def reduce(self) -> Any: ...
    @property
    def accumulate(self) -> Any: ...
    @property
    def reduceat(self) -> Any: ...
    @property
    def outer(self) -> Any: ...
    # Similarly at won't be defined for ufuncs that return multiple
    # outputs, so we can't type it very precisely.
    @property
    def at(self) -> Any: ...

absolute: ufunc
add: ufunc
arccos: ufunc
arccosh: ufunc
arcsin: ufunc
arcsinh: ufunc
arctan2: ufunc
arctan: ufunc
arctanh: ufunc
bitwise_and: ufunc
bitwise_or: ufunc
bitwise_xor: ufunc
cbrt: ufunc
ceil: ufunc
conjugate: ufunc
copysign: ufunc
cos: ufunc
cosh: ufunc
deg2rad: ufunc
degrees: ufunc
divmod: ufunc
equal: ufunc
exp2: ufunc
exp: ufunc
expm1: ufunc
fabs: ufunc
float_power: ufunc
floor: ufunc
floor_divide: ufunc
fmax: ufunc
fmin: ufunc
fmod: ufunc
frexp: ufunc
gcd: ufunc
greater: ufunc
greater_equal: ufunc
heaviside: ufunc
hypot: ufunc
invert: ufunc
isfinite: ufunc
isinf: ufunc
isnan: ufunc
isnat: ufunc
lcm: ufunc
ldexp: ufunc
left_shift: ufunc
less: ufunc
less_equal: ufunc
log10: ufunc
log1p: ufunc
log2: ufunc
log: ufunc
logaddexp2: ufunc
logaddexp: ufunc
logical_and: ufunc
logical_not: ufunc
logical_or: ufunc
logical_xor: ufunc
matmul: ufunc
maximum: ufunc
minimum: ufunc
modf: ufunc
multiply: ufunc
negative: ufunc
nextafter: ufunc
not_equal: ufunc
positive: ufunc
power: ufunc
rad2deg: ufunc
radians: ufunc
reciprocal: ufunc
remainder: ufunc
right_shift: ufunc
rint: ufunc
sign: ufunc
signbit: ufunc
sin: ufunc
sinh: ufunc
spacing: ufunc
sqrt: ufunc
square: ufunc
subtract: ufunc
tan: ufunc
tanh: ufunc
true_divide: ufunc
trunc: ufunc

abs = absolute

# Warnings
class ModuleDeprecationWarning(DeprecationWarning): ...
class VisibleDeprecationWarning(UserWarning): ...
class ComplexWarning(RuntimeWarning): ...
class RankWarning(UserWarning): ...

# Errors
class TooHardError(RuntimeError): ...

class AxisError(ValueError, IndexError):
    def __init__(
        self, axis: int, ndim: Optional[int] = ..., msg_prefix: Optional[str] = ...
    ) -> None: ...

_CallType = TypeVar("_CallType", bound=Union[_ErrFunc, _SupportsWrite])

class errstate(Generic[_CallType], ContextDecorator):
    call: _CallType
    kwargs: _ErrDictOptional

    # Expand `**kwargs` into explicit keyword-only arguments
    def __init__(
        self,
        *,
        call: _CallType = ...,
        all: Optional[_ErrKind] = ...,
        divide: Optional[_ErrKind] = ...,
        over: Optional[_ErrKind] = ...,
        under: Optional[_ErrKind] = ...,
        invalid: Optional[_ErrKind] = ...,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        __exc_type: Optional[Type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> None: ...

class ndenumerate(Generic[_ScalarType]):
    iter: flatiter[_ArrayND[_ScalarType]]
    @overload
    def __new__(
        cls, arr: _NestedSequence[_SupportsArray[dtype[_ScalarType]]],
    ) -> ndenumerate[_ScalarType]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[str]) -> ndenumerate[str_]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[bytes]) -> ndenumerate[bytes_]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[bool]) -> ndenumerate[bool_]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[int]) -> ndenumerate[int_]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[float]) -> ndenumerate[float_]: ...
    @overload
    def __new__(cls, arr: _NestedSequence[complex]) -> ndenumerate[complex_]: ...
    @overload
    def __new__(cls, arr: _RecursiveSequence) -> ndenumerate[Any]: ...
    def __next__(self: ndenumerate[_ScalarType]) -> Tuple[_Shape, _ScalarType]: ...
    def __iter__(self: _T) -> _T: ...

class ndindex:
    def __init__(self, *shape: SupportsIndex) -> None: ...
    def __iter__(self: _T) -> _T: ...
    def __next__(self) -> _Shape: ...
