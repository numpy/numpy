import builtins
import sys
import datetime as dt
from abc import abstractmethod

from numpy.core._internal import _ctypes
from numpy.typing import (
    ArrayLike,
    DtypeLike,
    _Shape,
    _ShapeLike,
    _CharLike,
    _BoolLike,
    _IntLike,
    _FloatLike,
    _ComplexLike,
    _NumberLike,
    _SupportsDtype,
    _VoidDtypeLike,
)
from numpy.typing._callable import (
    _BoolOp,
    _BoolBitOp,
    _BoolSub,
    _BoolTrueDiv,
    _TD64Div,
    _IntTrueDiv,
    _UnsignedIntOp,
    _UnsignedIntBitOp,
    _SignedIntOp,
    _SignedIntBitOp,
    _FloatOp,
    _ComplexOp,
    _NumberOp,
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
    Optional,
    overload,
    Sequence,
    Sized,
    SupportsAbs,
    SupportsBytes,
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
    from typing_extensions import Literal, Protocol, Final
    class SupportsIndex(Protocol):
        def __index__(self) -> int: ...

# Ensures that the stubs are picked up
from numpy import (
    char,
    ctypeslib,
    emath,
    fft,
    lib,
    linalg,
    ma,
    matrixlib,
    polynomial,
    random,
    rec,
    testing,
    version,
)

from numpy.core.function_base import (
    linspace,
    logspace,
    geomspace,
)

from numpy.core.fromnumeric import (
    take,
    reshape,
    choose,
    repeat,
    put,
    swapaxes,
    transpose,
    partition,
    argpartition,
    sort,
    argsort,
    argmax,
    argmin,
    searchsorted,
    resize,
    squeeze,
    diagonal,
    trace,
    ravel,
    nonzero,
    shape,
    compress,
    clip,
    sum,
    all,
    any,
    cumsum,
    ptp,
    amax,
    amin,
    prod,
    cumprod,
    ndim,
    size,
    around,
    mean,
    std,
    var,
)

from numpy.core._asarray import (
    asarray as asarray,
    asanyarray as asanyarray,
    ascontiguousarray as ascontiguousarray,
    asfortranarray as asfortranarray,
    require as require,
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

# Add an object to `__all__` if their stubs are defined in an external file;
# their stubs will not be recognized otherwise.
# NOTE: This is redundant for objects defined within this file.
__all__ = [
    "linspace",
    "logspace",
    "geomspace",
    "take",
    "reshape",
    "choose",
    "repeat",
    "put",
    "swapaxes",
    "transpose",
    "partition",
    "argpartition",
    "sort",
    "argsort",
    "argmax",
    "argmin",
    "searchsorted",
    "resize",
    "squeeze",
    "diagonal",
    "trace",
    "ravel",
    "nonzero",
    "shape",
    "compress",
    "clip",
    "sum",
    "all",
    "any",
    "cumsum",
    "ptp",
    "amax",
    "amin",
    "prod",
    "cumprod",
    "ndim",
    "size",
    "around",
    "mean",
    "std",
    "var",
]

DataSource: Any
MachAr: Any
ScalarType: Any
angle: Any
append: Any
apply_along_axis: Any
apply_over_axes: Any
arange: Any
array2string: Any
array_repr: Any
array_split: Any
array_str: Any
asarray_chkfinite: Any
asfarray: Any
asmatrix: Any
asscalar: Any
atleast_1d: Any
atleast_2d: Any
atleast_3d: Any
average: Any
bartlett: Any
bincount: Any
bitwise_not: Any
blackman: Any
block: Any
bmat: Any
bool8: Any
broadcast: Any
broadcast_arrays: Any
broadcast_to: Any
busday_count: Any
busday_offset: Any
busdaycalendar: Any
byte: Any
byte_bounds: Any
bytes0: Any
c_: Any
can_cast: Any
cast: Any
cdouble: Any
cfloat: Any
chararray: Any
clongdouble: Any
clongfloat: Any
column_stack: Any
common_type: Any
compare_chararrays: Any
complex256: Any
complex_: Any
concatenate: Any
conj: Any
copy: Any
copyto: Any
corrcoef: Any
cov: Any
csingle: Any
cumproduct: Any
datetime_as_string: Any
datetime_data: Any
delete: Any
deprecate: Any
deprecate_with_doc: Any
diag: Any
diag_indices: Any
diag_indices_from: Any
diagflat: Any
diff: Any
digitize: Any
disp: Any
divide: Any
dot: Any
double: Any
dsplit: Any
dstack: Any
ediff1d: Any
einsum: Any
einsum_path: Any
errstate: Any
expand_dims: Any
extract: Any
eye: Any
fill_diagonal: Any
finfo: Any
fix: Any
flip: Any
fliplr: Any
flipud: Any
float128: Any
float_: Any
format_float_positional: Any
format_float_scientific: Any
format_parser: Any
frombuffer: Any
fromfile: Any
fromiter: Any
frompyfunc: Any
fromregex: Any
fromstring: Any
genfromtxt: Any
get_include: Any
get_printoptions: Any
getbufsize: Any
geterr: Any
geterrcall: Any
geterrobj: Any
gradient: Any
half: Any
hamming: Any
hanning: Any
histogram: Any
histogram2d: Any
histogram_bin_edges: Any
histogramdd: Any
hsplit: Any
hstack: Any
i0: Any
iinfo: Any
imag: Any
in1d: Any
index_exp: Any
info: Any
inner: Any
insert: Any
int0: Any
int_: Any
intc: Any
interp: Any
intersect1d: Any
intp: Any
is_busday: Any
iscomplex: Any
iscomplexobj: Any
isin: Any
isneginf: Any
isposinf: Any
isreal: Any
isrealobj: Any
iterable: Any
ix_: Any
kaiser: Any
kron: Any
lexsort: Any
load: Any
loads: Any
loadtxt: Any
longcomplex: Any
longdouble: Any
longfloat: Any
longlong: Any
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
mgrid: Any
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
ndenumerate: Any
ndfromtxt: Any
ndindex: Any
nditer: Any
nested_iters: Any
newaxis: Any
numarray: Any
object0: Any
ogrid: Any
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
printoptions: Any
product: Any
promote_types: Any
put_along_axis: Any
putmask: Any
quantile: Any
r_: Any
ravel_multi_index: Any
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
s_: Any
save: Any
savetxt: Any
savez: Any
savez_compressed: Any
sctypeDict: Any
sctypes: Any
select: Any
set_printoptions: Any
set_string_function: Any
setbufsize: Any
setdiff1d: Any
seterr: Any
seterrcall: Any
seterrobj: Any
setxor1d: Any
shares_memory: Any
short: Any
show_config: Any
sinc: Any
single: Any
singlecomplex: Any
sort_complex: Any
source: Any
split: Any
stack: Any
string_: Any
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
ubyte: Any
uint: Any
uint0: Any
uintc: Any
uintp: Any
ulonglong: Any
union1d: Any
unique: Any
unpackbits: Any
unravel_index: Any
unwrap: Any
ushort: Any
vander: Any
vdot: Any
vectorize: Any
void0: Any
vsplit: Any
vstack: Any
where: Any
who: Any

_NdArraySubClass = TypeVar("_NdArraySubClass", bound=ndarray)
_DTypeScalar = TypeVar("_DTypeScalar", bound=generic)
_ByteOrder = Literal["S", "<", ">", "=", "|", "L", "B", "N", "I"]

class dtype(Generic[_DTypeScalar]):
    names: Optional[Tuple[str, ...]]
    # Overload for subclass of generic
    @overload
    def __new__(
        cls,
        dtype: Type[_DTypeScalar],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar]: ...
    # Overloads for string aliases, Python types, and some assorted
    # other special cases. Order is sometimes important because of the
    # subtype relationships
    #
    # bool < int < float < complex
    #
    # so we have to make sure the overloads for the narrowest type is
    # first.
    @overload
    def __new__(
        cls,
        dtype: Union[
            Type[bool],
            Literal[
                "?",
                "=?",
                "<?",
                ">?",
                "bool",
                "bool_",
            ],
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[bool_]: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "uint8",
            "u1",
            "=u1",
            "<u1",
            ">u1",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[uint8]: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "uint16",
            "u2",
            "=u2",
            "<u2",
            ">u2",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[uint16]: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "uint32",
            "u4",
            "=u4",
            "<u4",
            ">u4",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[uint32]: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "uint64",
            "u8",
            "=u8",
            "<u8",
            ">u8",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[uint64]: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "int8",
            "i1",
            "=i1",
            "<i1",
            ">i1",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[int8]: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "int16",
            "i2",
            "=i2",
            "<i2",
            ">i2",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[int16]: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "int32",
            "i4",
            "=i4",
            "<i4",
            ">i4",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[int32]: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "int64",
            "i8",
            "=i8",
            "<i8",
            ">i8",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[int64]: ...
    # "int"/int resolve to int_, which is system dependent and as of
    # now untyped. Long-term we'll do something fancier here.
    @overload
    def __new__(
        cls,
        dtype: Union[Type[int], Literal["int"]],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "float16",
            "f4",
            "=f4",
            "<f4",
            ">f4",
            "e",
            "=e",
            "<e",
            ">e",
            "half",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[float16]: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "float32",
            "f4",
            "=f4",
            "<f4",
            ">f4",
            "f",
            "=f",
            "<f",
            ">f",
            "single",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[float32]: ...
    @overload
    def __new__(
        cls,
        dtype: Union[
            None,
            Type[float],
            Literal[
                "float64",
                "f8",
                "=f8",
                "<f8",
                ">f8",
                "d",
                "<d",
                ">d",
                "float",
                "double",
                "float_",
            ],
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[float64]: ...
    @overload
    def __new__(
        cls,
        dtype: Literal[
            "complex64",
            "c8",
            "=c8",
            "<c8",
            ">c8",
            "F",
            "=F",
            "<F",
            ">F",
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[complex64]: ...
    @overload
    def __new__(
        cls,
        dtype: Union[
            Type[complex],
            Literal[
                "complex128",
                "c16",
                "=c16",
                "<c16",
                ">c16",
                "D",
                "=D",
                "<D",
                ">D",
            ],
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[complex128]: ...
    @overload
    def __new__(
        cls,
        dtype: Union[
            Type[bytes],
            Literal[
                "S",
                "=S",
                "<S",
                ">S",
                "bytes",
                "bytes_",
                "bytes0",
            ],
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[bytes_]: ...
    @overload
    def __new__(
        cls,
        dtype: Union[
            Type[str],
            Literal[
                "U",
                "=U",
                # <U and >U intentionally not included; they are not
                # the same dtype and which one dtype("U") translates
                # to is platform-dependent.
                "str",
                "str_",
                "str0",
            ],
        ],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[str_]: ...
    # dtype of a dtype is the same dtype
    @overload
    def __new__(
        cls,
        dtype: dtype[_DTypeScalar],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar]: ...
    # TODO: handle _SupportsDtype better
    @overload
    def __new__(
        cls,
        dtype: _SupportsDtype,
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[Any]: ...
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
        dtype: _VoidDtypeLike,
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[void]: ...
    def __eq__(self, other: DtypeLike) -> bool: ...
    def __ne__(self, other: DtypeLike) -> bool: ...
    def __gt__(self, other: DtypeLike) -> bool: ...
    def __ge__(self, other: DtypeLike) -> bool: ...
    def __lt__(self, other: DtypeLike) -> bool: ...
    def __le__(self, other: DtypeLike) -> bool: ...
    @property
    def alignment(self) -> int: ...
    @property
    def base(self) -> dtype: ...
    @property
    def byteorder(self) -> str: ...
    @property
    def char(self) -> str: ...
    @property
    def descr(self) -> List[Union[Tuple[str, str], Tuple[str, str, _Shape]]]: ...
    @property
    def fields(
        self,
    ) -> Optional[Mapping[str, Union[Tuple[dtype, int], Tuple[dtype, int, Any]]]]: ...
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
    def subdtype(self) -> Optional[Tuple[dtype, _Shape]]: ...
    def newbyteorder(self, __new_order: _ByteOrder = ...) -> dtype: ...
    # Leave str and type for end to avoid having to use `builtins.str`
    # everywhere. See https://github.com/python/mypy/issues/3775
    @property
    def str(self) -> builtins.str: ...
    @property
    def type(self) -> Type[generic]: ...

_Dtype = dtype  # to avoid name conflicts with ndarray.dtype

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

class flatiter(Generic[_ArraySelf]):
    @property
    def base(self) -> _ArraySelf: ...
    @property
    def coords(self) -> _Shape: ...
    @property
    def index(self) -> int: ...
    def copy(self) -> _ArraySelf: ...
    def __iter__(self: _FlatIterSelf) -> _FlatIterSelf: ...
    def __next__(self) -> generic: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(self, key: Union[int, integer]) -> generic: ...
    @overload
    def __getitem__(
        self, key: Union[_ArrayLikeInt, slice, ellipsis],
    ) -> _ArraySelf: ...
    def __array__(self, __dtype: DtypeLike = ...) -> ndarray: ...

_OrderKACF = Optional[Literal["K", "A", "C", "F"]]
_OrderACF = Optional[Literal["A", "C", "F"]]
_OrderCF = Optional[Literal["C", "F"]]

_ModeKind = Literal["raise", "wrap", "clip"]
_PartitionKind = Literal["introselect"]
_SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
_SortSide = Literal["left", "right"]

_ArrayLikeBool = Union[_BoolLike, Sequence[_BoolLike], ndarray]
_ArrayLikeIntOrBool = Union[
    _IntLike,
    _BoolLike,
    ndarray,
    Sequence[_IntLike],
    Sequence[_BoolLike],
    Sequence[Sequence[Any]],  # TODO: wait for support for recursive types
]

_ArraySelf = TypeVar("_ArraySelf", bound=_ArrayOrScalarCommon)

class _ArrayOrScalarCommon(
    SupportsInt, SupportsFloat, SupportsComplex, SupportsBytes, SupportsAbs[Any]
):
    @property
    def T(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def base(self) -> Optional[ndarray]: ...
    @property
    def dtype(self) -> _Dtype: ...
    @property
    def data(self) -> memoryview: ...
    @property
    def flags(self) -> _flagsobj: ...
    @property
    def size(self) -> int: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> _Shape: ...
    @property
    def strides(self) -> _Shape: ...
    def __array__(self, __dtype: DtypeLike = ...) -> ndarray: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __copy__(self: _ArraySelf) -> _ArraySelf: ...
    def __deepcopy__(self: _ArraySelf, __memo: Optional[dict] = ...) -> _ArraySelf: ...
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...
    def __mod__(self, other): ...
    def __rmod__(self, other): ...
    def __divmod__(self, other): ...
    def __rdivmod__(self, other): ...
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    def astype(
        self: _ArraySelf,
        dtype: DtypeLike,
        order: _OrderKACF = ...,
        casting: _Casting = ...,
        subok: bool = ...,
        copy: bool = ...,
    ) -> _ArraySelf: ...
    def byteswap(self: _ArraySelf, inplace: bool = ...) -> _ArraySelf: ...
    def copy(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...
    def dump(self, file: str) -> None: ...
    def dumps(self) -> bytes: ...
    def fill(self, value: Any) -> None: ...
    @property
    def flat(self: _ArraySelf) -> flatiter[_ArraySelf]: ...
    def flatten(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...
    def getfield(
        self: _ArraySelf, dtype: DtypeLike, offset: int = ...
    ) -> _ArraySelf: ...
    @overload
    def item(self, *args: int) -> Any: ...
    @overload
    def item(self, args: Tuple[int, ...]) -> Any: ...
    @overload
    def itemset(self, __value: Any) -> None: ...
    @overload
    def itemset(self, __item: _ShapeLike, __value: Any) -> None: ...
    def ravel(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...
    @overload
    def reshape(
        self: _ArraySelf, shape: Sequence[int], *, order: _OrderACF = ...
    ) -> _ArraySelf: ...
    @overload
    def reshape(
        self: _ArraySelf, *shape: int, order: _OrderACF = ...
    ) -> _ArraySelf: ...
    @overload
    def resize(self, new_shape: Sequence[int], *, refcheck: bool = ...) -> None: ...
    @overload
    def resize(self, *new_shape: int, refcheck: bool = ...) -> None: ...
    def setflags(
        self, write: bool = ..., align: bool = ..., uic: bool = ...
    ) -> None: ...
    def squeeze(
        self: _ArraySelf, axis: Union[int, Tuple[int, ...]] = ...
    ) -> _ArraySelf: ...
    def swapaxes(self: _ArraySelf, axis1: int, axis2: int) -> _ArraySelf: ...
    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...
    # NOTE: `tostring()` is deprecated and therefore excluded
    # def tostring(self, order=...): ...
    def tofile(
        self, fid: Union[IO[bytes], str], sep: str = ..., format: str = ...
    ) -> None: ...
    # generics and 0d arrays return builtin scalars
    def tolist(self) -> Any: ...
    @overload
    def transpose(self: _ArraySelf, axes: Sequence[int]) -> _ArraySelf: ...
    @overload
    def transpose(self: _ArraySelf, *axes: int) -> _ArraySelf: ...
    @overload
    def view(self, type: Type[_NdArraySubClass]) -> _NdArraySubClass: ...
    @overload
    def view(self: _ArraySelf, dtype: DtypeLike = ...) -> _ArraySelf: ...
    @overload
    def view(
        self, dtype: DtypeLike, type: Type[_NdArraySubClass]
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
    def argmax(self, axis: None = ..., out: None = ...) -> signedinteger: ...
    @overload
    def argmax(
        self, axis: _ShapeLike = ..., out: None = ...
    ) -> Union[signedinteger, ndarray]: ...
    @overload
    def argmax(
        self, axis: Optional[_ShapeLike] = ..., out: _NdArraySubClass = ...
    ) -> _NdArraySubClass: ...
    @overload
    def argmin(self, axis: None = ..., out: None = ...) -> signedinteger: ...
    @overload
    def argmin(
        self, axis: _ShapeLike = ..., out: None = ...
    ) -> Union[signedinteger, ndarray]: ...
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
        self, axis: Optional[int] = ..., dtype: DtypeLike = ..., out: None = ...,
    ) -> ndarray: ...
    @overload
    def cumprod(
        self,
        axis: Optional[int] = ...,
        dtype: DtypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def cumsum(
        self, axis: Optional[int] = ..., dtype: DtypeLike = ..., out: None = ...,
    ) -> ndarray: ...
    @overload
    def cumsum(
        self,
        axis: Optional[int] = ...,
        dtype: DtypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def max(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: Literal[False] = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> number: ...
    @overload
    def max(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def max(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def mean(
        self,
        axis: None = ...,
        dtype: DtypeLike = ...,
        out: None = ...,
        keepdims: Literal[False] = ...,
    ) -> number: ...
    @overload
    def mean(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DtypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def mean(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DtypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def min(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: Literal[False] = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> number: ...
    @overload
    def min(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def min(
        self,
        axis: Optional[_ShapeLike] = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> _NdArraySubClass: ...
    def newbyteorder(self: _ArraySelf, __new_order: _ByteOrder = ...) -> _ArraySelf: ...
    @overload
    def prod(
        self,
        axis: None = ...,
        dtype: DtypeLike = ...,
        out: None = ...,
        keepdims: Literal[False] = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> number: ...
    @overload
    def prod(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DtypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def prod(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DtypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike = ...,
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
        dtype: DtypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: Literal[False] = ...,
    ) -> number: ...
    @overload
    def std(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DtypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def std(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DtypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def sum(
        self,
        axis: None = ...,
        dtype: DtypeLike = ...,
        out: None = ...,
        keepdims: Literal[False] = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> number: ...
    @overload
    def sum(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DtypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def sum(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DtypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike = ...,
        where: _ArrayLikeBool = ...,
    ) -> _NdArraySubClass: ...
    @overload
    def take(
        self,
        indices: Union[_IntLike, _BoolLike],
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
        dtype: DtypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: Literal[False] = ...,
    ) -> number: ...
    @overload
    def var(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DtypeLike = ...,
        out: None = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def var(
        self,
        axis: Optional[_ShapeLike] = ...,
        dtype: DtypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: int = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

_BufferType = Union[ndarray, bytes, bytearray, memoryview]
_Casting = Literal["no", "equiv", "safe", "same_kind", "unsafe"]

class ndarray(_ArrayOrScalarCommon, Iterable, Sized, Container):
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
        dtype: DtypeLike = ...,
        buffer: _BufferType = ...,
        offset: int = ...,
        strides: _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> _ArraySelf: ...
    @property
    def dtype(self) -> _Dtype: ...
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
        self, val: ArrayLike, dtype: DtypeLike, offset: int = ...
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
        dtype: DtypeLike = ...,
        out: None = ...,
    ) -> Union[number, ndarray]: ...
    @overload
    def trace(
        self,  # >= 2D array
        offset: int = ...,
        axis1: int = ...,
        axis2: int = ...,
        dtype: DtypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...
    # Many of these special methods are irrelevant currently, since protocols
    # aren't supported yet. That said, I'm adding them for completeness.
    # https://docs.python.org/3/reference/datamodel.html
    def __len__(self) -> int: ...
    def __setitem__(self, key, value): ...
    def __iter__(self) -> Any: ...
    def __contains__(self, key) -> bool: ...
    def __index__(self) -> int: ...
    def __matmul__(self, other): ...
    def __imatmul__(self, other): ...
    def __rmatmul__(self, other): ...
    def __add__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __radd__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __sub__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __rsub__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __mul__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __rmul__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __floordiv__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __rfloordiv__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __pow__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __rpow__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __truediv__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __rtruediv__(self, other: ArrayLike) -> Union[ndarray, generic]: ...
    def __invert__(self: _ArraySelf) -> Union[_ArraySelf, integer, bool_]: ...
    def __lshift__(self, other: ArrayLike) -> Union[ndarray, integer]: ...
    def __rlshift__(self, other: ArrayLike) -> Union[ndarray, integer]: ...
    def __rshift__(self, other: ArrayLike) -> Union[ndarray, integer]: ...
    def __rrshift__(self, other: ArrayLike) -> Union[ndarray, integer]: ...
    def __and__(self, other: ArrayLike) -> Union[ndarray, integer, bool_]: ...
    def __rand__(self, other: ArrayLike) -> Union[ndarray, integer, bool_]: ...
    def __xor__(self, other: ArrayLike) -> Union[ndarray, integer, bool_]: ...
    def __rxor__(self, other: ArrayLike) -> Union[ndarray, integer, bool_]: ...
    def __or__(self, other: ArrayLike) -> Union[ndarray, integer, bool_]: ...
    def __ror__(self, other: ArrayLike) -> Union[ndarray, integer, bool_]: ...
    # `np.generic` does not support inplace operations
    def __iadd__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __isub__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __imul__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __itruediv__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __ifloordiv__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __ipow__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __imod__(self, other): ...
    def __ilshift__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __irshift__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __iand__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __ixor__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...
    def __ior__(self: _ArraySelf, other: ArrayLike) -> _ArraySelf: ...

# NOTE: while `np.generic` is not technically an instance of `ABCMeta`,
# the `@abstractmethod` decorator is herein used to (forcefully) deny
# the creation of `np.generic` instances.
# The `# type: ignore` comments are necessary to silence mypy errors regarding
# the missing `ABCMeta` metaclass.

# See https://github.com/numpy/numpy-stubs/pull/80 for more details.

class generic(_ArrayOrScalarCommon):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def base(self) -> None: ...

class number(generic):  # type: ignore
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
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

class bool_(generic):
    def __init__(self, __value: object = ...) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
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

class object_(generic):
    def __init__(self, __value: object = ...) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...

class datetime64(generic):
    @overload
    def __init__(
        self,
        __value: Union[None, datetime64, _CharLike, dt.datetime] = ...,
        __format: Union[_CharLike, Tuple[_CharLike, _IntLike]] = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        __value: int,
        __format: Union[_CharLike, Tuple[_CharLike, _IntLike]]
    ) -> None: ...
    def __add__(self, other: Union[timedelta64, _IntLike, _BoolLike]) -> datetime64: ...
    def __radd__(self, other: Union[timedelta64, _IntLike, _BoolLike]) -> datetime64: ...
    @overload
    def __sub__(self, other: datetime64) -> timedelta64: ...
    @overload
    def __sub__(self, other: Union[timedelta64, _IntLike, _BoolLike]) -> datetime64: ...
    def __rsub__(self, other: datetime64) -> timedelta64: ...

# Support for `__index__` was added in python 3.8 (bpo-20092)
if sys.version_info >= (3, 8):
    _IntValue = Union[SupportsInt, _CharLike, SupportsIndex]
    _FloatValue = Union[None, _CharLike, SupportsFloat, SupportsIndex]
    _ComplexValue = Union[None, _CharLike, SupportsFloat, SupportsComplex, SupportsIndex]
else:
    _IntValue = Union[SupportsInt, _CharLike]
    _FloatValue = Union[None, _CharLike, SupportsFloat]
    _ComplexValue = Union[None, _CharLike, SupportsFloat, SupportsComplex]

class integer(number):  # type: ignore
    # NOTE: `__index__` is technically defined in the bottom-most
    # sub-classes (`int64`, `uint32`, etc)
    def __index__(self) -> int: ...
    __truediv__: _IntTrueDiv
    __rtruediv__: _IntTrueDiv
    def __invert__(self: _IntType) -> _IntType: ...
    # Ensure that objects annotated as `integer` support bit-wise operations
    def __lshift__(self, other: Union[_IntLike, _BoolLike]) -> integer: ...
    def __rlshift__(self, other: Union[_IntLike, _BoolLike]) -> integer: ...
    def __rshift__(self, other: Union[_IntLike, _BoolLike]) -> integer: ...
    def __rrshift__(self, other: Union[_IntLike, _BoolLike]) -> integer: ...
    def __and__(self, other: Union[_IntLike, _BoolLike]) -> integer: ...
    def __rand__(self, other: Union[_IntLike, _BoolLike]) -> integer: ...
    def __or__(self, other: Union[_IntLike, _BoolLike]) -> integer: ...
    def __ror__(self, other: Union[_IntLike, _BoolLike]) -> integer: ...
    def __xor__(self, other: Union[_IntLike, _BoolLike]) -> integer: ...
    def __rxor__(self, other: Union[_IntLike, _BoolLike]) -> integer: ...

class signedinteger(integer):  # type: ignore
    __add__: _SignedIntOp
    __radd__: _SignedIntOp
    __sub__: _SignedIntOp
    __rsub__: _SignedIntOp
    __mul__: _SignedIntOp
    __rmul__: _SignedIntOp
    __floordiv__: _SignedIntOp
    __rfloordiv__: _SignedIntOp
    __pow__: _SignedIntOp
    __rpow__: _SignedIntOp
    __lshift__: _SignedIntBitOp
    __rlshift__: _SignedIntBitOp
    __rshift__: _SignedIntBitOp
    __rrshift__: _SignedIntBitOp
    __and__: _SignedIntBitOp
    __rand__: _SignedIntBitOp
    __xor__: _SignedIntBitOp
    __rxor__: _SignedIntBitOp
    __or__: _SignedIntBitOp
    __ror__: _SignedIntBitOp

class int8(signedinteger):
    def __init__(self, __value: _IntValue = ...) -> None: ...

class int16(signedinteger):
    def __init__(self, __value: _IntValue = ...) -> None: ...

class int32(signedinteger):
    def __init__(self, __value: _IntValue = ...) -> None: ...

class int64(signedinteger):
    def __init__(self, __value: _IntValue = ...) -> None: ...

class timedelta64(generic):
    def __init__(
        self,
        __value: Union[None, int, _CharLike, dt.timedelta, timedelta64] = ...,
        __format: Union[_CharLike, Tuple[_CharLike, _IntLike]] = ...,
    ) -> None: ...
    def __add__(self, other: Union[timedelta64, _IntLike, _BoolLike]) -> timedelta64: ...
    def __radd__(self, other: Union[timedelta64, _IntLike, _BoolLike]) -> timedelta64: ...
    def __sub__(self, other: Union[timedelta64, _IntLike, _BoolLike]) -> timedelta64: ...
    def __rsub__(self, other: Union[timedelta64, _IntLike, _BoolLike]) -> timedelta64: ...
    def __mul__(self, other: Union[_FloatLike, _BoolLike]) -> timedelta64: ...
    def __rmul__(self, other: Union[_FloatLike, _BoolLike]) -> timedelta64: ...
    __truediv__: _TD64Div[float64]
    __floordiv__: _TD64Div[signedinteger]
    def __rtruediv__(self, other: timedelta64) -> float64: ...
    def __rfloordiv__(self, other: timedelta64) -> signedinteger: ...
    def __mod__(self, other: timedelta64) -> timedelta64: ...

class unsignedinteger(integer):  # type: ignore
    # NOTE: `uint64 + signedinteger -> float64`
    __add__: _UnsignedIntOp
    __radd__: _UnsignedIntOp
    __sub__: _UnsignedIntOp
    __rsub__: _UnsignedIntOp
    __mul__: _UnsignedIntOp
    __rmul__: _UnsignedIntOp
    __floordiv__: _UnsignedIntOp
    __rfloordiv__: _UnsignedIntOp
    __pow__: _UnsignedIntOp
    __rpow__: _UnsignedIntOp
    __lshift__: _UnsignedIntBitOp
    __rlshift__: _UnsignedIntBitOp
    __rshift__: _UnsignedIntBitOp
    __rrshift__: _UnsignedIntBitOp
    __and__: _UnsignedIntBitOp
    __rand__: _UnsignedIntBitOp
    __xor__: _UnsignedIntBitOp
    __rxor__: _UnsignedIntBitOp
    __or__: _UnsignedIntBitOp
    __ror__: _UnsignedIntBitOp

class uint8(unsignedinteger):
    def __init__(self, __value: _IntValue = ...) -> None: ...

class uint16(unsignedinteger):
    def __init__(self, __value: _IntValue = ...) -> None: ...

class uint32(unsignedinteger):
    def __init__(self, __value: _IntValue = ...) -> None: ...

class uint64(unsignedinteger):
    def __init__(self, __value: _IntValue = ...) -> None: ...

class inexact(number): ...  # type: ignore

class floating(inexact):  # type: ignore
    __add__: _FloatOp
    __radd__: _FloatOp
    __sub__: _FloatOp
    __rsub__: _FloatOp
    __mul__: _FloatOp
    __rmul__: _FloatOp
    __truediv__: _FloatOp
    __rtruediv__: _FloatOp
    __floordiv__: _FloatOp
    __rfloordiv__: _FloatOp
    __pow__: _FloatOp
    __rpow__: _FloatOp

_IntType = TypeVar("_IntType", bound=integer)
_FloatType = TypeVar('_FloatType', bound=floating)

class float16(floating):
    def __init__(self, __value: _FloatValue = ...) -> None: ...

class float32(floating):
    def __init__(self, __value: _FloatValue = ...) -> None: ...

class float64(floating, float):
    def __init__(self, __value: _FloatValue = ...) -> None: ...

class complexfloating(inexact, Generic[_FloatType]):  # type: ignore
    @property
    def real(self) -> _FloatType: ...  # type: ignore[override]
    @property
    def imag(self) -> _FloatType: ...  # type: ignore[override]
    def __abs__(self) -> _FloatType: ...  # type: ignore[override]
    __add__: _ComplexOp
    __radd__: _ComplexOp
    __sub__: _ComplexOp
    __rsub__: _ComplexOp
    __mul__: _ComplexOp
    __rmul__: _ComplexOp
    __truediv__: _ComplexOp
    __rtruediv__: _ComplexOp
    __floordiv__: _ComplexOp
    __rfloordiv__: _ComplexOp
    __pow__: _ComplexOp
    __rpow__: _ComplexOp

class complex64(complexfloating[float32]):
    def __init__(self, __value: _ComplexValue = ...) -> None: ...

class complex128(complexfloating[float64], complex):
    def __init__(self, __value: _ComplexValue = ...) -> None: ...

class flexible(generic): ...  # type: ignore

class void(flexible):
    def __init__(self, __value: Union[_IntLike, _BoolLike, bytes]): ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def setfield(
        self, val: ArrayLike, dtype: DtypeLike, offset: int = ...
    ) -> None: ...

class character(flexible): ...  # type: ignore

# NOTE: Most `np.bytes_` / `np.str_` methods return their
# builtin `bytes` / `str` counterpart

class bytes_(character, bytes):
    @overload
    def __init__(self, __value: object = ...) -> None: ...
    @overload
    def __init__(
        self, __value: str, encoding: str = ..., errors: str = ...
    ) -> None: ...

class str_(character, str):
    @overload
    def __init__(self, __value: object = ...) -> None: ...
    @overload
    def __init__(
        self, __value: bytes, encoding: str = ..., errors: str = ...
    ) -> None: ...

unicode_ = str0 = str_

# TODO(alan): Platform dependent types
# longcomplex, longdouble, longfloat
# bytes, short, intc, intp, longlong
# half, single, double, longdouble
# uint_, int_, float_, complex_
# float128, complex256
# float96

def array(
    object: object,
    dtype: DtypeLike = ...,
    *,
    copy: bool = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    ndmin: int = ...,
    like: ArrayLike = ...,
) -> ndarray: ...
def zeros(
    shape: _ShapeLike,
    dtype: DtypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> ndarray: ...
def empty(
    shape: _ShapeLike,
    dtype: DtypeLike = ...,
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
        dtype: DtypeLike = ...,
        subok: bool = ...,
        signature: Union[str, Tuple[str]] = ...,
        # In reality this should be a length of list 3 containing an
        # int, an int, and a callable, but there's no way to express
        # that.
        extobj: List[Union[int, Callable]] = ...,
    ) -> Union[ndarray, generic]: ...
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
