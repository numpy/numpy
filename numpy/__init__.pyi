import builtins
import sys
import datetime as dt
from abc import abstractmethod

from numpy.core._internal import _ctypes
from numpy.typing import ArrayLike, DtypeLike, _Shape, _ShapeLike

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
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    Text,
    Tuple,
    Type,
    TypeVar,
    Union,
)

if sys.version_info[0] < 3:
    class SupportsBytes: ...

else:
    from typing import SupportsBytes

if sys.version_info >= (3, 8):
    from typing import Literal, Protocol
else:
    from typing_extensions import Literal, Protocol

# TODO: remove when the full numpy namespace is defined
def __getattr__(name: str) -> Any: ...

_NdArraySubClass = TypeVar("_NdArraySubClass", bound=ndarray)
_ByteOrder = Literal["S", "<", ">", "=", "|", "L", "B", "N", "I"]

class dtype:
    names: Optional[Tuple[str, ...]]
    def __init__(
        self,
        dtype: DtypeLike,
        align: bool = ...,
        copy: bool = ...,
    ) -> None: ...
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
    if sys.version_info[0] < 3:
        def __oct__(self) -> str: ...
        def __hex__(self) -> str: ...
        def __nonzero__(self) -> bool: ...
        def __unicode__(self) -> Text: ...
    else:
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
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def __iadd__(self, other): ...
    def __sub__(self, other): ...
    def __rsub__(self, other): ...
    def __isub__(self, other): ...
    def __mul__(self, other): ...
    def __rmul__(self, other): ...
    def __imul__(self, other): ...
    if sys.version_info[0] < 3:
        def __div__(self, other): ...
        def __rdiv__(self, other): ...
        def __idiv__(self, other): ...
    def __truediv__(self, other): ...
    def __rtruediv__(self, other): ...
    def __itruediv__(self, other): ...
    def __floordiv__(self, other): ...
    def __rfloordiv__(self, other): ...
    def __ifloordiv__(self, other): ...
    def __mod__(self, other): ...
    def __rmod__(self, other): ...
    def __imod__(self, other): ...
    def __divmod__(self, other): ...
    def __rdivmod__(self, other): ...
    # NumPy's __pow__ doesn't handle a third argument
    def __pow__(self, other): ...
    def __rpow__(self, other): ...
    def __ipow__(self, other): ...
    def __lshift__(self, other): ...
    def __rlshift__(self, other): ...
    def __ilshift__(self, other): ...
    def __rshift__(self, other): ...
    def __rrshift__(self, other): ...
    def __irshift__(self, other): ...
    def __and__(self, other): ...
    def __rand__(self, other): ...
    def __iand__(self, other): ...
    def __xor__(self, other): ...
    def __rxor__(self, other): ...
    def __ixor__(self, other): ...
    def __or__(self, other): ...
    def __ror__(self, other): ...
    def __ior__(self, other): ...
    if sys.version_info[:2] >= (3, 5):
        def __matmul__(self, other): ...
        def __rmatmul__(self, other): ...
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    def __invert__(self: _ArraySelf) -> _ArraySelf: ...
    # TODO(shoyer): remove when all methods are defined
    def __getattr__(self, name) -> Any: ...

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
    def flat(self: _ArraySelf) -> flatiter[_ArraySelf]: ...
    @property
    def strides(self) -> _Shape: ...
    @strides.setter
    def strides(self, value: _ShapeLike): ...
    # Array conversion
    @overload
    def item(self, *args: int) -> Any: ...
    @overload
    def item(self, args: Tuple[int, ...]) -> Any: ...
    def tolist(self) -> List[Any]: ...
    @overload
    def itemset(self, __value: Any) -> None: ...
    @overload
    def itemset(self, __item: _ShapeLike, __value: Any) -> None: ...
    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...
    def tofile(
        self, fid: Union[IO[bytes], str], sep: str = ..., format: str = ...
    ) -> None: ...
    def dump(self, file: str) -> None: ...
    def dumps(self) -> bytes: ...
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
    @overload
    def view(self, type: Type[_NdArraySubClass]) -> _NdArraySubClass: ...
    @overload
    def view(self: _ArraySelf, dtype: DtypeLike = ...) -> _ArraySelf: ...
    @overload
    def view(
        self, dtype: DtypeLike, type: Type[_NdArraySubClass]
    ) -> _NdArraySubClass: ...
    def getfield(
        self: _ArraySelf, dtype: DtypeLike, offset: int = ...
    ) -> _ArraySelf: ...
    def setflags(
        self, write: bool = ..., align: bool = ..., uic: bool = ...
    ) -> None: ...
    def fill(self, value: Any) -> None: ...
    # Shape manipulation
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
    @overload
    def transpose(self: _ArraySelf, axes: Sequence[int]) -> _ArraySelf: ...
    @overload
    def transpose(self: _ArraySelf, *axes: int) -> _ArraySelf: ...
    def swapaxes(self: _ArraySelf, axis1: int, axis2: int) -> _ArraySelf: ...
    def flatten(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...
    def ravel(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...
    def squeeze(
        self: _ArraySelf, axis: Union[int, Tuple[int, ...]] = ...
    ) -> _ArraySelf: ...
    # Many of these special methods are irrelevant currently, since protocols
    # aren't supported yet. That said, I'm adding them for completeness.
    # https://docs.python.org/3/reference/datamodel.html
    def __len__(self) -> int: ...
    def __getitem__(self, key) -> Any: ...
    def __setitem__(self, key, value): ...
    def __iter__(self) -> Any: ...
    def __contains__(self, key) -> bool: ...
    def __index__(self) -> int: ...

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

class _real_generic(generic):  # type: ignore
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...

class number(generic): ...  # type: ignore

class bool_(_real_generic):
    def __init__(self, __value: object = ...) -> None: ...

class object_(generic):
    def __init__(self, __value: object = ...) -> None: ...

class datetime64:
    @overload
    def __init__(
        self,
        __value: Union[None, datetime64, str, dt.datetime] = ...,
        __format: str = ...
    ) -> None: ...
    @overload
    def __init__(self, __value: int, __format: str) -> None: ...
    def __add__(self, other: Union[timedelta64, int]) -> datetime64: ...
    def __sub__(self, other: Union[timedelta64, datetime64, int]) -> timedelta64: ...

class integer(number, _real_generic): ...  # type: ignore
class signedinteger(integer): ...  # type: ignore

class int8(signedinteger):
    def __init__(self, __value: SupportsInt = ...) -> None: ...

class int16(signedinteger):
    def __init__(self, __value: SupportsInt = ...) -> None: ...

class int32(signedinteger):
    def __init__(self, __value: SupportsInt = ...) -> None: ...

class int64(signedinteger):
    def __init__(self, __value: SupportsInt = ...) -> None: ...

class timedelta64(signedinteger):
    def __init__(self, __value: Any = ..., __format: str = ...) -> None: ...
    @overload
    def __add__(self, other: Union[timedelta64, int]) -> timedelta64: ...
    @overload
    def __add__(self, other: datetime64) -> datetime64: ...
    def __sub__(self, other: Union[timedelta64, int]) -> timedelta64: ...
    if sys.version_info[0] < 3:
        @overload
        def __div__(self, other: timedelta64) -> float: ...
        @overload
        def __div__(self, other: float) -> timedelta64: ...
    @overload
    def __truediv__(self, other: timedelta64) -> float: ...
    @overload
    def __truediv__(self, other: float) -> timedelta64: ...
    def __mod__(self, other: timedelta64) -> timedelta64: ...

class unsignedinteger(integer): ...  # type: ignore

class uint8(unsignedinteger):
    def __init__(self, __value: SupportsInt = ...) -> None: ...

class uint16(unsignedinteger):
    def __init__(self, __value: SupportsInt = ...) -> None: ...

class uint32(unsignedinteger):
    def __init__(self, __value: SupportsInt = ...) -> None: ...

class uint64(unsignedinteger):
    def __init__(self, __value: SupportsInt = ...) -> None: ...

class inexact(number): ...  # type: ignore
class floating(inexact, _real_generic): ...  # type: ignore

_FloatType = TypeVar('_FloatType', bound=floating)

class float16(floating):
    def __init__(self, __value: Optional[SupportsFloat] = ...) -> None: ...

class float32(floating):
    def __init__(self, __value: Optional[SupportsFloat] = ...) -> None: ...

class float64(floating):
    def __init__(self, __value: Optional[SupportsFloat] = ...) -> None: ...

class complexfloating(inexact, Generic[_FloatType]):  # type: ignore
    @property
    def real(self) -> _FloatType: ...
    @property
    def imag(self) -> _FloatType: ...
    def __abs__(self) -> _FloatType: ...  # type: ignore[override]

class complex64(complexfloating[float32]):
    def __init__(
        self,
        __value: Union[None, SupportsInt, SupportsFloat, SupportsComplex] = ...
    ) -> None: ...

class complex128(complexfloating[float64]):
    def __init__(
        self,
        __value: Union[None, SupportsInt, SupportsFloat, SupportsComplex] = ...
    ) -> None: ...

class flexible(_real_generic): ...  # type: ignore

class void(flexible):
    def __init__(self, __value: Union[int, integer, bool_, bytes, bytes_]): ...

class character(_real_generic): ...  # type: ignore

class bytes_(character):
    @overload
    def __init__(self, __value: object = ...) -> None: ...
    @overload
    def __init__(
        self, __value: Union[str, str_], encoding: str = ..., errors: str = ...
    ) -> None: ...

class str_(character):
    @overload
    def __init__(self, __value: object = ...) -> None: ...
    @overload
    def __init__(
        self, __value: Union[bytes, bytes_], encoding: str = ..., errors: str = ...
    ) -> None: ...

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
def ones(
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
def zeros_like(
    a: ArrayLike,
    dtype: DtypeLike = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[Union[int, Sequence[int]]] = ...,
) -> ndarray: ...
def ones_like(
    a: ArrayLike,
    dtype: DtypeLike = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> ndarray: ...
def empty_like(
    a: ArrayLike,
    dtype: DtypeLike = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> ndarray: ...
def full(
    shape: _ShapeLike,
    fill_value: Any,
    dtype: DtypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
) -> ndarray: ...
def full_like(
    a: ArrayLike,
    fill_value: Any,
    dtype: DtypeLike = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[_ShapeLike] = ...,
) -> ndarray: ...
def count_nonzero(
    a: ArrayLike, axis: Optional[Union[int, Tuple[int], Tuple[int, int]]] = ...
) -> Union[int, ndarray]: ...
def isfortran(a: ndarray) -> bool: ...
def argwhere(a: ArrayLike) -> ndarray: ...
def flatnonzero(a: ArrayLike) -> ndarray: ...

_CorrelateMode = Literal["valid", "same", "full"]

def correlate(a: ArrayLike, v: ArrayLike, mode: _CorrelateMode = ...) -> ndarray: ...
def convolve(a: ArrayLike, v: ArrayLike, mode: _CorrelateMode = ...) -> ndarray: ...
def outer(a: ArrayLike, b: ArrayLike, out: ndarray = ...) -> ndarray: ...
def tensordot(
    a: ArrayLike,
    b: ArrayLike,
    axes: Union[
        int, Tuple[int, int], Tuple[Tuple[int, int], ...], Tuple[List[int, int], ...]
    ] = ...,
) -> ndarray: ...
def roll(
    a: ArrayLike,
    shift: Union[int, Tuple[int, ...]],
    axis: Optional[Union[int, Tuple[int, ...]]] = ...,
) -> ndarray: ...
def rollaxis(a: ArrayLike, axis: int, start: int = ...) -> ndarray: ...
def moveaxis(
    a: ndarray,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
) -> ndarray: ...
def cross(
    a: ArrayLike,
    b: ArrayLike,
    axisa: int = ...,
    axisb: int = ...,
    axisc: int = ...,
    axis: Optional[int] = ...,
) -> ndarray: ...
def indices(
    dimensions: Sequence[int], dtype: dtype = ..., sparse: bool = ...
) -> Union[ndarray, Tuple[ndarray, ...]]: ...
def fromfunction(
    function: Callable,
    shape: Tuple[int, int],
    *,
    like: ArrayLike = ...,
    **kwargs,
) -> Any: ...
def isscalar(element: Any) -> bool: ...
def binary_repr(num: int, width: Optional[int] = ...) -> str: ...
def base_repr(number: int, base: int = ..., padding: int = ...) -> str: ...
def identity(n: int, dtype: DtypeLike = ..., *, like: ArrayLike = ...) -> ndarray: ...
def allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> bool: ...
def isclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: float = ...,
    atol: float = ...,
    equal_nan: bool = ...,
) -> Union[bool_, ndarray]: ...
def array_equal(a1: ArrayLike, a2: ArrayLike) -> bool: ...
def array_equiv(a1: ArrayLike, a2: ArrayLike) -> bool: ...

#
# Constants
#

Inf: float
Infinity: float
NAN: float
NINF: float
NZERO: float
NaN: float
PINF: float
PZERO: float
e: float
euler_gamma: float
inf: float
infty: float
nan: float
pi: float

ALLOW_THREADS: int
BUFSIZE: int
CLIP: int
ERR_CALL: int
ERR_DEFAULT: int
ERR_IGNORE: int
ERR_LOG: int
ERR_PRINT: int
ERR_RAISE: int
ERR_WARN: int
FLOATING_POINT_SUPPORT: int
FPE_DIVIDEBYZERO: int
FPE_INVALID: int
FPE_OVERFLOW: int
FPE_UNDERFLOW: int
MAXDIMS: int
MAY_SHARE_BOUNDS: int
MAY_SHARE_EXACT: int
RAISE: int
SHIFT_DIVIDEBYZERO: int
SHIFT_INVALID: int
SHIFT_OVERFLOW: int
SHIFT_UNDERFLOW: int
UFUNC_BUFSIZE_DEFAULT: int
WRAP: int
little_endian: int
tracemalloc_domain: int

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

# Functions from np.core.numerictypes
_DefaultType = TypeVar("_DefaultType")

def maximum_sctype(t: DtypeLike) -> dtype: ...
def issctype(rep: object) -> bool: ...
@overload
def obj2sctype(rep: object) -> Optional[generic]: ...
@overload
def obj2sctype(rep: object, default: None) -> Optional[generic]: ...
@overload
def obj2sctype(
    rep: object, default: Type[_DefaultType]
) -> Union[generic, Type[_DefaultType]]: ...
def issubclass_(arg1: object, arg2: Union[object, Tuple[object, ...]]) -> bool: ...
def issubsctype(
    arg1: Union[ndarray, DtypeLike], arg2: Union[ndarray, DtypeLike]
) -> bool: ...
def issubdtype(arg1: DtypeLike, arg2: DtypeLike) -> bool: ...
def sctype2char(sctype: object) -> str: ...
def find_common_type(
    array_types: Sequence[DtypeLike], scalar_types: Sequence[DtypeLike]
) -> dtype: ...

# Functions from np.core.fromnumeric
_Mode = Literal["raise", "wrap", "clip"]
_PartitionKind = Literal["introselect"]
_SortKind = Literal["quicksort", "mergesort", "heapsort", "stable"]
_Side = Literal["left", "right"]

# Various annotations for scalars

# While dt.datetime and dt.timedelta are not technically part of NumPy,
# they are one of the rare few builtin scalars which serve as valid return types.
# See https://github.com/numpy/numpy-stubs/pull/67#discussion_r412604113.
_ScalarNumpy = Union[generic, dt.datetime, dt.timedelta]
_ScalarBuiltin = Union[str, bytes, dt.date, dt.timedelta, bool, int, float, complex]
_Scalar = Union[_ScalarBuiltin, _ScalarNumpy]

# Integers and booleans can generally be used interchangeably
_ScalarIntOrBool = TypeVar("_ScalarIntOrBool", bound=Union[integer, bool_])
_ScalarGeneric = TypeVar("_ScalarGeneric", bound=generic)
_ScalarGenericDT = TypeVar(
    "_ScalarGenericDT", bound=Union[dt.datetime, dt.timedelta, generic]
)

_Number = TypeVar('_Number', bound=number)
_NumberLike = Union[int, float, complex, number, bool_]

# An array-like object consisting of integers
_Int = Union[int, integer]
_Bool = Union[bool, bool_]
_IntOrBool = Union[_Int, _Bool]
_ArrayLikeIntNested = ArrayLike  # TODO: wait for support for recursive types
_ArrayLikeBoolNested = ArrayLike  # TODO: wait for support for recursive types

# Integers and booleans can generally be used interchangeably
_ArrayLikeIntOrBool = Union[
    _IntOrBool,
    ndarray,
    Sequence[_IntOrBool],
    Sequence[_ArrayLikeIntNested],
    Sequence[_ArrayLikeBoolNested],
]
_ArrayLikeBool = Union[
    _Bool,
    Sequence[_Bool],
    ndarray
]

# The signature of take() follows a common theme with its overloads:
# 1. A generic comes in; the same generic comes out
# 2. A scalar comes in; a generic comes out
# 3. An array-like object comes in; some keyword ensures that a generic comes out
# 4. An array-like object comes in; an ndarray or generic comes out
@overload
def take(
    a: _ScalarGenericDT,
    indices: int,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
    mode: _Mode = ...,
) -> _ScalarGenericDT: ...
@overload
def take(
    a: _Scalar,
    indices: int,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
    mode: _Mode = ...,
) -> _ScalarNumpy: ...
@overload
def take(
    a: ArrayLike,
    indices: int,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
    mode: _Mode = ...,
) -> _ScalarNumpy: ...
@overload
def take(
    a: ArrayLike,
    indices: _ArrayLikeIntOrBool,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
    mode: _Mode = ...,
) -> Union[_ScalarNumpy, ndarray]: ...
def reshape(a: ArrayLike, newshape: _ShapeLike, order: _OrderACF = ...) -> ndarray: ...
@overload
def choose(
    a: _ScalarIntOrBool,
    choices: ArrayLike,
    out: Optional[ndarray] = ...,
    mode: _Mode = ...,
) -> _ScalarIntOrBool: ...
@overload
def choose(
    a: _IntOrBool, choices: ArrayLike, out: Optional[ndarray] = ..., mode: _Mode = ...
) -> Union[integer, bool_]: ...
@overload
def choose(
    a: _ArrayLikeIntOrBool,
    choices: ArrayLike,
    out: Optional[ndarray] = ...,
    mode: _Mode = ...,
) -> ndarray: ...
def repeat(
    a: ArrayLike, repeats: _ArrayLikeIntOrBool, axis: Optional[int] = ...
) -> ndarray: ...
def put(
    a: ndarray, ind: _ArrayLikeIntOrBool, v: ArrayLike, mode: _Mode = ...
) -> None: ...
def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> ndarray: ...
def transpose(
    a: ArrayLike, axes: Union[None, Sequence[int], ndarray] = ...
) -> ndarray: ...
def partition(
    a: ArrayLike,
    kth: _ArrayLikeIntOrBool,
    axis: Optional[int] = ...,
    kind: _PartitionKind = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
@overload
def argpartition(
    a: generic,
    kth: _ArrayLikeIntOrBool,
    axis: Optional[int] = ...,
    kind: _PartitionKind = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> integer: ...
@overload
def argpartition(
    a: _ScalarBuiltin,
    kth: _ArrayLikeIntOrBool,
    axis: Optional[int] = ...,
    kind: _PartitionKind = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
@overload
def argpartition(
    a: ArrayLike,
    kth: _ArrayLikeIntOrBool,
    axis: Optional[int] = ...,
    kind: _PartitionKind = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
def sort(
    a: ArrayLike,
    axis: Optional[int] = ...,
    kind: Optional[_SortKind] = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
def argsort(
    a: ArrayLike,
    axis: Optional[int] = ...,
    kind: Optional[_SortKind] = ...,
    order: Union[None, str, Sequence[str]] = ...,
) -> ndarray: ...
@overload
def argmax(a: ArrayLike, axis: None = ..., out: Optional[ndarray] = ...) -> integer: ...
@overload
def argmax(
    a: ArrayLike, axis: int = ..., out: Optional[ndarray] = ...
) -> Union[integer, ndarray]: ...
@overload
def argmin(a: ArrayLike, axis: None = ..., out: Optional[ndarray] = ...) -> integer: ...
@overload
def argmin(
    a: ArrayLike, axis: int = ..., out: Optional[ndarray] = ...
) -> Union[integer, ndarray]: ...
@overload
def searchsorted(
    a: ArrayLike,
    v: _Scalar,
    side: _Side = ...,
    sorter: Optional[_ArrayLikeIntOrBool] = ...,  # 1D int array
) -> integer: ...
@overload
def searchsorted(
    a: ArrayLike,
    v: ArrayLike,
    side: _Side = ...,
    sorter: Optional[_ArrayLikeIntOrBool] = ...,  # 1D int array
) -> ndarray: ...
def resize(a: ArrayLike, new_shape: _ShapeLike) -> ndarray: ...
@overload
def squeeze(a: _ScalarGeneric, axis: Optional[_ShapeLike] = ...) -> _ScalarGeneric: ...
@overload
def squeeze(a: ArrayLike, axis: Optional[_ShapeLike] = ...) -> ndarray: ...
def diagonal(
    a: ArrayLike, offset: int = ..., axis1: int = ..., axis2: int = ...  # >= 2D array
) -> ndarray: ...
def trace(
    a: ArrayLike,  # >= 2D array
    offset: int = ...,
    axis1: int = ...,
    axis2: int = ...,
    dtype: DtypeLike = ...,
    out: Optional[ndarray] = ...,
) -> Union[number, ndarray]: ...
def ravel(a: ArrayLike, order: _OrderKACF = ...) -> ndarray: ...
def nonzero(a: ArrayLike) -> Tuple[ndarray, ...]: ...
def shape(a: ArrayLike) -> _Shape: ...
def compress(
    condition: ArrayLike,  # 1D bool array
    a: ArrayLike,
    axis: Optional[int] = ...,
    out: Optional[ndarray] = ...,
) -> ndarray: ...
@overload
def clip(
    a: _Number,
    a_min: ArrayLike,
    a_max: Optional[ArrayLike],
    out: Optional[ndarray] = ...,
    **kwargs: Any,
) -> _Number: ...
@overload
def clip(
    a: _Number,
    a_min: None,
    a_max: ArrayLike,
    out: Optional[ndarray] = ...,
    **kwargs: Any,
) -> _Number: ...
@overload
def clip(
    a: ArrayLike,
    a_min: ArrayLike,
    a_max: Optional[ArrayLike],
    out: Optional[ndarray] = ...,
    **kwargs: Any,
) -> Union[number, ndarray]: ...
@overload
def clip(
    a: ArrayLike,
    a_min: None,
    a_max: ArrayLike,
    out: Optional[ndarray] = ...,
    **kwargs: Any,
) -> Union[number, ndarray]: ...
@overload
def sum(
    a: _Number,
    axis: Optional[_ShapeLike] = ...,
    dtype: DtypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> _Number: ...
@overload
def sum(
    a: ArrayLike,
    axis: _ShapeLike = ...,
    dtype: DtypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> Union[number, ndarray]: ...
@overload
def all(
    a: ArrayLike,
    axis: None = ...,
    out: Optional[ndarray] = ...,
    keepdims: Literal[False] = ...,
) -> bool_: ...
@overload
def all(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Union[bool_, ndarray]: ...
@overload
def any(
    a: ArrayLike,
    axis: None = ...,
    out: Optional[ndarray] = ...,
    keepdims: Literal[False] = ...,
) -> bool_: ...
@overload
def any(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Union[bool_, ndarray]: ...
def cumsum(
    a: ArrayLike,
    axis: Optional[int] = ...,
    dtype: DtypeLike = ...,
    out: Optional[ndarray] = ...,
) -> ndarray: ...
@overload
def ptp(
    a: _Number,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> _Number: ...
@overload
def ptp(
    a: ArrayLike,
    axis: None = ...,
    out: Optional[ndarray] = ...,
    keepdims: Literal[False] = ...,
) -> number: ...
@overload
def ptp(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Union[number, ndarray]: ...
@overload
def amax(
    a: _Number,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> _Number: ...
@overload
def amax(
    a: ArrayLike,
    axis: None = ...,
    out: Optional[ndarray] = ...,
    keepdims: Literal[False] = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> number: ...
@overload
def amax(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> Union[number, ndarray]: ...
@overload
def amin(
    a: _Number,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> _Number: ...
@overload
def amin(
    a: ArrayLike,
    axis: None = ...,
    out: Optional[ndarray] = ...,
    keepdims: Literal[False] = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> number: ...
@overload
def amin(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> Union[number, ndarray]: ...

# TODO: `np.prod()``: For object arrays `initial` does not necessarily
# have to be a numerical scalar.
# The only requirement is that it is compatible
# with the `.__mul__()` method(s) of the passed array's elements.

# Note that the same situation holds for all wrappers around
# `np.ufunc.reduce`, e.g. `np.sum()` (`.__add__()`).

@overload
def prod(
    a: _Number,
    axis: Optional[_ShapeLike] = ...,
    dtype: DtypeLike = ...,
    out: None = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> _Number: ...
@overload
def prod(
    a: ArrayLike,
    axis: None = ...,
    dtype: DtypeLike = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> number: ...
@overload
def prod(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DtypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
    initial: _NumberLike = ...,
    where: _ArrayLikeBool = ...,
) -> Union[number, ndarray]: ...
def cumprod(
    a: ArrayLike,
    axis: Optional[int] = ...,
    dtype: DtypeLike = ...,
    out: Optional[ndarray] = ...,
) -> ndarray: ...
def ndim(a: ArrayLike) -> int: ...
def size(a: ArrayLike, axis: Optional[int] = ...) -> int: ...
@overload
def around(
    a: _Number, decimals: int = ..., out: Optional[ndarray] = ...
) -> _Number: ...
@overload
def around(
    a: _NumberLike, decimals: int = ..., out: Optional[ndarray] = ...
) -> number: ...
@overload
def around(
    a: ArrayLike, decimals: int = ..., out: Optional[ndarray] = ...
) -> ndarray: ...
@overload
def mean(
    a: ArrayLike,
    axis: None = ...,
    dtype: DtypeLike = ...,
    out: None = ...,
    keepdims: Literal[False] = ...,
) -> number: ...
@overload
def mean(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DtypeLike = ...,
    out: Optional[ndarray] = ...,
    keepdims: bool = ...,
) -> Union[number, ndarray]: ...
@overload
def std(
    a: ArrayLike,
    axis: None = ...,
    dtype: DtypeLike = ...,
    out: None = ...,
    ddof: int = ...,
    keepdims: Literal[False] = ...,
) -> number: ...
@overload
def std(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DtypeLike = ...,
    out: Optional[ndarray] = ...,
    ddof: int = ...,
    keepdims: bool = ...,
) -> Union[number, ndarray]: ...
@overload
def var(
    a: ArrayLike,
    axis: None = ...,
    dtype: DtypeLike = ...,
    out: None = ...,
    ddof: int = ...,
    keepdims: Literal[False] = ...,
) -> number: ...
@overload
def var(
    a: ArrayLike,
    axis: Optional[_ShapeLike] = ...,
    dtype: DtypeLike = ...,
    out: Optional[ndarray] = ...,
    ddof: int = ...,
    keepdims: bool = ...,
) -> Union[number, ndarray]: ...
