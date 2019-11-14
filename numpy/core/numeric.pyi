import numbers
from types import ModuleType
from typing import Sequence, Optional, TypeVar, Union, Tuple, List, Iterable, Callable, Any

from numpy import (ndarray, dtype,
                   bool_, bool8,
                   byte, short, intc, int_, longlong, intp, int8, int16, int32, int64,
                   ubyte, ushort, uintc, uint, ulonglong, uintp, uint8, uint16, uint32, uint64,
                   half, single, double, float_, longfloat, float36, float32, float64, float96, float128,
                   csingle, complex_, clongfloat, complex64, complex128, complex192, complex256, generic)

T = TypeVar('T')
S = TypeVar['S']

NShape = Union[int, Sequence[int]]
OShape = Optional[NShape]

ODType = Optional[dtype]

# See https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html
Scalar = Union[generic,
               numbers.Number,
               bool_, bool8,
               byte, short, intc, int_, longlong, intp, int8, int16, int32, int64,
               ubyte, ushort, uintc, uint, ulonglong, uintp, uint8, uint16, uint32, uint64,
               half, single, double, float_, longfloat, float36, float32, float64, float96, float128,
               csingle, complex_, clongfloat, complex64, complex128, complex192, complex256]

TScalar = TypeVar('TScalar', bound=Scalar)


def zeros_like(
        a: T,
        dtype: ODType = None,
        order: str = 'K',
        subok: bool = True,
        shape: OShape = None) -> ndarray[T]: ...


def ones(
        shape: NShape,
        dtype: ODType = None,
        order: str = 'C') -> ndarray[dtype]: ...


def ones_like(
        a: T,
        dtype: ODType = None,
        order: str = 'K',
        subok: bool = True,
        shape: OShape = None) -> ndarray[T]: ...


def full(
        shape: NShape,
        fill_value: TScalar,
        dtype: ODType = None,
        order: str = 'C') -> ndarray[TScalar]:  ...


def full_like(
        a: T,
        fill_value: TScalar,
        dtype: ODType = None,
        order: str = 'K',
        subok: bool = True,
        shape: OShape = None) -> ndarray[TScalar]: ...


def count_nonzero(
        a: T,
        axis: Optional[Union[int, Tuple[int]]] = None) -> Union[int, Sequence[int]]: ...


def isfortran(a: ndarray) -> bool: ...


def argwhere(a: T) -> ndarray: ...


def flatnonzero(a: T) -> ndarray: ...


def correlate(a: T, v: S, mode: str = 'valid') -> ndarray: ...


def convolve(a: T, v: S, mode: str = 'full') -> ndarray: ...


def outer(a: T, b: S, out: ndarray = None) -> ndarray: ...


def tensordot(
        a: T,
        b: S,
        axes: Union[int, Tuple[int, int], Tuple[Tuple[int, int], ...], Tuple[List[int, int], ...]] = 2) -> ndarray: ...


def roll(a: T,
         shift: Union[int, Tuple[int, ...]],
         axis: Optional[Union[int, Tuple[int, ...]]] = None) -> T: ...


def rollaxis(
        a: T,
        axis: int,
        start: int = 0) -> T: ...


def normalize_axis_tuple(
        axis: Union[int, Iterable[int]],
        ndim: int,
        argname: Optional[str] = None,
        allow_duplicate: bool = False) -> Tuple[int]: ...


def moveaxis(
        a: ndarray,
        source: Union[int, Sequence[int]],
        destination: Union[int, Sequence[int]]) -> ndarray: ...


def cross(
        a: T,
        b: S,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: Optional[int] = None) -> ndarray: ...


def indices(
        dimensions: Sequence[int],
        dtype: dtype = int,
        sparse: bool = False) -> Union[ndarray, Tuple[ndarray, ...]]: ...


def fromfunction(
        function: Callable,
        shape: Tuple[int, int],
        **kwargs) -> Any: ...


def isscalar(element: Any) -> bool: ...


def binary_repr(num: int, width: Optional[int] = None) -> str: ...


def base_repr(number: int, base: int = 2, padding: int = 0) -> str: ...


def identity(n: int, dtype: Optional[dtype] = None) -> ndarray: ...


def allclose(
        a: T,
        b: S,
        rtol: float = 1.e-5,
        atol: float = 1.e-8,
        equal_nan: bool = False) -> bool: ...


def isclose(
        a: T,
        b: S,
        rtol: float = 1.e-5,
        atol: float = 1.e-8,
        equal_nan: bool = False) -> T: ...


def array_equal(a1: T, a2: S) -> bool: ...


def array_equiv(a1: T, a2: S) -> bool: ...


def extend_all(module: ModuleType): ...
