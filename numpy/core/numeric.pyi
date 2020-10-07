from typing import Any, Optional, Union, Sequence, Tuple

from numpy import ndarray, dtype, bool_, _OrderKACF, _OrderCF
from numpy.typing import ArrayLike, DtypeLike, _ShapeLike

def zeros_like(
    a: ArrayLike,
    dtype: DtypeLike = ...,
    order: _OrderKACF = ...,
    subok: bool = ...,
    shape: Optional[Union[int, Sequence[int]]] = ...,
) -> ndarray: ...
def ones(
    shape: _ShapeLike,
    dtype: DtypeLike = ...,
    order: _OrderCF = ...,
    *,
    like: ArrayLike = ...,
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
