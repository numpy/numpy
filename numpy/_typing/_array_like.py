from collections.abc import Buffer, Callable, Collection, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.dtypes import StringDType
else:
    from numpy._core.multiarray import StringDType

from ._nbit_base import _32Bit, _64Bit
from ._nested_sequence import _NestedSequence
from ._shape import _AnyShape

type NDArray[ScalarT: np.generic] = np.ndarray[_AnyShape, np.dtype[ScalarT]]

# The `_SupportsArray` protocol only cares about the default dtype
# (i.e. `dtype=None` or no `dtype` parameter at all) of the to-be returned
# array.
# Concrete implementations of the protocol are responsible for adding
# any and all remaining overloads
@runtime_checkable
class _SupportsArray[DTypeT: np.dtype](Protocol):
    def __array__(self) -> np.ndarray[Any, DTypeT]: ...


@runtime_checkable
class _SupportsArrayFunc(Protocol):
    """A protocol class representing `~class.__array_function__`."""
    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Collection[type[Any]],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> object: ...


# TODO: Wait until mypy supports recursive objects in combination with typevars
type _FiniteNestedSequence[T] = (
    T
    | Sequence[T]
    | Sequence[Sequence[T]]
    | Sequence[Sequence[Sequence[T]]]
    | Sequence[Sequence[Sequence[Sequence[T]]]]
)

# A subset of `npt.ArrayLike` that can be parametrized w.r.t. `np.generic`
type _ArrayLike[ScalarT: np.generic] = (
    _SupportsArray[np.dtype[ScalarT]]
    | _NestedSequence[_SupportsArray[np.dtype[ScalarT]]]
)

# A union representing array-like objects; consists of two typevars:
# One representing types that can be parametrized w.r.t. `np.dtype`
# and another one for the rest
type _DualArrayLike[DTypeT: np.dtype, BuiltinT] = (
    _SupportsArray[DTypeT]
    | _NestedSequence[_SupportsArray[DTypeT]]
    | BuiltinT
    | _NestedSequence[BuiltinT]
)

type ArrayLike = Buffer | _DualArrayLike[np.dtype, complex | bytes | str]

# `ArrayLike<X>_co`: array-like objects that can be coerced into `X`
# given the casting rules `same_kind`
type _ArrayLikeBool_co = _DualArrayLike[np.dtype[np.bool], bool]
type _ArrayLikeUInt_co = _DualArrayLike[np.dtype[np.bool | np.unsignedinteger], bool]
type _ArrayLikeInt_co = _DualArrayLike[np.dtype[np.bool | np.integer], int]
type _ArrayLikeFloat_co = _DualArrayLike[np.dtype[np.bool | np.integer | np.floating], float]
type _ArrayLikeComplex_co = _DualArrayLike[np.dtype[np.bool | np.number], complex]
type _ArrayLikeNumber_co = _ArrayLikeComplex_co
type _ArrayLikeTD64_co = _DualArrayLike[np.dtype[np.bool | np.integer | np.timedelta64], int]
type _ArrayLikeDT64_co = _ArrayLike[np.datetime64]
type _ArrayLikeObject_co = _ArrayLike[np.object_]

type _ArrayLikeVoid_co = _ArrayLike[np.void]
type _ArrayLikeBytes_co = _DualArrayLike[np.dtype[np.bytes_], bytes]
type _ArrayLikeStr_co = _DualArrayLike[np.dtype[np.str_], str]
type _ArrayLikeString_co = _DualArrayLike[StringDType, str]
type _ArrayLikeAnyString_co = _DualArrayLike[np.dtype[np.character] | StringDType, bytes | str]

type __Float64_co = np.floating[_64Bit] | np.float32 | np.float16 | np.integer | np.bool
type __Complex128_co = np.number[_64Bit] | np.number[_32Bit] | np.float16 | np.integer | np.bool
type _ArrayLikeFloat64_co = _DualArrayLike[np.dtype[__Float64_co], float]
type _ArrayLikeComplex128_co = _DualArrayLike[np.dtype[__Complex128_co], complex]

# NOTE: This includes `builtins.bool`, but not `numpy.bool`.
type _ArrayLikeInt = _DualArrayLike[np.dtype[np.integer], int]
