from __future__ import annotations

import sys
from typing import Any, overload, Sequence, TYPE_CHECKING, Union, TypeVar

from numpy import (
    ndarray,
    dtype,
    generic,
    bool_,
    unsignedinteger,
    integer,
    floating,
    complexfloating,
    timedelta64,
    datetime64,
    object_,
    void,
    str_,
    bytes_,
)

from ._nested_sequence import NestedSequence
from ._dtype_like import DTypeLike

if sys.version_info >= (3, 8):
    from typing import Protocol
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

_T = TypeVar("_T")
_DType = TypeVar("_DType", bound="dtype[Any]")

if TYPE_CHECKING or HAVE_PROTOCOL:
    # The `_SupportsArray` protocol only cares about the default dtype
    # (i.e. `dtype=None`) of the to-be returned array.
    # Concrete implementations of the protocol are responsible for adding
    # any and all remaining overloads
    class _SupportsArray(Protocol[_DType]):
        def __array__(self, dtype: None = ...) -> ndarray[Any, _DType]: ...
else:
    _SupportsArray = Any

# A union representing array-like objects; consists of two typevars:
# One representing types that can be parametrized w.r.t. `np.dtype`
# and another one for the rest
_ArrayLike = Union[
    _SupportsArray[_DType],
    NestedSequence[_SupportsArray[_DType]],
    _T,
    NestedSequence[_T],
]

# TODO: support buffer protocols once
#
# https://bugs.python.org/issue27501
#
# is resolved. See also the mypy issue:
#
# https://github.com/python/typing/issues/593
ArrayLike = _ArrayLike[
    "dtype[Any]",
    Union[bool, int, float, complex, str, bytes],
]

# `ArrayLike<X>`: array-like objects that can be coerced into `X`
# given the casting rules `same_kind`
_ArrayLikeBool = _ArrayLike[
    "dtype[bool_]",
    bool,
]
_ArrayLikeUInt = _ArrayLike[
    "dtype[Union[bool_, unsignedinteger[Any]]]",
    bool,
]
_ArrayLikeInt = _ArrayLike[
    "dtype[Union[bool_, integer[Any]]]",
    Union[bool, int],
]
_ArrayLikeFloat = _ArrayLike[
    "dtype[Union[bool_, integer[Any], floating[Any]]]",
    Union[bool, int, float],
]
_ArrayLikeComplex = _ArrayLike[
    "dtype[Union[bool_, integer[Any], floating[Any], complexfloating[Any, Any]]]",
    Union[bool, int, float, complex],
]
_ArrayLikeTD64 = _ArrayLike[
    "dtype[Union[bool_, integer[Any], timedelta64]]",
    Union[bool, int],
]
_ArrayLikeDT64 = Union[
    _SupportsArray["dtype[datetime64]"],
    NestedSequence[_SupportsArray["dtype[datetime64]"]],
]
_ArrayLikeObject = Union[
    _SupportsArray["dtype[object_]"],
    NestedSequence[_SupportsArray["dtype[object_]"]],
]

_ArrayLikeVoid = Union[
    _SupportsArray["dtype[void]"],
    NestedSequence[_SupportsArray["dtype[void]"]],
]
_ArrayLikeStr = _ArrayLike[
    "dtype[str_]",
    str,
]
_ArrayLikeBytes = _ArrayLike[
    "dtype[bytes_]",
    bytes,
]
