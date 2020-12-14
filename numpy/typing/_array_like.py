from __future__ import annotations

import sys
from typing import Any, overload, Sequence, TYPE_CHECKING, Union, TypeVar

from numpy import ndarray, dtype
from ._scalars import _ScalarLike
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

# TODO: support buffer protocols once
#
# https://bugs.python.org/issue27501
#
# is resolved. See also the mypy issue:
#
# https://github.com/python/typing/issues/593
ArrayLike = Union[
    _ScalarLike,
    Sequence[_ScalarLike],
    Sequence[Sequence[Any]],  # TODO: Wait for support for recursive types
    "_SupportsArray[Any]",
]
