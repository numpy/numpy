import sys
from typing import Any, overload, Sequence, TYPE_CHECKING, Union

from numpy import ndarray
from ._dtype_like import DtypeLike

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

if TYPE_CHECKING or HAVE_PROTOCOL:
    class _SupportsArray(Protocol):
        @overload
        def __array__(self, __dtype: DtypeLike = ...) -> ndarray: ...
        @overload
        def __array__(self, dtype: DtypeLike = ...) -> ndarray: ...
else:
    _SupportsArray = Any

ArrayLike = Union[bool, int, float, complex, _SupportsArray, Sequence]
