"""A module containing the `NestedSequence` protocol."""

import sys
from abc import abstractmethod, ABCMeta
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    overload,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np

if sys.version_info >= (3, 8):
    from typing import Protocol, runtime_checkable
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol, runtime_checkable
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True

__all__ = ["NestedSequence"]

_TT = TypeVar("_TT", bound=type)
_T_co = TypeVar("_T_co", covariant=True)

_SeqOrScalar = Union[_T_co, "NestedSequence[_T_co]"]

_NBitInt = f"_{8 * np.int_().itemsize}Bit"
_DOC = f"""A protocol for representing nested sequences.

    Runtime usage of the protocol requires either Python >= 3.8 or
    the typing-extensions_ package.

    .. _typing-extensions: https://pypi.org/project/typing-extensions/

    See Also
    --------
    :class:`collections.abc.Sequence`
        ABCs for read-only and mutable :term:`sequences<sequence>`.

    Examples
    --------
    .. code-block:: python

        >>> from __future__ import annotations
        >>> from typing import Any, List, TYPE_CHECKING
        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> def get_dtype(seq: npt.NestedSequence[int]) -> np.dtype[np.int_]:
        ...     return np.asarray(seq).dtype

        >>> a = func([1])
        >>> b = func([[1]])
        >>> c = func([[[1]]])
        >>> d = func([[[[1]]]])

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.dtype[numpy.signedinteger[numpy.typing.{_NBitInt}]]
        ...     # note:     b: numpy.dtype[numpy.signedinteger[numpy.typing.{_NBitInt}]]
        ...     # note:     c: numpy.dtype[numpy.signedinteger[numpy.typing.{_NBitInt}]]
        ...     # note:     d: numpy.dtype[numpy.signedinteger[numpy.typing.{_NBitInt}]]

"""


def _set_module_and_doc(module: str, doc: str) -> Callable[[_TT], _TT]:
    """A decorator for setting ``__module__`` and `__doc__`."""
    def decorator(func):
        func.__module__ = module
        func.__doc__ = doc
        return func
    return decorator


# E: Error message for `_ProtocolMeta` and `_ProtocolMixin`
_ERR_MSG = (
    "runtime usage of `NestedSequence` requires "
    "either typing-extensions or Python >= 3.8"
)


class _ProtocolMeta(ABCMeta):
    """Metaclass of `_ProtocolMixin`."""

    def __instancecheck__(self, params):
        raise RuntimeError(_ERR_MSG)

    def __subclasscheck__(self, params):
        raise RuntimeError(_ERR_MSG)


class _ProtocolMixin(Generic[_T_co], metaclass=_ProtocolMeta):
    """A mixin that raises upon executing methods that require `typing.Protocol`."""

    __slots__ = ()

    def __init__(self):
        raise RuntimeError(_ERR_MSG)

    def __init_subclass__(cls):
        if cls is not NestedSequence:
            raise RuntimeError(_ERR_MSG)
        super().__init_subclass__()


# Plan B in case `typing.Protocol` is unavailable.
#
# A `RuntimeError` will be raised if one attempts to execute
# methods that absolutelly require `typing.Protocol`.
if not TYPE_CHECKING and not HAVE_PROTOCOL:
    Protocol = _ProtocolMixin


@_set_module_and_doc("numpy.typing", doc=_DOC)
@runtime_checkable
class NestedSequence(Protocol[_T_co]):
    if not TYPE_CHECKING:
        __slots__ = ()

    # Can't directly inherit from `collections.abc.Sequence`
    # (as it is not a Protocol), but we can forward to its' methods
    def __contains__(self, x: object) -> bool:
        """Return ``x in self``."""
        return Sequence.__contains__(self, x)  # type: ignore[operator]

    @overload
    @abstractmethod
    def __getitem__(self, i: int) -> _SeqOrScalar[_T_co]: ...
    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> "NestedSequence[_T_co]": ...
    @abstractmethod
    def __getitem__(self, s):
        """Return ``self[s]``."""
        raise NotImplementedError("Trying to call an abstract method")

    def __iter__(self) -> Iterator[_SeqOrScalar[_T_co]]:
        """Return ``iter(self)``."""
        return Sequence.__iter__(self)  # type: ignore[arg-type]

    @abstractmethod
    def __len__(self) -> int:
        """Return ``len(self)``."""
        raise NotImplementedError("Trying to call an abstract method")

    def __reversed__(self) -> Iterator[_SeqOrScalar[_T_co]]:
        """Return ``reversed(self)``."""
        return Sequence.__reversed__(self)  # type: ignore[arg-type]

    def count(self, value: Any) -> int:
        """Return the number of occurrences of `value`."""
        return Sequence.count(self, value)  # type: ignore[arg-type]

    def index(self, value: Any, start: int = 0, stop: int = sys.maxsize) -> int:
        """Return the first index of `value`."""
        return Sequence.index(self, value, start, stop)  # type: ignore[arg-type]
