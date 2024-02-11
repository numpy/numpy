"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

from __future__ import annotations

__all__ = [
    "Array",
    "Device",
    "Dtype",
    "SupportsDLPack",
    "SupportsBufferProtocol",
    "PyCapsule",
]

import sys

from typing import (
    Any,
    Literal,
    Sequence,
    Type,
    Union,
    TypeVar,
    Protocol,
)

from ._array_object import Array, _cpu_device
from ._dtypes import _DType

_T_co = TypeVar("_T_co", covariant=True)

class NestedSequence(Protocol[_T_co]):
    def __getitem__(self, key: int, /) -> _T_co | NestedSequence[_T_co]: ...
    def __len__(self, /) -> int: ...


Device = _cpu_device

Dtype = _DType

if sys.version_info >= (3, 12):
    from collections.abc import Buffer as SupportsBufferProtocol
else:
    SupportsBufferProtocol = Any

PyCapsule = Any

class SupportsDLPack(Protocol):
    def __dlpack__(self, /, *, stream: None = ...) -> PyCapsule: ...
