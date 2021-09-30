"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

__all__ = [
    "Array",
    "Device",
    "Dtype",
    "SupportsDLPack",
    "SupportsBufferProtocol",
    "PyCapsule",
]

import sys
from typing import Any, Literal, Sequence, Type, Union, TYPE_CHECKING, TypeVar

from ._array_object import Array
from numpy import (
    dtype,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float32,
    float64,
)

# This should really be recursive, but that isn't supported yet. See the
# similar comment in numpy/typing/_array_like.py
_T = TypeVar("_T")
NestedSequence = Sequence[Sequence[_T]]

Device = Literal["cpu"]
if TYPE_CHECKING or sys.version_info >= (3, 9):
    Dtype = dtype[Union[
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float32,
        float64,
    ]]
else:
    Dtype = dtype

SupportsDLPack = Any
SupportsBufferProtocol = Any
PyCapsule = Any
