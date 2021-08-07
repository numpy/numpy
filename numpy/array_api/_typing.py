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

from typing import Any, Sequence, Type, Union

from . import (
    Array,
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
NestedSequence = Sequence[Sequence[Any]]

Device = Any
Dtype = Type[
    Union[[int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32, float64]]
]
SupportsDLPack = Any
SupportsBufferProtocol = Any
PyCapsule = Any
