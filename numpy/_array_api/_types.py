"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

__all__ = ['Any', 'List', 'Literal', 'Optional', 'Tuple', 'Union', 'Array',
           'Device', 'Dtype', 'SupportsDLPack', 'SupportsBufferProtocol',
           'PyCapsule']

from typing import Any, List, Literal, Optional, Tuple, Union, TypeVar

from . import (ndarray, int8, int16, int32, int64, uint8, uint16, uint32,
               uint64, float32, float64)

Array = ndarray
Device = TypeVar('device')
Dtype = Literal[int8, int16, int32, int64, uint8, uint16,
                uint32, uint64, float32, float64]
SupportsDLPack = TypeVar('SupportsDLPack')
SupportsBufferProtocol = TypeVar('SupportsBufferProtocol')
PyCapsule = TypeVar('PyCapsule')
