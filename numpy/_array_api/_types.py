"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

__all__ = ['Any', 'List', 'Literal', 'Optional', 'Tuple', 'Union', 'array',
           'device', 'dtype', 'SupportsDLPack', 'SupportsBufferProtocol',
           'PyCapsule']

from typing import Any, List, Literal, NamedTuple, Optional, Tuple, Union, TypeVar

from . import (ndarray, int8, int16, int32, int64, uint8, uint16, uint32,
               uint64, float32, float64)

array = ndarray
device = TypeVar('device')
dtype = Literal[int8, int16, int32, int64, uint8, uint16,
                uint32, uint64, float32, float64]
SupportsDLPack = TypeVar('SupportsDLPack')
SupportsBufferProtocol = TypeVar('SupportsBufferProtocol')
PyCapsule = TypeVar('PyCapsule')

class eighresult(NamedTuple):
    u: array
    v: array

class lstsqresult(NamedTuple):
    x: array
    residuals: array
    rank: array
    s: array

class qrresult(NamedTuple):
    q: array
    r: array

class slogdetresult(NamedTuple):
    sign: array
    logabsdet: array

class svdresult(NamedTuple):
    u: array
    s: array
    v: array
