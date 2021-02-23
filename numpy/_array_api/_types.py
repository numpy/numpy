"""
This file defines the types for type annotations.

These names aren't part of the module namespace, but they are used in the
annotations in the function signatures. The functions in the module are only
valid for inputs that match the given type annotations.
"""

__all__ = ['Literal', 'Optional', 'Tuple', 'Union', 'array', 'device',
           'dtype', 'SupportsDLPack', 'SupportsBufferProtocol', 'PyCapsule']

from typing import Literal, Optional, Tuple, Union, TypeVar

import numpy as np

array = np.ndarray
device = TypeVar('device')
dtype = Literal[np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                np.uint32, np.uint64, np.float32, np.float64]
SupportsDLPack = TypeVar('SupportsDLPack')
SupportsBufferProtocol = TypeVar('SupportsBufferProtocol')
PyCapsule = TypeVar('PyCapsule')
