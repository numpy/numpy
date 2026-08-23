from _typeshed import Incomplete
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

__all__ = ["interface"]

type _CDataVoidPointer = Incomplete  # currently not expressible

class interface(NamedTuple):
    state_address: int
    state: _CDataVoidPointer
    next_uint64: Callable[..., np.uint64]
    next_uint32: Callable[..., np.uint32]
    next_double: Callable[..., np.float64]
    bit_generator: _CDataVoidPointer
