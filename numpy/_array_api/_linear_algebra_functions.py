from __future__ import annotations

from ._array_object import ndarray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._types import Optional, Sequence, Tuple, Union, array

import numpy as np

# These functions are also exposed to the top-level
from .linalg import einsum, matmul, tensordot, transpose, vecdot
