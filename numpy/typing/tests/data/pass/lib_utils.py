from __future__ import annotations

from io import StringIO

import numpy as np

FILE = StringIO()
AR = np.arange(10, dtype=np.float64)

def func(a: int) -> bool: ...

np.byte_bounds(AR)
np.byte_bounds(np.float64())

np.info(1, output=FILE)

