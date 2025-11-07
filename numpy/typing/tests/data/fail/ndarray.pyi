# pyright: reportUnusedExpression=none

from typing import Any

import numpy as np
import numpy.typing as npt

# Ban setting dtype since mutating the type of the array in place
# makes having ndarray be generic over dtype impossible. Generally
# users should use `ndarray.view` in this situation anyway. See
#
# https://github.com/numpy/numpy-stubs/issues/7
#
# for more context.
float_array = np.array([1.0])
float_array.dtype = np.bool  # type: ignore[assignment, misc]

# https://github.com/numpy/numpy/issues/30173
dt_array: npt.NDArray[np.datetime64]
int_array: npt.NDArray[np.int_]
any_array: npt.NDArray[Any]

dt_array / int_array  # type: ignore[operator]
dt_array / any_array  # type: ignore[operator]
