from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

type _Matrix64 = np.matrix[tuple[int, int], np.dtype[np.float64]]
type _MatrixI8 = np.matrix[tuple[int, int], np.dtype[np.int8]]
type _MatrixAny = np.matrix[tuple[int, int], np.dtype[Any]]

dtype_like: npt.DTypeLike

###

assert_type(np.matlib.empty(2), _Matrix64)
assert_type(np.matlib.empty(()), _Matrix64)
assert_type(np.matlib.empty((2,)), _Matrix64)
assert_type(np.matlib.empty((2, 3)), _Matrix64)
assert_type(np.matlib.empty((2,), np.int8), _MatrixI8)
assert_type(np.matlib.empty((2,), dtype_like), _MatrixAny)

assert_type(np.matlib.ones(2), _Matrix64)
assert_type(np.matlib.ones(()), _Matrix64)
assert_type(np.matlib.ones((2,)), _Matrix64)
assert_type(np.matlib.ones((2, 3)), _Matrix64)
assert_type(np.matlib.ones((2,), np.int8), _MatrixI8)
assert_type(np.matlib.ones((2,), dtype_like), _MatrixAny)

assert_type(np.matlib.zeros(2), _Matrix64)
assert_type(np.matlib.zeros(()), _Matrix64)
assert_type(np.matlib.zeros((2,)), _Matrix64)
assert_type(np.matlib.zeros((2, 3)), _Matrix64)
assert_type(np.matlib.zeros((2,), np.int8), _MatrixI8)
assert_type(np.matlib.zeros((2,), dtype_like), _MatrixAny)
