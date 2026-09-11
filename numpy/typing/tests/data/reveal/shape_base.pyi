from typing import Any, Self, assert_type

import numpy as np
import numpy.typing as npt

type _Array0D[ScalarT: np.generic] = np.ndarray[tuple[()], np.dtype[ScalarT]]
type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]
type _Array4D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int, int], np.dtype[ScalarT]]
type _Array5D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int, int, int], np.dtype[ScalarT]]
type _Array6D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int, int, int, int], np.dtype[ScalarT]]

i8: np.int64
f8: np.float64

AR_b: npt.NDArray[np.bool]
AR_i8: npt.NDArray[np.int64]
AR_i8_0d: _Array0D[np.int64]
AR_i8_1d: _Array1D[np.int64]
AR_i8_2d: _Array2D[np.int64]
AR_i8_3d: _Array3D[np.int64]
AR_i8_4d: _Array4D[np.int64]
AR_f8: npt.NDArray[np.float64]

AR_LIKE_f8: list[float]

# Duck-typed class implementing _SupportsSplitOps protocol for testing
class _SplitableArray:
    shape: tuple[int, ...]
    ndim: int
    def swapaxes(self, axis1: int, axis2: int, /) -> Self: ...
    def __getitem__(self, key: Any, /) -> Self: ...

splitable: _SplitableArray

assert_type(np.take_along_axis(AR_f8, AR_i8, axis=1), npt.NDArray[np.float64])
assert_type(np.take_along_axis(f8, AR_i8, axis=None), npt.NDArray[np.float64])

assert_type(np.put_along_axis(AR_f8, AR_i8, "1.0", axis=1), None)

assert_type(np.expand_dims(AR_LIKE_f8, 0), np.ndarray)
assert_type(np.expand_dims(AR_i8, ()), npt.NDArray[np.int64])
assert_type(np.expand_dims(AR_i8, 0), npt.NDArray[np.int64])
assert_type(np.expand_dims(AR_i8, (0,)), npt.NDArray[np.int64])
assert_type(np.expand_dims(AR_i8, (0, 1)), npt.NDArray[np.int64])
assert_type(np.expand_dims(AR_i8_0d, ()), _Array0D[np.int64])
assert_type(np.expand_dims(AR_i8_0d, 0), _Array1D[np.int64])
assert_type(np.expand_dims(AR_i8_0d, (0,)), _Array1D[np.int64])
assert_type(np.expand_dims(AR_i8_0d, (0, 1)), _Array2D[np.int64])
assert_type(np.expand_dims(AR_i8_1d, ()), _Array1D[np.int64])
assert_type(np.expand_dims(AR_i8_1d, 0), _Array2D[np.int64])
assert_type(np.expand_dims(AR_i8_1d, (0,)), _Array2D[np.int64])
assert_type(np.expand_dims(AR_i8_1d, (0, 1)), _Array3D[np.int64])
assert_type(np.expand_dims(AR_i8_2d, ()), _Array2D[np.int64])
assert_type(np.expand_dims(AR_i8_2d, 0), _Array3D[np.int64])
assert_type(np.expand_dims(AR_i8_2d, (0,)), _Array3D[np.int64])
assert_type(np.expand_dims(AR_i8_2d, (0, 1)), _Array4D[np.int64])
assert_type(np.expand_dims(AR_i8_3d, ()), _Array3D[np.int64])
assert_type(np.expand_dims(AR_i8_3d, 0), _Array4D[np.int64])
assert_type(np.expand_dims(AR_i8_3d, (0,)), _Array4D[np.int64])
assert_type(np.expand_dims(AR_i8_3d, (0, 1)), _Array5D[np.int64])
assert_type(np.expand_dims(AR_i8_4d, ()), _Array4D[np.int64])
assert_type(np.expand_dims(AR_i8_4d, 0), _Array5D[np.int64])
assert_type(np.expand_dims(AR_i8_4d, (0,)), _Array5D[np.int64])
assert_type(np.expand_dims(AR_i8_4d, (0, 1)), _Array6D[np.int64])

assert_type(np.column_stack([AR_i8]), npt.NDArray[np.int64])
assert_type(np.column_stack([AR_LIKE_f8]), npt.NDArray[Any])
assert_type(np.column_stack([AR_i8_0d, AR_i8_0d]), _Array2D[np.int64])
assert_type(np.column_stack([AR_i8_1d, AR_i8_1d]), _Array2D[np.int64])
assert_type(np.column_stack([AR_i8_2d, AR_i8_2d]), _Array2D[np.int64])
assert_type(np.column_stack([AR_i8_3d, AR_i8_3d]), _Array3D[np.int64])

assert_type(np.dstack([AR_i8]), npt.NDArray[np.int64])
assert_type(np.dstack([AR_LIKE_f8]), npt.NDArray[Any])

assert_type(np.array_split(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])
assert_type(np.array_split(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])
assert_type(np.array_split(splitable, 2), list[_SplitableArray])

assert_type(np.split(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])
assert_type(np.split(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])
assert_type(np.split(splitable, 2), list[_SplitableArray])

assert_type(np.hsplit(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])
assert_type(np.hsplit(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])
assert_type(np.hsplit(splitable, 2), list[_SplitableArray])

assert_type(np.vsplit(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])
assert_type(np.vsplit(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])
assert_type(np.vsplit(splitable, 2), list[_SplitableArray])

assert_type(np.dsplit(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])
assert_type(np.dsplit(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])
assert_type(np.dsplit(splitable, 2), list[_SplitableArray])

assert_type(np.kron(AR_b, AR_b), npt.NDArray[np.bool])
assert_type(np.kron(AR_b, AR_i8), npt.NDArray[np.signedinteger])
assert_type(np.kron(AR_f8, AR_f8), npt.NDArray[np.floating])

assert_type(np.tile(AR_i8, 1), npt.NDArray[np.int64])
assert_type(np.tile(AR_i8, (1,)), npt.NDArray[np.int64])
assert_type(np.tile(AR_i8, (1, 2)), npt.NDArray[np.int64])
assert_type(np.tile(AR_i8, (1, 2, 3)), npt.NDArray[np.int64])
assert_type(np.tile(AR_i8, (1, 2, 3, 4)), npt.NDArray[np.int64])
assert_type(np.tile(AR_i8_1d, ()), _Array1D[np.int64])
assert_type(np.tile(AR_i8_1d, 1), _Array1D[np.int64])
assert_type(np.tile(AR_i8_1d, (1,)), _Array1D[np.int64])
assert_type(np.tile(AR_i8_1d, (1, 2)), _Array2D[np.int64])
assert_type(np.tile(AR_i8_1d, (1, 2, 3)), _Array3D[np.int64])
assert_type(np.tile(AR_i8_1d, (1, 2, 3, 4)), _Array4D[np.int64])
assert_type(np.tile(AR_i8_2d, 1), _Array2D[np.int64])
assert_type(np.tile(AR_i8_2d, (1,)), _Array2D[np.int64])
assert_type(np.tile(AR_i8_2d, (1, 2)), _Array2D[np.int64])
assert_type(np.tile(AR_i8_2d, (1, 2, 3)), _Array3D[np.int64])
assert_type(np.tile(AR_i8_2d, (1, 2, 3, 4)), _Array4D[np.int64])
assert_type(np.tile(AR_i8_3d, 1), _Array3D[np.int64])
assert_type(np.tile(AR_i8_3d, (1,)), _Array3D[np.int64])
assert_type(np.tile(AR_i8_3d, (1, 2)), _Array3D[np.int64])
assert_type(np.tile(AR_i8_3d, (1, 2, 3)), _Array3D[np.int64])
assert_type(np.tile(AR_i8_3d, (1, 2, 3, 4)), _Array4D[np.int64])
assert_type(np.tile(AR_LIKE_f8, 1), _Array1D[Any])
assert_type(np.tile(AR_LIKE_f8, (1,)), _Array1D[Any])
assert_type(np.tile(AR_LIKE_f8, (1, 2)), _Array2D[Any])
assert_type(np.tile(AR_LIKE_f8, (1, 2, 3)), _Array3D[Any])
assert_type(np.tile(AR_LIKE_f8, (1, 2, 3, 4)), _Array4D[Any])
assert_type(np.tile(AR_LIKE_f8, [2, 2]), npt.NDArray[Any])

assert_type(np.unstack(AR_i8, axis=0), tuple[npt.NDArray[np.int64], ...])
assert_type(np.unstack(AR_LIKE_f8, axis=0), tuple[npt.NDArray[Any], ...])
