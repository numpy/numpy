from typing import Any, Self, assert_type

import numpy as np
import numpy.typing as npt

type _Array5D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int, int, int], np.dtype[ScalarT]]
type _Array6D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int, int, int, int], np.dtype[ScalarT]]

i8: np.int64
f8: np.float64

AR_b: npt.NDArray[np.bool]
AR_i8: npt.NDArray[np.int64]
AR_i8_0d: npt.Array0D[np.int64]
AR_i8_1d: npt.Array1D[np.int64]
AR_i8_2d: npt.Array2D[np.int64]
AR_i8_3d: npt.Array3D[np.int64]
AR_i8_4d: npt.Array4D[np.int64]
AR_i8_5d: _Array5D[np.int64]
AR_i8_nd: np.ndarray[tuple[int, *tuple[int, ...]], np.dtype[np.int64]]
AR_f8: npt.NDArray[np.float64]
AR_O: npt.NDArray[np.object_]
AR_O_int_1d: npt.Array1D[np.object_[int]]
AR_T_1d: np.ndarray[tuple[int], np.dtypes.StringDType]

AR_LIKE_b: list[bool]
AR_LIKE_f8: list[float]
AR_LIKE_c16: list[complex]

# Duck-typed class implementing _SupportsSplitOps protocol for testing
class _SplitableArray:
    shape: tuple[int, ...]
    ndim: int
    def swapaxes(self, axis1: int, axis2: int, /) -> Self: ...
    def __getitem__(self, key: Any, /) -> Self: ...

splitable: _SplitableArray

def _func_f64(hamster: npt.Array1D[np.int64]) -> np.float64: ...
def _func_f64_1d(clay: npt.Array1D[np.int64]) -> npt.Array1D[np.float64]: ...
def _func_f64_2d(fork: npt.Array1D[np.int64]) -> npt.Array2D[np.float64]: ...
def _func_f64_nd(drum: npt.Array1D[np.int64]) -> npt.NDArray[np.float64]: ...
def _func_b(spleen: npt.Array1D[np.int64]) -> bool: ...
def _func_i(pulsar: npt.Array1D[np.int64]) -> int: ...
def _func_f(yoneda: npt.Array1D[np.int64]) -> float: ...
def _func_c(potato: npt.Array1D[np.int64]) -> complex: ...
def _func_axis_f64(axolotl: npt.Array1D[np.int64], axis: int) -> np.float64: ...
def _func_axis_f64_2d(foam: npt.Array3D[np.int64], axis: int) -> npt.Array2D[np.float64]: ...
def _func_axis_f64_nd(newt: npt.NDArray[np.int64], axis: int) -> npt.NDArray[np.float64]: ...

###

assert_type(np.take_along_axis(AR_f8, AR_i8_2d, axis=1), npt.Array2D[np.float64])
assert_type(np.take_along_axis(f8, AR_i8, axis=None), npt.Array1D[np.float64])

assert_type(np.put_along_axis(AR_f8, AR_i8, "1.0", axis=1), None)

assert_type(np.expand_dims(AR_LIKE_f8, 0), np.ndarray)
assert_type(np.expand_dims(AR_LIKE_f8, [0, 1]), np.ndarray)
assert_type(np.expand_dims(AR_i8, ()), npt.NDArray[np.int64])
assert_type(np.expand_dims(AR_i8, 0), npt.NDArray[np.int64])
assert_type(np.expand_dims(AR_i8, (0,)), npt.NDArray[np.int64])
assert_type(np.expand_dims(AR_i8, (0, 1)), npt.NDArray[np.int64])
assert_type(np.expand_dims(AR_i8, [0, 1]), npt.NDArray[np.int64])
assert_type(np.expand_dims(AR_i8_0d, ()), npt.Array0D[np.int64])
assert_type(np.expand_dims(AR_i8_0d, 0), npt.Array1D[np.int64])
assert_type(np.expand_dims(AR_i8_0d, (0,)), npt.Array1D[np.int64])
assert_type(np.expand_dims(AR_i8_0d, (0, 1)), npt.Array2D[np.int64])
assert_type(np.expand_dims(AR_i8_1d, ()), npt.Array1D[np.int64])
assert_type(np.expand_dims(AR_i8_1d, 0), npt.Array2D[np.int64])
assert_type(np.expand_dims(AR_i8_1d, (0,)), npt.Array2D[np.int64])
assert_type(np.expand_dims(AR_i8_1d, (0, 1)), npt.Array3D[np.int64])
assert_type(np.expand_dims(AR_i8_2d, ()), npt.Array2D[np.int64])
assert_type(np.expand_dims(AR_i8_2d, 0), npt.Array3D[np.int64])
assert_type(np.expand_dims(AR_i8_2d, (0,)), npt.Array3D[np.int64])
assert_type(np.expand_dims(AR_i8_2d, (0, 1)), npt.Array4D[np.int64])
assert_type(np.expand_dims(AR_i8_3d, ()), npt.Array3D[np.int64])
assert_type(np.expand_dims(AR_i8_3d, 0), npt.Array4D[np.int64])
assert_type(np.expand_dims(AR_i8_3d, (0,)), npt.Array4D[np.int64])
assert_type(np.expand_dims(AR_i8_3d, (0, 1)), _Array5D[np.int64])
assert_type(np.expand_dims(AR_i8_4d, ()), npt.Array4D[np.int64])
assert_type(np.expand_dims(AR_i8_4d, 0), _Array5D[np.int64])
assert_type(np.expand_dims(AR_i8_4d, (0,)), _Array5D[np.int64])
assert_type(np.expand_dims(AR_i8_4d, (0, 1)), _Array6D[np.int64])
assert_type(np.expand_dims(AR_i8_nd, [0, 1]), npt.NDArray[np.int64])

assert_type(np.column_stack([AR_i8]), npt.NDArray[np.int64])
assert_type(np.column_stack([AR_LIKE_f8]), npt.NDArray[Any])
assert_type(np.column_stack([AR_i8_0d, AR_i8_0d]), npt.Array2D[np.int64])
assert_type(np.column_stack([AR_i8_1d, AR_i8_1d]), npt.Array2D[np.int64])
assert_type(np.column_stack([AR_i8_2d, AR_i8_2d]), npt.Array2D[np.int64])
assert_type(np.column_stack([AR_i8_3d, AR_i8_3d]), npt.Array3D[np.int64])

assert_type(np.dstack([AR_i8]), npt.NDArray[np.int64])
assert_type(np.dstack([AR_i8_2d]), npt.Array3D[np.int64])
assert_type(np.dstack([AR_i8_4d]), npt.Array4D[np.int64])
assert_type(np.dstack([AR_LIKE_f8]), npt.NDArray[Any])

assert_type(np.array_split(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])
assert_type(np.array_split(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])
assert_type(np.array_split(AR_LIKE_f8, AR_i8), list[npt.NDArray[Any]])
assert_type(np.array_split(splitable, 2), list[_SplitableArray])
assert_type(np.array_split(splitable, AR_i8_1d), list[_SplitableArray])

assert_type(np.split(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])
assert_type(np.split(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])
assert_type(np.split(AR_LIKE_f8, AR_i8), list[npt.NDArray[Any]])
assert_type(np.split(splitable, 2), list[_SplitableArray])
assert_type(np.split(splitable, AR_i8_1d), list[_SplitableArray])

assert_type(np.hsplit(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])
assert_type(np.hsplit(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])
assert_type(np.hsplit(AR_LIKE_f8, AR_i8), list[npt.NDArray[Any]])
assert_type(np.hsplit(splitable, 2), list[_SplitableArray])
assert_type(np.hsplit(splitable, AR_i8_1d), list[_SplitableArray])

assert_type(np.vsplit(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])
assert_type(np.vsplit(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])
assert_type(np.vsplit(splitable, 2), list[_SplitableArray])

assert_type(np.dsplit(AR_i8, [3, 5, 6, 10]), list[npt.NDArray[np.int64]])
assert_type(np.dsplit(AR_LIKE_f8, [3, 5, 6, 10]), list[npt.NDArray[Any]])
assert_type(np.dsplit(splitable, 2), list[_SplitableArray])

assert_type(np.kron(AR_b, AR_b), npt.NDArray[np.bool])
assert_type(np.kron(AR_b, AR_i8), npt.NDArray[np.int_ | Any])
assert_type(np.kron(AR_i8, AR_f8), npt.NDArray[np.float64 | Any])
assert_type(np.kron(AR_i8, AR_O), npt.NDArray[np.object_])
assert_type(np.kron(AR_i8_1d, AR_i8), npt.NDArray[np.int64])
assert_type(np.kron(AR_i8_1d, AR_i8_1d), npt.Array1D[np.int64])
assert_type(np.kron(AR_i8_1d, AR_i8_2d), npt.Array2D[np.int64])
assert_type(np.kron(AR_i8_2d, AR_i8_1d), npt.Array2D[np.int64])
assert_type(np.kron(AR_i8_2d, AR_i8_3d), npt.Array3D[np.int64])
assert_type(np.kron(AR_i8_3d, AR_i8_2d), npt.Array3D[np.int64])
assert_type(np.kron(AR_i8_4d, AR_i8_4d), npt.NDArray[np.int64])
assert_type(np.kron(AR_f8, AR_LIKE_c16), npt.NDArray[np.complex128 | Any])
assert_type(np.kron(AR_O, AR_i8), npt.NDArray[np.object_])
assert_type(np.kron(AR_LIKE_b, AR_LIKE_b), npt.NDArray[np.bool])

assert_type(np.tile(AR_i8, 1), npt.NDArray[np.int64])
assert_type(np.tile(AR_i8, (1,)), npt.NDArray[np.int64])
assert_type(np.tile(AR_i8, (1, 2)), npt.NDArray[np.int64])
assert_type(np.tile(AR_i8, (1, 2, 3)), npt.NDArray[np.int64])
assert_type(np.tile(AR_i8, (1, 2, 3, 4)), npt.NDArray[np.int64])
assert_type(np.tile(AR_i8_1d, ()), npt.Array1D[np.int64])
assert_type(np.tile(AR_i8_1d, 1), npt.Array1D[np.int64])
assert_type(np.tile(AR_i8_1d, (1,)), npt.Array1D[np.int64])
assert_type(np.tile(AR_i8_1d, (1, 2)), npt.Array2D[np.int64])
assert_type(np.tile(AR_i8_1d, (1, 2, 3)), npt.Array3D[np.int64])
assert_type(np.tile(AR_i8_1d, (1, 2, 3, 4)), npt.Array4D[np.int64])
assert_type(np.tile(AR_i8_2d, 1), npt.Array2D[np.int64])
assert_type(np.tile(AR_i8_2d, (1,)), npt.Array2D[np.int64])
assert_type(np.tile(AR_i8_2d, (1, 2)), npt.Array2D[np.int64])
assert_type(np.tile(AR_i8_2d, (1, 2, 3)), npt.Array3D[np.int64])
assert_type(np.tile(AR_i8_2d, (1, 2, 3, 4)), npt.Array4D[np.int64])
assert_type(np.tile(AR_i8_3d, 1), npt.Array3D[np.int64])
assert_type(np.tile(AR_i8_3d, (1,)), npt.Array3D[np.int64])
assert_type(np.tile(AR_i8_3d, (1, 2)), npt.Array3D[np.int64])
assert_type(np.tile(AR_i8_3d, (1, 2, 3)), npt.Array3D[np.int64])
assert_type(np.tile(AR_i8_3d, (1, 2, 3, 4)), npt.Array4D[np.int64])
assert_type(np.tile(AR_LIKE_f8, 1), npt.Array1D[Any])
assert_type(np.tile(AR_LIKE_f8, (1,)), npt.Array1D[Any])
assert_type(np.tile(AR_LIKE_f8, (1, 2)), npt.Array2D[Any])
assert_type(np.tile(AR_LIKE_f8, (1, 2, 3)), npt.Array3D[Any])
assert_type(np.tile(AR_LIKE_f8, (1, 2, 3, 4)), npt.Array4D[Any])
assert_type(np.tile(AR_LIKE_f8, [2, 2]), npt.NDArray[Any])

assert_type(np.unstack(AR_i8), tuple[npt.NDArray[np.int64], ...])
assert_type(np.unstack(AR_i8_1d), tuple[np.int64, ...])
assert_type(np.unstack(AR_i8_2d, axis=1), tuple[npt.Array1D[np.int64], ...])
assert_type(np.unstack(AR_i8_3d, axis=-1), tuple[npt.Array2D[np.int64], ...])
assert_type(np.unstack(AR_i8_4d), tuple[npt.Array3D[np.int64], ...])
assert_type(np.unstack(AR_i8_5d), tuple[npt.NDArray[np.int64], ...])
assert_type(np.unstack(AR_i8_nd), tuple[Any, ...])
assert_type(np.unstack(AR_O_int_1d), tuple[int, ...])
assert_type(np.unstack(AR_T_1d), tuple[str, ...])

assert_type(np.apply_along_axis(_func_f64_nd, 0, AR_i8_2d), npt.NDArray[np.float64])
assert_type(np.apply_along_axis(_func_f64, 0, AR_i8), npt.NDArray[np.float64])
assert_type(np.apply_along_axis(_func_b, 0, AR_i8), npt.NDArray[np.bool])
assert_type(np.apply_along_axis(_func_i, 0, AR_i8), npt.NDArray[np.int_])
assert_type(np.apply_along_axis(_func_f, 0, AR_i8), npt.NDArray[np.float64])
assert_type(np.apply_along_axis(_func_c, 0, AR_i8), npt.NDArray[np.complex128])
assert_type(np.apply_along_axis(_func_f64, 0, AR_i8_1d), npt.Array0D[np.float64])
assert_type(np.apply_along_axis(_func_f64_1d, 0, AR_i8_1d), npt.Array1D[np.float64])
assert_type(np.apply_along_axis(_func_f64, 0, AR_i8_2d), npt.Array1D[np.float64])
assert_type(np.apply_along_axis(_func_f64_1d, 0, AR_i8_2d), npt.Array2D[np.float64])
assert_type(np.apply_along_axis(_func_f64_2d, 0, AR_i8_2d), npt.Array3D[np.float64])
assert_type(np.apply_along_axis(_func_b, 0, AR_i8_2d), npt.Array1D[np.bool])
assert_type(np.apply_along_axis(_func_i, 0, AR_i8_2d), npt.Array1D[np.int_])
assert_type(np.apply_along_axis(_func_f, 0, AR_i8_2d), npt.Array1D[np.float64])
assert_type(np.apply_along_axis(_func_c, 0, AR_i8_2d), npt.Array1D[np.complex128])
assert_type(np.apply_along_axis(_func_f64, 0, AR_i8_3d), npt.Array2D[np.float64])
assert_type(np.apply_along_axis(_func_f64_1d, 0, AR_i8_3d), npt.Array3D[np.float64])
assert_type(np.apply_along_axis(_func_b, 0, AR_i8_3d), npt.Array2D[np.bool])
assert_type(np.apply_along_axis(_func_i, 0, AR_i8_3d), npt.Array2D[np.int_])
assert_type(np.apply_along_axis(_func_f, 0, AR_i8_3d), npt.Array2D[np.float64])
assert_type(np.apply_along_axis(_func_c, 0, AR_i8_3d), npt.Array2D[np.complex128])
assert_type(np.apply_along_axis(_func_f64_nd, 0, AR_LIKE_f8), npt.NDArray[np.float64])
assert_type(np.apply_along_axis(str, 0, AR_i8_2d), npt.NDArray[Any])

assert_type(np.apply_over_axes(_func_axis_f64_nd, AR_i8, 0), npt.NDArray[np.float64])
assert_type(np.apply_over_axes(_func_axis_f64, AR_i8_1d, 0), npt.Array1D[np.float64])
assert_type(np.apply_over_axes(np.sum, AR_i8_2d, 0), npt.Array2D[np.int_])
assert_type(np.apply_over_axes(np.mean, AR_i8_3d, [0, 2]), npt.Array3D[np.float64])
assert_type(np.apply_over_axes(_func_axis_f64_nd, AR_i8_2d, [0]), npt.Array2D[np.float64])
assert_type(np.apply_over_axes(_func_axis_f64_2d, AR_i8_3d, 0), npt.Array3D[np.float64])
assert_type(np.apply_over_axes(lambda x, ax: np.sum(x, axis=ax, keepdims=True), AR_i8_4d, (0, 1)), npt.Array4D[np.int_])
