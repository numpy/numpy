from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]
type _Array4D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int, int], np.dtype[ScalarT]]

AR_f8: npt.NDArray[np.float64]
AR_LIKE_f: list[float]
AR_f8_2d: np.ndarray[tuple[int, int], np.dtype[np.float64]]
AR_i8_1d: np.ndarray[tuple[int], np.dtype[np.int64]]
AR_i8_2d: np.ndarray[tuple[int, int], np.dtype[np.int64]]
AR_i8_3d: np.ndarray[tuple[int, int, int], np.dtype[np.int64]]
AR_i8_4d: np.ndarray[tuple[int, int, int, int], np.dtype[np.int64]]
f8: np.float64

shape_1d: tuple[int]
shape_2d: tuple[int, int]
shape_3d: tuple[int, int, int]
interface_dict: dict[str, Any]

assert_type(np.lib.stride_tricks.as_strided(AR_f8_2d), _Array2D[np.float64])
assert_type(np.lib.stride_tricks.as_strided(AR_f8, strides=(1, 5)), npt.NDArray[np.float64])
assert_type(np.lib.stride_tricks.as_strided(AR_f8, shape_2d, (8, 8)), _Array2D[np.float64])
assert_type(np.lib.stride_tricks.as_strided(AR_f8, shape=shape_3d), _Array3D[np.float64])
assert_type(np.lib.stride_tricks.as_strided(AR_f8, shape=[9, 20]), npt.NDArray[np.float64])
assert_type(np.lib.stride_tricks.as_strided(AR_LIKE_f, shape=shape_1d), _Array1D[Any])
assert_type(np.lib.stride_tricks.as_strided(AR_LIKE_f), npt.NDArray[Any])

assert_type(np.lib.stride_tricks.sliding_window_view(AR_i8_1d, 5), _Array2D[np.int64])
assert_type(np.lib.stride_tricks.sliding_window_view(AR_i8_2d, 5, axis=0), _Array3D[np.int64])
assert_type(np.lib.stride_tricks.sliding_window_view(AR_i8_2d, (2, 5)), _Array4D[np.int64])
assert_type(np.lib.stride_tricks.sliding_window_view(AR_i8_3d, (5,), axis=-1), _Array4D[np.int64])
assert_type(np.lib.stride_tricks.sliding_window_view(AR_f8, 5), npt.NDArray[np.float64])
assert_type(np.lib.stride_tricks.sliding_window_view(AR_f8, [9], axis=1), npt.NDArray[np.float64])
assert_type(np.lib.stride_tricks.sliding_window_view(AR_LIKE_f, (1, 5)), npt.NDArray[Any])

assert_type(np.broadcast_to(AR_f8, 1), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.broadcast_to(AR_f8, ()), np.ndarray[tuple[()], np.dtype[np.float64]])
assert_type(np.broadcast_to(AR_f8, (1,)), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.broadcast_to(AR_f8, (1, 2)), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.broadcast_to(AR_f8, (1, 2, 3)), np.ndarray[tuple[int, int, int], np.dtype[np.float64]])
assert_type(np.broadcast_to(AR_f8, [1, 2]), npt.NDArray[np.float64])
assert_type(np.broadcast_to(AR_LIKE_f, 1), np.ndarray[tuple[int], np.dtype[Any]])
assert_type(np.broadcast_to(AR_LIKE_f, ()), np.ndarray[tuple[()], np.dtype[Any]])
assert_type(np.broadcast_to(AR_LIKE_f, (1,)), np.ndarray[tuple[int], np.dtype[Any]])
assert_type(np.broadcast_to(AR_LIKE_f, (1, 2)), np.ndarray[tuple[int, int], np.dtype[Any]])
assert_type(np.broadcast_to(AR_LIKE_f, (1, 2, 3)), np.ndarray[tuple[int, int, int], np.dtype[Any]])
assert_type(np.broadcast_to(AR_LIKE_f, [1, 2]), npt.NDArray[Any])

assert_type(np.broadcast_shapes(), tuple[()])
assert_type(np.broadcast_shapes(1), tuple[int])
assert_type(np.broadcast_shapes(shape_2d), tuple[int, int])
assert_type(np.broadcast_shapes([3, 1]), tuple[Any, ...])
assert_type(np.broadcast_shapes(AR_f8.shape, shape_2d), tuple[Any, ...])
assert_type(np.broadcast_shapes(shape_2d, AR_f8.shape), tuple[Any, ...])
assert_type(np.broadcast_shapes((1, *AR_f8.shape), shape_1d), tuple[Any, ...])
assert_type(np.broadcast_shapes(shape_1d, (1, *AR_f8.shape)), tuple[Any, ...])
assert_type(np.broadcast_shapes(1, shape_1d), tuple[int])
assert_type(np.broadcast_shapes(shape_1d, shape_2d), tuple[int, int])
assert_type(np.broadcast_shapes(shape_2d, 2), tuple[int, int])
assert_type(np.broadcast_shapes(shape_2d, shape_3d), tuple[int, int, int])
assert_type(np.broadcast_shapes(shape_3d, shape_2d), tuple[int, int, int])
assert_type(np.broadcast_shapes((1, 2), [3, 1], (3, 2)), tuple[Any, ...])
assert_type(np.broadcast_shapes((6, 7), (5, 6, 1), 7, (5, 1, 7)), tuple[Any, ...])

assert_type(np.broadcast_arrays(), tuple[()])
assert_type(np.broadcast_arrays(AR_f8_2d), tuple[_Array2D[np.float64]])
assert_type(np.broadcast_arrays(AR_LIKE_f), tuple[npt.NDArray[Any]])
assert_type(np.broadcast_arrays(AR_f8, AR_i8_2d), tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]])
assert_type(np.broadcast_arrays(AR_i8_2d, AR_f8), tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]])
assert_type(np.broadcast_arrays(AR_i8_1d, AR_i8_1d), tuple[_Array1D[np.int64], _Array1D[np.int64]])
assert_type(np.broadcast_arrays(AR_i8_1d, AR_f8_2d), tuple[_Array2D[np.int64], _Array2D[np.float64]])
assert_type(np.broadcast_arrays(AR_f8_2d, AR_i8_1d), tuple[_Array2D[np.float64], _Array2D[np.int64]])
assert_type(np.broadcast_arrays(AR_f8_2d, AR_i8_3d), tuple[_Array3D[np.float64], _Array3D[np.int64]])
assert_type(np.broadcast_arrays(AR_i8_3d, AR_f8_2d), tuple[_Array3D[np.int64], _Array3D[np.float64]])
assert_type(np.broadcast_arrays(AR_f8_2d, f8), tuple[_Array2D[np.float64], _Array2D[np.float64]])
assert_type(np.broadcast_arrays(f8, AR_i8_2d), tuple[_Array2D[np.float64], _Array2D[np.int64]])
assert_type(np.broadcast_arrays(AR_f8_2d, 2.0), tuple[_Array2D[np.float64], _Array2D[Any]])
assert_type(np.broadcast_arrays(2.0, AR_i8_2d), tuple[_Array2D[Any], _Array2D[np.int64]])
assert_type(np.broadcast_arrays(AR_f8_2d, AR_i8_4d), tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]])
assert_type(np.broadcast_arrays(AR_f8_2d, AR_LIKE_f), tuple[npt.NDArray[Any], npt.NDArray[Any]])
assert_type(np.broadcast_arrays(AR_f8, AR_f8, AR_f8), tuple[npt.NDArray[np.float64], ...])
assert_type(np.broadcast_arrays(AR_f8, AR_LIKE_f, AR_f8), tuple[npt.NDArray[Any], ...])
