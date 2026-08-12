from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]
AR_LIKE_f: list[float]
interface_dict: dict[str, Any]

assert_type(np.lib.stride_tricks.as_strided(AR_f8), npt.NDArray[np.float64])
assert_type(np.lib.stride_tricks.as_strided(AR_LIKE_f), npt.NDArray[Any])
assert_type(np.lib.stride_tricks.as_strided(AR_f8, strides=(1, 5)), npt.NDArray[np.float64])
assert_type(np.lib.stride_tricks.as_strided(AR_f8, shape=[9, 20]), npt.NDArray[np.float64])

assert_type(np.lib.stride_tricks.sliding_window_view(AR_f8, 5), npt.NDArray[np.float64])
assert_type(np.lib.stride_tricks.sliding_window_view(AR_LIKE_f, (1, 5)), npt.NDArray[Any])
assert_type(np.lib.stride_tricks.sliding_window_view(AR_f8, [9], axis=1), npt.NDArray[np.float64])

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

assert_type(np.broadcast_shapes((1, 2), [3, 1], (3, 2)), tuple[Any, ...])
assert_type(np.broadcast_shapes((6, 7), (5, 6, 1), 7, (5, 1, 7)), tuple[Any, ...])

assert_type(np.broadcast_arrays(AR_f8, AR_f8), tuple[npt.NDArray[Any], ...])
assert_type(np.broadcast_arrays(AR_f8, AR_LIKE_f), tuple[npt.NDArray[Any], ...])
