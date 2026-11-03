from typing import assert_type

import numpy as np
import numpy.typing as npt

type _ArrayND = npt.NDArray[np.int64]

_nd: _ArrayND
_2d: npt.Array2D[np.int8]
_3d: npt.Array3D[np.bool]

# reshape
assert_type(_nd.reshape(None), npt.NDArray[np.int64])
assert_type(_nd.reshape(4), np.ndarray[tuple[int], np.dtype[np.int64]])
assert_type(_nd.reshape((4,)), np.ndarray[tuple[int], np.dtype[np.int64]])
assert_type(_nd.reshape(2, 2), np.ndarray[tuple[int, int], np.dtype[np.int64]])
assert_type(_nd.reshape((2, 2)), np.ndarray[tuple[int, int], np.dtype[np.int64]])

assert_type(_nd.reshape((2, 2), order="C"),  np.ndarray[tuple[int, int], np.dtype[np.int64]])
assert_type(_nd.reshape(4, order="C"),  np.ndarray[tuple[int], np.dtype[np.int64]])

# resize does not return a value

# transpose
assert_type(_nd.transpose(), npt.NDArray[np.int64])
assert_type(_nd.transpose(1, 0), npt.NDArray[np.int64])
assert_type(_nd.transpose((1, 0)), npt.NDArray[np.int64])

# swapaxes
assert_type(_nd.swapaxes(0, 1), _ArrayND)
assert_type(_2d.swapaxes(0, 1), npt.Array2D[np.int8])
assert_type(_3d.swapaxes(0, 1), npt.Array3D[np.bool])

# flatten
assert_type(_nd.flatten(), np.ndarray[tuple[int], np.dtype[np.int64]])
assert_type(_nd.flatten("C"), np.ndarray[tuple[int], np.dtype[np.int64]])

# ravel
assert_type(_nd.ravel(), np.ndarray[tuple[int], np.dtype[np.int64]])
assert_type(_nd.ravel("C"), np.ndarray[tuple[int], np.dtype[np.int64]])

# squeeze
assert_type(_nd.squeeze(), npt.NDArray[np.int64])
assert_type(_nd.squeeze(0), npt.NDArray[np.int64])
assert_type(_nd.squeeze((0, 2)), npt.NDArray[np.int64])
assert_type(_2d.squeeze(axis=0), np.ndarray[tuple[int], np.dtype[np.int8]])
assert_type(_2d.squeeze(axis=(0, 1)), npt.NDArray[np.int8])
assert_type(_3d.squeeze(axis=0), np.ndarray[tuple[int, int], np.dtype[np.bool]])
