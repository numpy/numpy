import numpy as np

from ._shape import _AnyShape

type NDArray[ST: np.generic] = np.ndarray[_AnyShape, np.dtype[ST]]

type Array0D[ST: np.generic] = np.ndarray[tuple[()], np.dtype[ST]]
type Array1D[ST: np.generic] = np.ndarray[tuple[int], np.dtype[ST]]
type Array2D[ST: np.generic] = np.ndarray[tuple[int, int], np.dtype[ST]]
type Array3D[ST: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ST]]
type Array4D[ST: np.generic] = np.ndarray[tuple[int, int, int, int], np.dtype[ST]]
type Array5D[ST: np.generic] = np.ndarray[tuple[int, int, int, int, int], np.dtype[ST]]
