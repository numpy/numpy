from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]

AR_i4: npt.NDArray[np.int32]
AR_i8: npt.NDArray[np.int64]
AR_f4: npt.NDArray[np.float32]
AR_f8: npt.NDArray[np.float64]
AR_c8: npt.NDArray[np.complex64]
AR_c16: npt.NDArray[np.complex128]

list_i: list[int]
list_f: list[float]
list_c: list[complex]

###

assert_type(np.histogram_bin_edges(AR_i8, bins="auto"), _Array1D[np.float64])
assert_type(np.histogram_bin_edges(AR_i8, bins="rice", range=(0, 3)), _Array1D[np.float64])
assert_type(np.histogram_bin_edges(AR_i8, bins="scott", weights=AR_f8), _Array1D[np.float64])
assert_type(np.histogram_bin_edges(AR_f4), _Array1D[np.float32])
assert_type(np.histogram_bin_edges(AR_f8), _Array1D[np.float64])
assert_type(np.histogram_bin_edges(AR_c8), _Array1D[np.complex64])
assert_type(np.histogram_bin_edges(AR_c16), _Array1D[np.complex128])
assert_type(np.histogram_bin_edges(list_i), _Array1D[np.float64])
assert_type(np.histogram_bin_edges(list_f), _Array1D[np.float64])
assert_type(np.histogram_bin_edges(list_c), _Array1D[np.complex128])

assert_type(np.histogram(AR_i8, bins="auto"), tuple[_Array1D[np.intp], _Array1D[np.float64]])
assert_type(np.histogram(AR_i8, bins="rice", range=(0, 3)), tuple[_Array1D[np.intp], _Array1D[np.float64]])
assert_type(np.histogram(AR_i8, bins="scott", weights=AR_f8), tuple[_Array1D[np.float64], _Array1D[np.float64]])
assert_type(np.histogram(AR_f8, bins=1, density=True), tuple[_Array1D[np.float64], _Array1D[np.float64]])
assert_type(np.histogram(AR_f4), tuple[_Array1D[np.intp], _Array1D[np.float32]])
assert_type(np.histogram(AR_f8), tuple[_Array1D[np.intp], _Array1D[np.float64]])
assert_type(np.histogram(AR_c8), tuple[_Array1D[np.intp], _Array1D[np.complex64]])
assert_type(np.histogram(AR_c16), tuple[_Array1D[np.intp], _Array1D[np.complex128]])
assert_type(np.histogram(list_i), tuple[_Array1D[np.intp], _Array1D[np.float64]])
assert_type(np.histogram(list_f), tuple[_Array1D[np.intp], _Array1D[np.float64]])
assert_type(np.histogram(list_c), tuple[_Array1D[np.intp], _Array1D[np.complex128]])
assert_type(np.histogram(AR_f4, density=True), tuple[_Array1D[np.float64], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, density=True, weights=AR_i4), tuple[_Array1D[np.float64], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, density=True, weights=AR_f4), tuple[_Array1D[np.float64], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, density=True, weights=AR_f8), tuple[_Array1D[np.float64], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, density=True, weights=AR_c8), tuple[_Array1D[np.complex128], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, density=True, weights=AR_c16), tuple[_Array1D[np.complex128], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, density=True, weights=list_i), tuple[_Array1D[np.float64], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, density=True, weights=list_f), tuple[_Array1D[np.float64], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, density=True, weights=list_c), tuple[_Array1D[np.complex128], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, weights=AR_i4), tuple[_Array1D[np.int32], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, weights=AR_f4), tuple[_Array1D[np.float32], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, weights=AR_f8), tuple[_Array1D[np.float64], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, weights=AR_c8), tuple[_Array1D[np.complex64], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, weights=AR_c16), tuple[_Array1D[np.complex128], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, weights=list_i), tuple[_Array1D[np.intp], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, weights=list_f), tuple[_Array1D[Any], _Array1D[np.float32]])
assert_type(np.histogram(AR_f4, weights=list_c), tuple[_Array1D[Any], _Array1D[np.float32]])

assert_type(np.histogramdd(AR_i8, bins=[1]), tuple[npt.NDArray[np.float64], tuple[_Array1D[np.float64], ...]])
assert_type(np.histogramdd(AR_i8, range=[(0, 3)]), tuple[npt.NDArray[np.float64], tuple[_Array1D[np.float64], ...]])
assert_type(np.histogramdd(AR_i8, weights=AR_f8), tuple[npt.NDArray[np.float64], tuple[_Array1D[np.float64], ...]])
assert_type(np.histogramdd(AR_f8, density=True), tuple[npt.NDArray[np.float64], tuple[_Array1D[np.float64], ...]])
assert_type(np.histogramdd(AR_i4), tuple[npt.NDArray[np.float64], tuple[_Array1D[np.float64], ...]])
assert_type(np.histogramdd(AR_i8), tuple[npt.NDArray[np.float64], tuple[_Array1D[np.float64], ...]])
assert_type(np.histogramdd(AR_f4), tuple[npt.NDArray[np.float64], tuple[_Array1D[np.float32], ...]])
assert_type(np.histogramdd(AR_c8), tuple[npt.NDArray[np.float64], tuple[_Array1D[np.complex64], ...]])
assert_type(np.histogramdd(AR_c16), tuple[npt.NDArray[np.float64], tuple[_Array1D[np.complex128], ...]])
