from collections.abc import Sequence
from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

type _Array1D[ScalarT: np.generic] = np.ndarray[tuple[int], np.dtype[ScalarT]]
type _Array2D[ScalarT: np.generic] = np.ndarray[tuple[int, int], np.dtype[ScalarT]]
type _Array3D[ScalarT: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ScalarT]]

AR_i4: npt.NDArray[np.int32]
AR_i8: npt.NDArray[np.int64]
AR_f4: npt.NDArray[np.float32]
AR_f8: npt.NDArray[np.float64]
AR_c8: npt.NDArray[np.complex64]
AR_c16: npt.NDArray[np.complex128]
AR_i8_1d: _Array1D[np.int64]
AR_f4_1d: _Array1D[np.float32]
AR_f8_1d: _Array1D[np.float64]
AR_f4_2d: _Array2D[np.float32]
AR_f8_2d: _Array2D[np.float64]

list_i: list[int]
list_f: list[float]
list_c: list[complex]
seq_c: Sequence[complex]
seq_seq_c: Sequence[Sequence[complex]]

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
assert_type(np.histogram_bin_edges(AR_f8, AR_i8), _Array1D[np.int64])
assert_type(np.histogram_bin_edges(AR_f8, list_i), _Array1D[np.int_])
assert_type(np.histogram_bin_edges(AR_f4, list_f), _Array1D[np.float64])

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
assert_type(np.histogram(AR_f8, AR_i8, density=True), tuple[_Array1D[np.float64], _Array1D[np.int64]])
assert_type(np.histogram(AR_f8, AR_i8, density=True, weights=AR_c16), tuple[_Array1D[np.complex128], _Array1D[np.int64]])
assert_type(np.histogram(AR_f8, AR_i8), tuple[_Array1D[np.intp], _Array1D[np.int64]])
assert_type(np.histogram(AR_f8, AR_i8, weights=AR_f4), tuple[_Array1D[np.float32], _Array1D[np.int64]])
assert_type(np.histogram(AR_f8, AR_i8, weights=list_f), tuple[_Array1D[Any], _Array1D[np.int64]])
assert_type(np.histogram(AR_f8, list_i, density=True), tuple[_Array1D[np.float64], _Array1D[np.int_]])
assert_type(np.histogram(AR_f8, list_i, density=True, weights=AR_c16), tuple[_Array1D[np.complex128], _Array1D[np.int_]])
assert_type(np.histogram([1, 2, 1], bins=[0, 1, 2, 3]), tuple[_Array1D[np.intp], _Array1D[np.int_]])
assert_type(np.histogram(AR_f8, list_i, weights=AR_f4), tuple[_Array1D[np.float32], _Array1D[np.int_]])
assert_type(np.histogram(AR_f8, list_i, weights=list_f), tuple[_Array1D[Any], _Array1D[np.int_]])
assert_type(np.histogram(AR_f4, list_f, density=True), tuple[_Array1D[np.float64], _Array1D[np.float64]])
assert_type(np.histogram(AR_f4, list_f, density=True, weights=AR_c16), tuple[_Array1D[np.complex128], _Array1D[np.float64]])
assert_type(np.histogram(AR_f4, list_f), tuple[_Array1D[np.intp], _Array1D[np.float64]])
assert_type(np.histogram(AR_f4, list_f, weights=AR_f4), tuple[_Array1D[np.float32], _Array1D[np.float64]])
assert_type(np.histogram(AR_f4, list_f, weights=list_f), tuple[_Array1D[Any], _Array1D[np.float64]])

assert_type(np.histogramdd(AR_f8_2d, (AR_i8_1d, AR_i8_1d)), tuple[npt.NDArray[np.float64], list[_Array1D[np.int64]]])
assert_type(np.histogramdd(AR_f8_2d, [[0, 1, 2], [0, 1, 2]]), tuple[npt.NDArray[np.float64], list[_Array1D[np.int_]]])
assert_type(np.histogramdd(AR_f8_2d, [list_f, list_f]), tuple[npt.NDArray[np.float64], list[_Array1D[np.float64]]])
assert_type(np.histogramdd(AR_f4), tuple[npt.NDArray[np.float64], list[_Array1D[np.float32]]])
assert_type(np.histogramdd(AR_i8), tuple[npt.NDArray[np.float64], list[_Array1D[np.float64]]])
assert_type(np.histogramdd(AR_f4_1d), tuple[_Array1D[np.float64], list[_Array1D[np.float32]]])
assert_type(np.histogramdd([1.0, np.int64(2)]), tuple[_Array1D[np.float64], list[_Array1D[np.float64]]])
assert_type(np.histogramdd(list_c), tuple[_Array1D[np.float64], list[_Array1D[np.complex128]]])
assert_type(np.histogramdd(seq_c), tuple[_Array1D[np.float64], list[_Array1D[Any]]])
assert_type(np.histogramdd((AR_f4_1d, AR_f4_1d)), tuple[_Array2D[np.float64], list[_Array1D[np.float32]]])
assert_type(np.histogramdd((AR_i8_1d, list_f)), tuple[_Array2D[np.float64], list[_Array1D[np.float64]]])
assert_type(np.histogramdd((list_c, list_c)), tuple[_Array2D[np.float64], list[_Array1D[np.complex128]]])
assert_type(np.histogramdd((AR_f8_1d, seq_c)), tuple[_Array2D[np.float64], list[_Array1D[Any]]])
assert_type(np.histogramdd((AR_f4_1d, AR_f4_1d, AR_f4_1d)), tuple[_Array3D[np.float64], list[_Array1D[np.float32]]])
assert_type(np.histogramdd((AR_i8_1d, AR_i8_1d, list_f)), tuple[_Array3D[np.float64], list[_Array1D[np.float64]]])
assert_type(np.histogramdd((list_c, list_c, list_c)), tuple[_Array3D[np.float64], list[_Array1D[np.complex128]]])
assert_type(np.histogramdd((AR_f8_1d, AR_f8_1d, seq_c)), tuple[_Array3D[np.float64], list[_Array1D[Any]]])
assert_type(np.histogramdd(AR_f4_2d), tuple[npt.NDArray[np.float64], list[_Array1D[np.float32]]])
assert_type(np.histogramdd([[1.0, 2.0], [3.0, 4.0]]), tuple[npt.NDArray[np.float64], list[_Array1D[np.float64]]])
assert_type(np.histogramdd([[1j, 2j], [3j, 4j]]), tuple[npt.NDArray[np.float64], list[_Array1D[np.complex128]]])
assert_type(np.histogramdd(seq_seq_c), tuple[npt.NDArray[np.float64], list[_Array1D[Any]]])
