from collections import deque
from pathlib import Path
from typing import Any, assert_type
from typing_extensions import CapsuleType

import numpy as np
import numpy.typing as npt
from numpy._typing import _AnyShape

class SubClass[ScalarT: np.generic](np.ndarray[_AnyShape, np.dtype[ScalarT]]): ...

class IntoSubClass[ScalarT: np.generic]:
    def __array__(self) -> SubClass[ScalarT]: ...

i8: np.int64

A: npt.NDArray[np.float64]
B: SubClass[np.float64]
C: list[int]
D: SubClass[np.float64 | np.int64]
E: IntoSubClass[np.float64 | np.int64]

class _SupportsDLPack:
    def __dlpack__(self, /, *, stream: None = None) -> CapsuleType: ...

_dlpack_obj: _SupportsDLPack

_f32_0d: np.float32
_f32_1d: npt.Array1D[np.float32]
_f32_2d: npt.Array2D[np.float32]
_f32_3d: npt.Array3D[np.float32]
_f32_4d_list: list[npt.Array4D[np.float32]]
_f64_0d: npt.Array0D[np.float64]
_obj_str_1d: npt.Array1D[np.object_[str]]

_py_b_1d: list[bool]
_py_b_2d: list[list[bool]]
_py_b_3d: list[list[list[bool]]]
_py_i_1d: list[int]
_py_i_2d: list[list[int]]
_py_i_3d: list[list[list[int]]]
_py_f_1d: list[float]
_py_f_2d: list[list[float]]
_py_f_3d: list[list[list[float]]]
_py_c_1d: list[complex]
_py_c_2d: list[list[complex]]
_py_c_3d: list[list[list[complex]]]

_py_rec_1d: list[tuple[int, float]]
_py_rec_2d: list[list[tuple[int, float]]]
_rec_spec: list[tuple[str, str]]
_void_dtype: np.dtype[np.void]

mixed_shape: tuple[int, np.int64]

def _func_1d_i8(i: npt.Array1D[np.int8]) -> npt.Array1D[np.int8]: ...
def _func_1d_i64(i: npt.Array1D[np.int_]) -> npt.Array1D[np.int_]: ...
def _func_1d_f64(i: npt.Array1D[np.float64]) -> npt.Array1D[np.float64]: ...
def _func_1d[ScalarT: np.generic](i: npt.Array1D[ScalarT]) -> npt.Array1D[ScalarT]: ...
def _func_2d_i8(i: npt.Array2D[np.int8], j: npt.Array2D[np.int8]) -> npt.Array2D[np.int8]: ...
def _func_2d_i64(i: npt.Array2D[np.int_], j: npt.Array2D[np.int_]) -> npt.Array2D[np.int_]: ...
def _func_2d_f64(i: npt.Array2D[np.float64], j: npt.Array2D[np.float64]) -> npt.Array2D[np.float64]: ...
def _func_2d[ScalarT: np.generic](i: npt.Array2D[ScalarT], j: npt.Array2D[ScalarT]) -> npt.Array2D[ScalarT]: ...
def _func_3d_i8(i: npt.Array3D[np.int8], j: npt.Array3D[np.int8], k: npt.Array3D[np.int8]) -> npt.Array3D[np.int8]: ...
def _func_3d_i64(i: npt.Array3D[np.int_], j: npt.Array3D[np.int_], k: npt.Array3D[np.int_]) -> npt.Array3D[np.int_]: ...
def _func_3d_f64(i: npt.Array3D[np.float64], j: npt.Array3D[np.float64], k: npt.Array3D[np.float64]) -> npt.Array3D[np.float64]: ...
def _func_3d[ScalarT: np.generic](i: npt.Array3D[ScalarT], j: npt.Array3D[ScalarT], k: npt.Array3D[ScalarT]) -> npt.Array3D[ScalarT]: ...
def _func_nd(*args: npt.NDArray[np.float64]) -> SubClass[np.float64]: ...

###

assert_type(np.array(_py_b_1d), npt.Array1D[np.bool])
assert_type(np.array(_py_b_2d), npt.Array2D[np.bool])
assert_type(np.array(_py_b_3d), npt.Array3D[np.bool])
assert_type(np.array(_py_i_1d), npt.Array1D[np.int_])
assert_type(np.array(_py_i_2d), npt.Array2D[np.int_])
assert_type(np.array(_py_i_3d), npt.Array3D[np.int_])
assert_type(np.array(_py_f_1d), npt.Array1D[np.float64])
assert_type(np.array(_py_f_2d), npt.Array2D[np.float64])
assert_type(np.array(_py_f_3d), npt.Array3D[np.float64])
assert_type(np.array(_py_c_1d), npt.Array1D[np.complex128])
assert_type(np.array(_py_c_2d), npt.Array2D[np.complex128])
assert_type(np.array(_py_c_3d), npt.Array3D[np.complex128])
assert_type(np.array(_py_i_1d, dtype=np.float32), npt.Array1D[np.float32])
assert_type(np.array(_py_i_2d, dtype=np.float32), npt.Array2D[np.float32])
assert_type(np.array(_py_i_3d, dtype=np.float32), npt.Array3D[np.float32])
assert_type(np.array(_py_i_1d, dtype="f"), npt.Array1D[Any])
assert_type(np.array(_py_i_2d, dtype="f"), npt.Array2D[Any])
assert_type(np.array(_py_i_3d, dtype="f"), npt.Array3D[Any])
assert_type(np.array(_f32_0d), npt.Array0D[np.float32])
assert_type(np.array(_f32_1d), npt.Array1D[np.float32])
assert_type(np.array(_f32_2d), npt.Array2D[np.float32])
assert_type(np.array(_f32_1d, dtype="f8"), npt.Array1D[Any])
assert_type(np.array(A), npt.NDArray[np.float64])
assert_type(np.array(B), npt.NDArray[np.float64])
assert_type(np.array([1, 1.0]), npt.Array1D[np.float64])
assert_type(np.array([[1, 2], [3, 4.5]]), npt.Array2D[np.float64])
assert_type(np.array([[[1, 2]], [[3, 4.5]]]), npt.Array3D[np.float64])
assert_type(np.array(deque([1, 2, 3])), npt.NDArray[Any])
assert_type(np.array(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.array(A, dtype="c16"), npt.NDArray[Any])
assert_type(np.array(A, like=A), npt.NDArray[np.float64])
assert_type(np.array(A, subok=True), npt.NDArray[np.float64])
assert_type(np.array(B, subok=True), SubClass[np.float64])
assert_type(np.array(B, subok=True, ndmin=0), SubClass[np.float64])
assert_type(np.array(B, subok=True, ndmin=1), npt.NDArray[np.float64])  # subtype erased because unknown shape-type could be modified
assert_type(np.array(D), npt.NDArray[np.float64 | np.int64])
assert_type(np.array(E, subok=True), SubClass[np.float64 | np.int64])
# https://github.com/numpy/numpy/issues/29245
assert_type(np.array([]), npt.NDArray[Any])
assert_type(np.array([], dtype=np.bool), npt.Array1D[np.bool[Any]])
assert_type(np.array(None, dtype=np.object_), npt.Array0D[np.object_[None]])
assert_type(np.array(1, dtype=np.object_), npt.Array0D[np.object_[int]])
assert_type(np.array(_py_i_1d, dtype=np.object_), npt.Array1D[np.object_[int]])
# mypy bug; pyright correctly infer `object_[int | Any]` instead of `object_[Any]`
assert_type(np.array(_py_i_2d, dtype=np.object_), npt.NDArray[np.object_[Any]])
assert_type(np.array(_f32_0d, dtype=np.object_), npt.Array0D[np.object_[float]])
assert_type(np.array(_f32_1d, dtype=np.object_), npt.Array1D[np.object_[float]])
assert_type(np.array(_f32_2d, dtype=np.object_), npt.Array2D[np.object_[float]])
assert_type(np.array(True), npt.Array0D[np.bool])
assert_type(np.array(1), npt.Array0D[np.int_ | Any])
assert_type(np.array(1.0), npt.Array0D[np.float64 | Any])
assert_type(np.array(1j), npt.Array0D[np.complex128 | Any])
assert_type(np.array(1, dtype=np.float32), npt.Array0D[np.float32])
assert_type(np.array(b"x", dtype=np.bytes_), npt.NDArray[np.bytes_])
assert_type(np.array([b"x"], dtype=np.bytes_), npt.NDArray[np.bytes_])
assert_type(np.array(1, dtype="f"), npt.Array0D[Any])
assert_type(np.array([[np.float64(1), 2]], dtype=np.float32), npt.Array2D[np.float32])
assert_type(np.array([[np.float64(1), 2]], dtype="f4"), npt.Array2D[Any])
assert_type(np.array(b"x"), npt.NDArray[Any])
assert_type(np.array([b"x"]), npt.NDArray[Any])
assert_type(np.array(_py_rec_1d, dtype=_void_dtype), npt.Array1D[np.void])
assert_type(np.array(_py_rec_2d, dtype=_rec_spec), npt.NDArray[np.void])

assert_type(np.zeros([1, 5, 6]), npt.NDArray[np.float64])
assert_type(np.zeros([1, 5, 6], dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.zeros([1, 5, 6], dtype="c16"), npt.NDArray[Any])
assert_type(np.zeros(3, dtype=bool), npt.Array1D[np.bool])
assert_type(np.zeros((2, 3), dtype=bool), npt.Array2D[np.bool])
assert_type(np.zeros([1, 5, 6], dtype=bool), npt.NDArray[np.bool])
assert_type(np.zeros(mixed_shape), npt.NDArray[np.float64])

assert_type(np.empty([1, 5, 6]), npt.NDArray[np.float64])
assert_type(np.empty([1, 5, 6], dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.empty([1, 5, 6], dtype="c16"), npt.NDArray[Any])
assert_type(np.empty(3, dtype=bool), npt.Array1D[np.bool])
assert_type(np.empty((2, 3), dtype=bool), npt.Array2D[np.bool])
assert_type(np.empty([1, 5, 6], dtype=bool), npt.NDArray[np.bool])
assert_type(np.empty(mixed_shape), npt.NDArray[np.float64])

assert_type(np.concatenate(A), npt.NDArray[np.float64])
assert_type(np.concatenate([A, A]), npt.NDArray[np.float64])
assert_type(np.concatenate([[1], A]), npt.NDArray[Any])
assert_type(np.concatenate([[1], [1]]), npt.NDArray[Any])
assert_type(np.concatenate((A, A)), npt.NDArray[np.float64])
assert_type(np.concatenate(([1], [1])), npt.NDArray[Any])
assert_type(np.concatenate([1, 1.0]), npt.NDArray[Any])
assert_type(np.concatenate(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.concatenate(A, dtype="c16"), npt.NDArray[Any])
assert_type(np.concatenate([1, 1.0], out=A), npt.NDArray[np.float64])

assert_type(np.asarray(A), npt.NDArray[np.float64])
assert_type(np.asarray(B), npt.NDArray[np.float64])
assert_type(np.asarray(C), npt.Array1D[np.int_])
assert_type(np.asarray(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.asarray(A, dtype="c16"), npt.NDArray[Any])
assert_type(np.asarray(_f32_0d), npt.Array0D[np.float32])
assert_type(np.asarray(_f32_1d), npt.Array1D[np.float32])
assert_type(np.asarray(_f32_2d), npt.Array2D[np.float32])
assert_type(np.asarray(_f32_3d), npt.Array3D[np.float32])
assert_type(np.asarray(_f32_1d, dtype=np.float64), npt.Array1D[np.float64])
assert_type(np.asarray(_f32_1d, dtype="f8"), npt.Array1D[Any])
assert_type(np.asarray(i8, dtype=np.object_), npt.Array0D[np.object_[int]])
assert_type(np.asarray(_f32_1d, dtype=np.object_), npt.Array1D[np.object_[float]])
assert_type(np.asarray(_f32_1d, dtype=np.void), npt.Array1D[np.void])
assert_type(np.asarray(1, dtype=np.object_), npt.Array0D[np.object_[int]])
assert_type(np.asarray(_py_i_1d, dtype=np.object_), npt.Array1D[np.object_[int]])
# mypy bug; pyright correctly infer `object_[int | Any]` instead of `object_[Any]`
assert_type(np.asarray(_py_i_2d, dtype=np.object_), npt.NDArray[np.object_[Any]])
assert_type(np.asarray([]), npt.NDArray[Any])
assert_type(np.asarray([[]]), npt.Array2D[np.bool])
assert_type(np.asarray(True), npt.Array0D[np.bool_])
assert_type(np.asarray(1), npt.Array0D[np.int_ | Any])
assert_type(np.asarray(1.0), npt.Array0D[np.float64 | Any])
assert_type(np.asarray(1j), npt.Array0D[np.complex128 | Any])
assert_type(np.asarray(_py_b_1d), npt.Array1D[np.bool_])
assert_type(np.asarray(_py_b_2d), npt.Array2D[np.bool_])
assert_type(np.asarray(_py_b_3d), npt.Array3D[np.bool_])
assert_type(np.asarray(_py_i_1d), npt.Array1D[np.int_])
assert_type(np.asarray(_py_i_2d), npt.Array2D[np.int_])
assert_type(np.asarray(_py_i_3d), npt.Array3D[np.int_])
assert_type(np.asarray(_py_f_1d), npt.Array1D[np.float64])
assert_type(np.asarray(_py_f_2d), npt.Array2D[np.float64])
assert_type(np.asarray(_py_f_3d), npt.Array3D[np.float64])
assert_type(np.asarray(_py_c_1d), npt.Array1D[np.complex128])
assert_type(np.asarray(_py_c_2d), npt.Array2D[np.complex128])
assert_type(np.asarray(_py_c_3d), npt.Array3D[np.complex128])
assert_type(np.asarray([1, 1.0]), npt.Array1D[np.float64])
assert_type(np.asarray([[1, 2], [3, 4.5]]), npt.Array2D[np.float64])
assert_type(np.asarray([[[1, 2]], [[3, 4.5]]]), npt.Array3D[np.float64])
assert_type(np.asarray(1, dtype=np.float32), npt.Array0D[np.float32])
assert_type(np.asarray(1, dtype="f"), npt.Array0D[Any])
assert_type(np.asarray(b"x", dtype=np.bytes_), npt.NDArray[np.bytes_])
assert_type(np.asarray(b"x", dtype="S"), npt.NDArray[Any])
assert_type(np.asarray(_py_i_1d, dtype=np.float32), npt.Array1D[np.float32])
assert_type(np.asarray(_py_i_1d, dtype="f4"), npt.Array1D[Any])
assert_type(np.asarray([b"x"], dtype=np.bytes_), npt.NDArray[np.bytes_])
assert_type(np.asarray([b"x"], dtype="S"), npt.NDArray[Any])
assert_type(np.asarray(_py_i_2d, dtype=np.float32), npt.Array2D[np.float32])
assert_type(np.asarray(_py_i_2d, dtype="f4"), npt.Array2D[Any])
assert_type(np.asarray([[np.float64(1), 2]], dtype=np.float32), npt.Array2D[np.float32])
assert_type(np.asarray([[np.float64(1), 2]], dtype="f4"), npt.Array2D[Any])
assert_type(np.asarray(_py_i_3d, dtype=np.float32), npt.Array3D[np.float32])
assert_type(np.asarray(_py_i_3d, dtype="f4"), npt.Array3D[Any])
assert_type(np.asarray(_py_rec_1d, dtype=_void_dtype), npt.Array1D[np.void])
assert_type(np.asarray(_py_rec_2d, dtype=_rec_spec), npt.NDArray[np.void])

assert_type(np.asanyarray(A), npt.NDArray[np.float64])
assert_type(np.asanyarray(B), SubClass[np.float64])
assert_type(np.asanyarray(E), SubClass[np.float64 | np.int64])
assert_type(np.asanyarray(C), npt.Array1D[np.int_])
assert_type(np.asanyarray(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.asanyarray(A, dtype="c16"), npt.NDArray[Any])
assert_type(np.asanyarray(_f32_0d), npt.Array0D[np.float32])
assert_type(np.asanyarray(_f32_1d), npt.Array1D[np.float32])
assert_type(np.asanyarray(_f32_2d), npt.Array2D[np.float32])
assert_type(np.asanyarray(_f32_3d), npt.Array3D[np.float32])
assert_type(np.asanyarray(_f32_1d, dtype=np.float64), npt.Array1D[np.float64])
assert_type(np.asanyarray(_f32_1d, dtype="f8"), npt.Array1D[Any])
assert_type(np.asanyarray(i8, dtype=np.object_), npt.Array0D[np.object_[int]])
assert_type(np.asanyarray(_f32_1d, dtype=np.object_), npt.Array1D[np.object_[float]])
assert_type(np.asanyarray(_f32_1d, dtype=np.void), npt.Array1D[np.void])
assert_type(np.asanyarray(1, dtype=np.object_), npt.Array0D[np.object_[int]])
assert_type(np.asanyarray(_py_i_1d, dtype=np.object_), npt.Array1D[np.object_[int]])
# mypy bug; pyright correctly infer `object_[int | Any]` instead of `object_[Any]`
assert_type(np.asanyarray(_py_i_2d, dtype=np.object_), npt.NDArray[np.object_[Any]])
assert_type(np.asanyarray([]), npt.NDArray[Any])
assert_type(np.asanyarray([[]]), npt.Array2D[np.bool])
assert_type(np.asanyarray(True), npt.Array0D[np.bool_])
assert_type(np.asanyarray(1), npt.Array0D[np.int_ | Any])
assert_type(np.asanyarray(1.0), npt.Array0D[np.float64 | Any])
assert_type(np.asanyarray(1j), npt.Array0D[np.complex128 | Any])
assert_type(np.asanyarray(_py_b_1d), npt.Array1D[np.bool_])
assert_type(np.asanyarray(_py_b_2d), npt.Array2D[np.bool_])
assert_type(np.asanyarray(_py_b_3d), npt.Array3D[np.bool_])
assert_type(np.asanyarray(_py_i_1d), npt.Array1D[np.int_])
assert_type(np.asanyarray(_py_i_2d), npt.Array2D[np.int_])
assert_type(np.asanyarray(_py_i_3d), npt.Array3D[np.int_])
assert_type(np.asanyarray(_py_f_1d), npt.Array1D[np.float64])
assert_type(np.asanyarray(_py_f_2d), npt.Array2D[np.float64])
assert_type(np.asanyarray(_py_f_3d), npt.Array3D[np.float64])
assert_type(np.asanyarray(_py_c_1d), npt.Array1D[np.complex128])
assert_type(np.asanyarray(_py_c_2d), npt.Array2D[np.complex128])
assert_type(np.asanyarray(_py_c_3d), npt.Array3D[np.complex128])
assert_type(np.asanyarray([1, 1.0]), npt.Array1D[np.float64])
assert_type(np.asanyarray([[1, 2], [3, 4.5]]), npt.Array2D[np.float64])
assert_type(np.asanyarray([[[1, 2]], [[3, 4.5]]]), npt.Array3D[np.float64])
assert_type(np.asanyarray(1, dtype=np.float32), npt.Array0D[np.float32])
assert_type(np.asanyarray(1, dtype="f"), npt.Array0D[Any])
assert_type(np.asanyarray(b"x", dtype=np.bytes_), npt.NDArray[np.bytes_])
assert_type(np.asanyarray(b"x", dtype="S"), npt.NDArray[Any])
assert_type(np.asanyarray(_py_i_1d, dtype=np.float32), npt.Array1D[np.float32])
assert_type(np.asanyarray(_py_i_1d, dtype="f4"), npt.Array1D[Any])
assert_type(np.asanyarray([b"x"], dtype=np.bytes_), npt.NDArray[np.bytes_])
assert_type(np.asanyarray([b"x"], dtype="S"), npt.NDArray[Any])
assert_type(np.asanyarray(_py_i_2d, dtype=np.float32), npt.Array2D[np.float32])
assert_type(np.asanyarray(_py_i_2d, dtype="f4"), npt.Array2D[Any])
assert_type(np.asanyarray([[np.float64(1), 2]], dtype=np.float32), npt.Array2D[np.float32])
assert_type(np.asanyarray([[np.float64(1), 2]], dtype="f4"), npt.Array2D[Any])
assert_type(np.asanyarray(_py_i_3d, dtype=np.float32), npt.Array3D[np.float32])
assert_type(np.asanyarray(_py_i_3d, dtype="f4"), npt.Array3D[Any])
assert_type(np.asanyarray(_py_rec_1d, dtype=_void_dtype), npt.Array1D[np.void])
assert_type(np.asanyarray(_py_rec_2d, dtype=_rec_spec), npt.NDArray[np.void])

# same as below
assert_type(np.ascontiguousarray(A), npt.NDArray[np.float64])
assert_type(np.ascontiguousarray(B), npt.NDArray[np.float64])
assert_type(np.ascontiguousarray(C), npt.Array1D[np.int_])
assert_type(np.ascontiguousarray(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.ascontiguousarray(A, dtype="c16"), npt.NDArray[Any])
assert_type(np.ascontiguousarray(_f64_0d), npt.Array1D[np.float64])
assert_type(np.ascontiguousarray(_f64_0d, dtype=np.int64), npt.Array1D[np.int64])
assert_type(np.ascontiguousarray(_f64_0d, dtype="c16"), npt.Array1D[Any])
assert_type(np.ascontiguousarray(_f32_0d), npt.Array1D[np.float32])
assert_type(np.ascontiguousarray(_f32_1d), npt.Array1D[np.float32])
assert_type(np.ascontiguousarray(_f32_2d), npt.Array2D[np.float32])
assert_type(np.ascontiguousarray(_f32_3d), npt.Array3D[np.float32])
assert_type(np.ascontiguousarray(_f32_1d, dtype=np.float64), npt.Array1D[np.float64])
assert_type(np.ascontiguousarray(_f32_1d, dtype="f8"), npt.Array1D[Any])
assert_type(np.ascontiguousarray(i8, dtype=np.object_), npt.Array1D[np.object_[int]])
assert_type(np.ascontiguousarray(_f32_0d, dtype=np.object_), npt.Array1D[np.object_[float]])
assert_type(np.ascontiguousarray(_f32_1d, dtype=np.object_), npt.Array1D[np.object_[float]])
assert_type(np.ascontiguousarray(_f32_1d, dtype=np.void), npt.Array1D[np.void])
assert_type(np.ascontiguousarray(1, dtype=np.object_), npt.Array1D[np.object_[int]])
assert_type(np.ascontiguousarray(_py_i_1d, dtype=np.object_), npt.Array1D[np.object_[int]])
# mypy bug; pyright correctly infer `object_[int | Any]` instead of `object_[Any]`
assert_type(np.ascontiguousarray(_py_i_2d, dtype=np.object_), npt.NDArray[np.object_[Any]])
assert_type(np.ascontiguousarray([]), npt.NDArray[Any])
assert_type(np.ascontiguousarray([[]]), npt.Array2D[np.bool])
assert_type(np.ascontiguousarray(True), npt.Array1D[np.bool_])
assert_type(np.ascontiguousarray(1), npt.Array1D[np.int_ | Any])
assert_type(np.ascontiguousarray(1.0), npt.Array1D[np.float64 | Any])
assert_type(np.ascontiguousarray(1j), npt.Array1D[np.complex128 | Any])
assert_type(np.ascontiguousarray(_py_b_1d), npt.Array1D[np.bool_])
assert_type(np.ascontiguousarray(_py_b_2d), npt.Array2D[np.bool_])
assert_type(np.ascontiguousarray(_py_b_3d), npt.Array3D[np.bool_])
assert_type(np.ascontiguousarray(_py_i_1d), npt.Array1D[np.int_])
assert_type(np.ascontiguousarray(_py_i_2d), npt.Array2D[np.int_])
assert_type(np.ascontiguousarray(_py_i_3d), npt.Array3D[np.int_])
assert_type(np.ascontiguousarray(_py_f_1d), npt.Array1D[np.float64])
assert_type(np.ascontiguousarray(_py_f_2d), npt.Array2D[np.float64])
assert_type(np.ascontiguousarray(_py_f_3d), npt.Array3D[np.float64])
assert_type(np.ascontiguousarray(_py_c_1d), npt.Array1D[np.complex128])
assert_type(np.ascontiguousarray(_py_c_2d), npt.Array2D[np.complex128])
assert_type(np.ascontiguousarray(_py_c_3d), npt.Array3D[np.complex128])
assert_type(np.ascontiguousarray([1, 1.0]), npt.Array1D[np.float64])
assert_type(np.ascontiguousarray([[1, 2], [3, 4.5]]), npt.Array2D[np.float64])
assert_type(np.ascontiguousarray([[[1, 2]], [[3, 4.5]]]), npt.Array3D[np.float64])
assert_type(np.ascontiguousarray(1, dtype=np.float32), npt.Array1D[np.float32])
assert_type(np.ascontiguousarray(1, dtype="f"), npt.Array1D[Any])
assert_type(np.ascontiguousarray(_py_i_1d, dtype=np.float32), npt.Array1D[np.float32])
assert_type(np.ascontiguousarray(_py_i_1d, dtype="f4"), npt.Array1D[Any])
assert_type(np.ascontiguousarray([b"x"], dtype=np.bytes_), npt.NDArray[np.bytes_])
assert_type(np.ascontiguousarray([b"x"], dtype="S"), npt.NDArray[Any])
assert_type(np.ascontiguousarray(_py_i_2d, dtype=np.float32), npt.Array2D[np.float32])
assert_type(np.ascontiguousarray(_py_i_2d, dtype="f4"), npt.Array2D[Any])
assert_type(np.ascontiguousarray([[np.float64(1), 2]], dtype=np.float32), npt.Array2D[np.float32])
assert_type(np.ascontiguousarray([[np.float64(1), 2]], dtype="f4"), npt.Array2D[Any])
assert_type(np.ascontiguousarray(_py_i_3d, dtype=np.float32), npt.Array3D[np.float32])
assert_type(np.ascontiguousarray(_py_i_3d, dtype="f4"), npt.Array3D[Any])
assert_type(np.ascontiguousarray(_py_rec_1d, dtype=_void_dtype), npt.Array1D[np.void])
assert_type(np.ascontiguousarray(_py_rec_2d, dtype=_rec_spec), npt.NDArray[np.void])

# same as above
assert_type(np.asfortranarray(A), npt.NDArray[np.float64])
assert_type(np.asfortranarray(B), npt.NDArray[np.float64])
assert_type(np.asfortranarray(C), npt.Array1D[np.int_])
assert_type(np.asfortranarray(A, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.asfortranarray(A, dtype="c16"), npt.NDArray[Any])
assert_type(np.asfortranarray(_f64_0d), npt.Array1D[np.float64])
assert_type(np.asfortranarray(_f64_0d, dtype=np.int64), npt.Array1D[np.int64])
assert_type(np.asfortranarray(_f64_0d, dtype="c16"), npt.Array1D[Any])
assert_type(np.asfortranarray(_f32_0d), npt.Array1D[np.float32])
assert_type(np.asfortranarray(_f32_1d), npt.Array1D[np.float32])
assert_type(np.asfortranarray(_f32_2d), npt.Array2D[np.float32])
assert_type(np.asfortranarray(_f32_3d), npt.Array3D[np.float32])
assert_type(np.asfortranarray(_f32_1d, dtype=np.float64), npt.Array1D[np.float64])
assert_type(np.asfortranarray(_f32_1d, dtype="f8"), npt.Array1D[Any])
assert_type(np.asfortranarray(i8, dtype=np.object_), npt.Array1D[np.object_[int]])
assert_type(np.asfortranarray(_f32_0d, dtype=np.object_), npt.Array1D[np.object_[float]])
assert_type(np.asfortranarray(_f32_1d, dtype=np.object_), npt.Array1D[np.object_[float]])
assert_type(np.asfortranarray(_f32_1d, dtype=np.void), npt.Array1D[np.void])
assert_type(np.asfortranarray(1, dtype=np.object_), npt.Array1D[np.object_[int]])
assert_type(np.asfortranarray(_py_i_1d, dtype=np.object_), npt.Array1D[np.object_[int]])
# mypy bug; pyright correctly infer `object_[int | Any]` instead of `object_[Any]`
assert_type(np.asfortranarray(_py_i_2d, dtype=np.object_), npt.NDArray[np.object_[Any]])
assert_type(np.asfortranarray([]), npt.NDArray[Any])
assert_type(np.asfortranarray([[]]), npt.Array2D[np.bool])
assert_type(np.asfortranarray(True), npt.Array1D[np.bool_])
assert_type(np.asfortranarray(1), npt.Array1D[np.int_ | Any])
assert_type(np.asfortranarray(1.0), npt.Array1D[np.float64 | Any])
assert_type(np.asfortranarray(1j), npt.Array1D[np.complex128 | Any])
assert_type(np.asfortranarray(_py_b_1d), npt.Array1D[np.bool_])
assert_type(np.asfortranarray(_py_b_2d), npt.Array2D[np.bool_])
assert_type(np.asfortranarray(_py_b_3d), npt.Array3D[np.bool_])
assert_type(np.asfortranarray(_py_i_1d), npt.Array1D[np.int_])
assert_type(np.asfortranarray(_py_i_2d), npt.Array2D[np.int_])
assert_type(np.asfortranarray(_py_i_3d), npt.Array3D[np.int_])
assert_type(np.asfortranarray(_py_f_1d), npt.Array1D[np.float64])
assert_type(np.asfortranarray(_py_f_2d), npt.Array2D[np.float64])
assert_type(np.asfortranarray(_py_f_3d), npt.Array3D[np.float64])
assert_type(np.asfortranarray(_py_c_1d), npt.Array1D[np.complex128])
assert_type(np.asfortranarray(_py_c_2d), npt.Array2D[np.complex128])
assert_type(np.asfortranarray(_py_c_3d), npt.Array3D[np.complex128])
assert_type(np.asfortranarray([1, 1.0]), npt.Array1D[np.float64])
assert_type(np.asfortranarray([[1, 2], [3, 4.5]]), npt.Array2D[np.float64])
assert_type(np.asfortranarray([[[1, 2]], [[3, 4.5]]]), npt.Array3D[np.float64])
assert_type(np.asfortranarray(1, dtype=np.float32), npt.Array1D[np.float32])
assert_type(np.asfortranarray(1, dtype="f"), npt.Array1D[Any])
assert_type(np.asfortranarray(_py_i_1d, dtype=np.float32), npt.Array1D[np.float32])
assert_type(np.asfortranarray(_py_i_1d, dtype="f4"), npt.Array1D[Any])
assert_type(np.asfortranarray([b"x"], dtype=np.bytes_), npt.NDArray[np.bytes_])
assert_type(np.asfortranarray([b"x"], dtype="S"), npt.NDArray[Any])
assert_type(np.asfortranarray(_py_i_2d, dtype=np.float32), npt.Array2D[np.float32])
assert_type(np.asfortranarray(_py_i_2d, dtype="f4"), npt.Array2D[Any])
assert_type(np.asfortranarray([[np.float64(1), 2]], dtype=np.float32), npt.Array2D[np.float32])
assert_type(np.asfortranarray([[np.float64(1), 2]], dtype="f4"), npt.Array2D[Any])
assert_type(np.asfortranarray(_py_i_3d, dtype=np.float32), npt.Array3D[np.float32])
assert_type(np.asfortranarray(_py_i_3d, dtype="f4"), npt.Array3D[Any])
assert_type(np.asfortranarray(_py_rec_1d, dtype=_void_dtype), npt.Array1D[np.void])
assert_type(np.asfortranarray(_py_rec_2d, dtype=_rec_spec), npt.NDArray[np.void])

assert_type(np.fromstring("1 1 1", sep=" "), npt.Array1D[np.float64])
assert_type(np.fromstring(b"1 1 1", sep=" "), npt.Array1D[np.float64])
assert_type(np.fromstring("1 1 1", dtype=np.int64, sep=" "), npt.Array1D[np.int64])
assert_type(np.fromstring(b"1 1 1", dtype=np.int64, sep=" "), npt.Array1D[np.int64])
assert_type(np.fromstring("1 1 1", dtype="c16", sep=" "), npt.Array1D[Any])
assert_type(np.fromstring(b"1 1 1", dtype="c16", sep=" "), npt.Array1D[Any])

assert_type(np.fromfile("test.txt", sep=" "), npt.Array1D[np.float64])
assert_type(np.fromfile("test.txt", dtype=np.int64, sep=" "), npt.Array1D[np.int64])
assert_type(np.fromfile("test.txt", dtype="c16", sep=" "), npt.Array1D[Any])
with open("test.txt") as f:
    assert_type(np.fromfile(f, sep=" "), npt.Array1D[np.float64])
    assert_type(np.fromfile(b"test.txt", sep=" "), npt.Array1D[np.float64])
    assert_type(np.fromfile(Path("test.txt"), sep=" "), npt.Array1D[np.float64])

assert_type(np.fromiter("12345", np.float32), npt.Array1D[np.float32])
assert_type(np.fromiter("12345", np.float64), npt.Array1D[np.float64])
assert_type(np.fromiter("12345", bool), npt.Array1D[np.bool])
assert_type(np.fromiter("12345", int), npt.Array1D[np.int_ | Any])
assert_type(np.fromiter("12345", float), npt.Array1D[np.float64 | Any])
assert_type(np.fromiter("12345", complex), npt.Array1D[np.complex128 | Any])
assert_type(np.fromiter("12345", None), npt.Array1D[np.float64])
assert_type(np.fromiter("12345", object), npt.Array1D[Any])

assert_type(np.frombuffer(A), npt.Array1D[np.float64])
assert_type(np.frombuffer(A, dtype=np.int64), npt.Array1D[np.int64])
assert_type(np.frombuffer(A, dtype="c16"), npt.Array1D[Any])

assert_type(np.from_dlpack(i8), npt.Array0D[np.int64])
assert_type(np.from_dlpack(A), npt.NDArray[np.float64])
assert_type(np.from_dlpack(B), npt.NDArray[np.float64])
assert_type(np.from_dlpack(_f32_2d), npt.Array2D[np.float32])
assert_type(np.from_dlpack(_dlpack_obj), npt.NDArray[np.number | np.bool])

_x_bool: bool
_x_int: int
_x_float: float
_x_timedelta: np.timedelta64[int]
_x_datetime: np.datetime64[int]

assert_type(np.arange(False, True), npt.Array1D[np.int_])
assert_type(np.arange(10), npt.Array1D[np.int_])
assert_type(np.arange(0, 10, step=2), npt.Array1D[np.int_])
assert_type(np.arange(10.0), npt.Array1D[np.float64 | Any])
assert_type(np.arange(0, stop=10.0), npt.Array1D[np.float64 | Any])
assert_type(np.arange(_x_timedelta), npt.Array1D[np.timedelta64])
assert_type(np.arange(0, _x_timedelta), npt.Array1D[np.timedelta64])
assert_type(np.arange(_x_datetime, _x_datetime), npt.Array1D[np.datetime64])
assert_type(np.arange(10, dtype=np.float64), npt.Array1D[np.float64])
assert_type(np.arange(0, 10, step=2, dtype=np.int16), npt.Array1D[np.int16])
assert_type(np.arange(10, dtype=int), npt.Array1D[np.int_])
assert_type(np.arange(0, 10, dtype="f8"), npt.Array1D[Any])
# https://github.com/numpy/numpy/issues/30628
assert_type(np.arange("2025-12-20", "2025-12-23", dtype="datetime64[D]"), npt.Array1D[np.datetime64])

assert_type(np.require(B, requirements=["C", "OWNDATA"]), SubClass[np.float64])
assert_type(np.require(B, dtype=int), npt.NDArray[Any])
assert_type(np.require(C), npt.NDArray[Any])
assert_type(np.require(C, dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.require(_f32_2d, dtype=np.int64), npt.Array2D[np.int64])
assert_type(np.require(_f32_2d, requirements={"F", "E"}), npt.Array2D[np.float32])

assert_type(np.linspace(0, 10), npt.Array1D[np.float64])
assert_type(np.linspace(0, 10j), npt.Array1D[np.complex128 | Any])
assert_type(np.linspace(0, 10, dtype=np.int64), npt.Array1D[np.int64])
assert_type(np.linspace(0, 10, dtype=int), npt.NDArray[Any])
assert_type(np.linspace(0, 10, retstep=True), tuple[npt.Array1D[np.float64], np.float64])
assert_type(np.linspace(0j, 10, retstep=True), tuple[npt.Array1D[np.complex128 | Any], np.complex128 | Any])
assert_type(np.linspace(0, 10, retstep=True, dtype=np.int64), tuple[npt.Array1D[np.int64], np.int64])
assert_type(np.linspace(0j, 10, retstep=True, dtype=int), tuple[npt.NDArray[Any], Any])

assert_type(np.logspace(0, 10), npt.Array1D[np.float64])
assert_type(np.logspace(0, 10j), npt.Array1D[np.complex128 | Any])
assert_type(np.logspace(0, 10, dtype=np.int64), npt.Array1D[np.int64])
assert_type(np.logspace(0, 10, dtype=int), npt.NDArray[Any])

assert_type(np.geomspace(0, 10), npt.Array1D[np.float64])
assert_type(np.geomspace(0, 10j), npt.Array1D[np.complex128 | Any])
assert_type(np.geomspace(0, 10, dtype=np.int64), npt.Array1D[np.int64])
assert_type(np.geomspace(0, 10, dtype=int), npt.NDArray[Any])

assert_type(np.zeros_like(A), npt.NDArray[np.float64])
assert_type(np.zeros_like(C), npt.NDArray[Any])
assert_type(np.zeros_like(A, dtype=float), npt.NDArray[Any])
assert_type(np.zeros_like(B), SubClass[np.float64])
assert_type(np.zeros_like(B, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.zeros_like(_f32_1d), npt.Array1D[np.float32])
assert_type(np.zeros_like(_f32_1d, dtype=np.int64), npt.Array1D[np.int64])
assert_type(np.zeros_like(_f32_1d, dtype=int), npt.Array1D[Any])
assert_type(np.zeros_like(_f32_1d, shape=_shape_2d), npt.Array2D[np.float32])
assert_type(np.zeros_like(_f32_1d, shape=_shape_like), npt.NDArray[np.float32])
assert_type(np.zeros_like(_obj_str_1d), npt.Array1D[np.object_[int]])
assert_type(np.zeros_like(_obj_str_1d, shape=_shape_2d), npt.Array2D[np.object_[int]])

assert_type(np.ones_like(A), npt.NDArray[np.float64])
assert_type(np.ones_like(C), npt.NDArray[Any])
assert_type(np.ones_like(A, dtype=float), npt.NDArray[Any])
assert_type(np.ones_like(B), SubClass[np.float64])
assert_type(np.ones_like(B, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.ones_like(_f32_1d), npt.Array1D[np.float32])
assert_type(np.ones_like(_f32_1d, dtype=np.int64), npt.Array1D[np.int64])
assert_type(np.ones_like(_f32_1d, dtype=int), npt.Array1D[Any])
assert_type(np.ones_like(_f32_1d, shape=_shape_2d), npt.Array2D[np.float32])
assert_type(np.ones_like(_f32_1d, shape=_shape_like), npt.NDArray[np.float32])
assert_type(np.ones_like(_obj_str_1d), npt.Array1D[np.object_[int]])
assert_type(np.ones_like(_obj_str_1d, shape=_shape_2d), npt.Array2D[np.object_[int]])

assert_type(np.empty_like(A), npt.NDArray[np.float64])
assert_type(np.empty_like(C), npt.NDArray[Any])
assert_type(np.empty_like(A, dtype=float), npt.NDArray[Any])
assert_type(np.empty_like(B), SubClass[np.float64])
assert_type(np.empty_like(B, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.empty_like(_f32_1d), npt.Array1D[np.float32])
assert_type(np.empty_like(_f32_1d, dtype=np.int64), npt.Array1D[np.int64])
assert_type(np.empty_like(_f32_1d, dtype=int), npt.Array1D[Any])
assert_type(np.empty_like(_f32_1d, shape=_shape_2d), npt.Array2D[np.float32])
assert_type(np.empty_like(_f32_1d, shape=_shape_like), npt.NDArray[np.float32])
assert_type(np.empty_like(_obj_str_1d), npt.Array1D[np.object_[Any | None]])
assert_type(np.empty_like(_obj_str_1d, shape=_shape_2d), npt.Array2D[np.object_[Any | None]])

assert_type(np.full_like(A, i8), npt.NDArray[np.float64])
assert_type(np.full_like(C, i8), npt.NDArray[Any])
assert_type(np.full_like(A, i8, dtype=int), npt.NDArray[Any])
assert_type(np.full_like(B, i8), SubClass[np.float64])
assert_type(np.full_like(B, i8, dtype=np.int64), npt.NDArray[np.int64])
assert_type(np.full_like(_f32_1d, i8), npt.Array1D[np.float32])
assert_type(np.full_like(_f32_1d, i8, dtype=np.int64), npt.Array1D[np.int64])
assert_type(np.full_like(_f32_1d, i8, dtype=int), npt.Array1D[Any])
assert_type(np.full_like(_f32_1d, i8, shape=_shape_2d), npt.Array2D[np.float32])
assert_type(np.full_like(_f32_1d, i8, shape=_shape_like), npt.NDArray[np.float32])
assert_type(np.full_like(_obj_str_1d, i8), npt.Array1D[np.object_[Any]])
assert_type(np.full_like(_obj_str_1d, i8, shape=_shape_2d), npt.Array2D[np.object_[Any]])

_size: int
_shape_0d: tuple[()]
_shape_1d: tuple[int]
_shape_2d: tuple[int, int]
_shape_nd: tuple[int, ...]
_shape_like: list[int]

assert_type(np.ones(_shape_0d), np.ndarray[tuple[()], np.dtype[np.float64]])
assert_type(np.ones(_size), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.ones(_shape_2d), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.ones(_shape_nd), np.ndarray[tuple[Any, ...], np.dtype[np.float64]])
assert_type(np.ones(_shape_1d, dtype=np.int64), np.ndarray[tuple[int], np.dtype[np.int64]])
assert_type(np.ones(_shape_like), npt.NDArray[np.float64])
assert_type(
    np.ones(_shape_like, dtype=np.dtypes.Int64DType()),
    np.ndarray[tuple[Any, ...], np.dtypes.Int64DType],
)
assert_type(np.ones(_shape_like, dtype=int), npt.NDArray[Any])
assert_type(np.ones(_size, dtype=bool), npt.Array1D[np.bool])
assert_type(np.ones(_shape_2d, dtype=bool), npt.Array2D[np.bool])
assert_type(np.ones(_shape_like, dtype=bool), npt.NDArray[np.bool])
assert_type(np.ones(mixed_shape), npt.NDArray[np.float64])

assert_type(np.full(_size, i8), np.ndarray[tuple[int], np.dtype[np.int64]])
assert_type(np.full(_shape_2d, i8), np.ndarray[tuple[int, int], np.dtype[np.int64]])
assert_type(np.full(_shape_like, i8), npt.NDArray[np.int64])
assert_type(np.full(_shape_like, 42), npt.NDArray[Any])
assert_type(np.full(_size, i8, dtype=np.float64), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(np.full(_size, i8, dtype=float), np.ndarray[tuple[int], np.dtype])
assert_type(np.full(_shape_like, 42, dtype=float), npt.NDArray[Any])
assert_type(np.full(_shape_0d, i8, dtype=object), np.ndarray[tuple[()], np.dtype])

assert_type(np.indices([1, 2, 3]), npt.NDArray[np.int_])
assert_type(np.indices([1, 2, 3], sparse=True), tuple[npt.NDArray[np.int_], ...])

assert_type(np.fromfunction(_func_1d_i8, (3,), dtype=np.int8), npt.Array1D[np.int8])
assert_type(np.fromfunction(_func_1d_i64, (3,), dtype=int), npt.Array1D[np.int_])
assert_type(np.fromfunction(_func_1d_f64, (3,)), npt.Array1D[np.float64])
assert_type(np.fromfunction(_func_1d, (3,), dtype="i"), npt.Array1D[Any])
assert_type(np.fromfunction(_func_2d_i8, (3, 3), dtype=np.int8), npt.Array2D[np.int8])
assert_type(np.fromfunction(_func_2d_i64, (3, 3), dtype=int), npt.Array2D[np.int_])
assert_type(np.fromfunction(_func_2d_f64, (3, 3)), npt.Array2D[np.float64])
assert_type(np.fromfunction(_func_2d, (3, 3), dtype="i"), npt.Array2D[Any])
assert_type(np.fromfunction(_func_3d_i8, (2, 3, 4), dtype=np.int8), npt.Array3D[np.int8])
assert_type(np.fromfunction(_func_3d_i64, (2, 3, 4), dtype=int), npt.Array3D[np.int_])
assert_type(np.fromfunction(_func_3d_f64, (2, 3, 4)), npt.Array3D[np.float64])
assert_type(np.fromfunction(_func_3d, (2, 3, 4), dtype="i"), npt.Array3D[Any])
assert_type(np.fromfunction(_func_nd, (2, 3, 4, 5)), SubClass[np.float64])

assert_type(np.identity(3), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(np.identity(3, dtype=np.int8), np.ndarray[tuple[int, int], np.dtype[np.int8]])
assert_type(np.identity(3, dtype=bool), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.identity(3, dtype="bool"), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.identity(3, dtype="b1"), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.identity(3, dtype="?"), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(np.identity(3, dtype=int), np.ndarray[tuple[int, int], np.dtype[np.int_ | Any]])
assert_type(np.identity(3, dtype="int"), np.ndarray[tuple[int, int], np.dtype[np.int_ | Any]])
assert_type(np.identity(3, dtype="n"), np.ndarray[tuple[int, int], np.dtype[np.int_ | Any]])
assert_type(np.identity(3, dtype=float), np.ndarray[tuple[int, int], np.dtype[np.float64 | Any]])
assert_type(np.identity(3, dtype="float"), np.ndarray[tuple[int, int], np.dtype[np.float64 | Any]])
assert_type(np.identity(3, dtype="f8"), np.ndarray[tuple[int, int], np.dtype[np.float64 | Any]])
assert_type(np.identity(3, dtype="d"), np.ndarray[tuple[int, int], np.dtype[np.float64 | Any]])
assert_type(np.identity(3, dtype=complex), np.ndarray[tuple[int, int], np.dtype[np.complex128 | Any]])
assert_type(np.identity(3, dtype="complex"), np.ndarray[tuple[int, int], np.dtype[np.complex128 | Any]])
assert_type(np.identity(3, dtype="c16"), np.ndarray[tuple[int, int], np.dtype[np.complex128 | Any]])
assert_type(np.identity(3, dtype="D"), np.ndarray[tuple[int, int], np.dtype[np.complex128 | Any]])

assert_type(np.atleast_1d(_f32_0d), npt.Array1D[np.float32])
assert_type(np.atleast_1d(_f32_1d), npt.Array1D[np.float32])
assert_type(np.atleast_1d(A), npt.NDArray[np.float64])
assert_type(np.atleast_1d(_x_bool), npt.Array1D[np.bool])
assert_type(np.atleast_1d(_x_int), npt.Array1D[np.int_ | Any])
assert_type(np.atleast_1d(_x_float), npt.Array1D[np.float64 | Any])
assert_type(np.atleast_1d(_py_i_1d), npt.Array1D[np.int_])
assert_type(np.atleast_1d(_py_f_1d), npt.Array1D[np.float64])
assert_type(np.atleast_1d(_py_c_1d), npt.Array1D[np.complex128])
assert_type(np.atleast_1d(_py_f_2d), npt.NDArray[Any])
assert_type(
    np.atleast_1d(_f32_1d, _f32_2d),
    tuple[npt.Array1D[np.float32], npt.Array2D[np.float32]],
)
assert_type(
    np.atleast_1d(_f32_0d, A),
    tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]],
)
assert_type(
    np.atleast_1d(A, C),
    tuple[npt.NDArray[Any], npt.NDArray[Any]],
)
assert_type(
    np.atleast_1d(A, A, A),
    tuple[npt.NDArray[np.float64], ...],
)
assert_type(
    np.atleast_1d(C, C, C),
    tuple[npt.NDArray[Any], ...],
)

assert_type(np.atleast_2d(_f32_1d), npt.Array2D[np.float32])
assert_type(np.atleast_2d(_f32_2d), npt.Array2D[np.float32])
assert_type(np.atleast_2d(A), npt.NDArray[np.float64])
assert_type(np.atleast_2d(_x_bool), npt.Array2D[np.bool])
assert_type(np.atleast_2d(_x_int), npt.Array2D[np.int_ | Any])
assert_type(np.atleast_2d(_x_float), npt.Array2D[np.float64 | Any])
assert_type(np.atleast_2d(_py_i_1d), npt.Array2D[np.int_])
assert_type(np.atleast_2d(_py_f_2d), npt.Array2D[np.float64])
assert_type(np.atleast_2d(_py_c_1d), npt.Array2D[np.complex128])
assert_type(np.atleast_2d(_py_f_3d), npt.NDArray[Any])
assert_type(
    np.atleast_2d(_f32_2d, _f32_3d),
    tuple[npt.Array2D[np.float32], npt.Array3D[np.float32]],
)
assert_type(
    np.atleast_2d(_f32_0d, A),
    tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]],
)
assert_type(
    np.atleast_2d(A, C),
    tuple[npt.NDArray[Any], npt.NDArray[Any]],
)
assert_type(
    np.atleast_2d(A, A, A),
    tuple[npt.NDArray[np.float64], ...],
)
assert_type(
    np.atleast_2d(C, C, C),
    tuple[npt.NDArray[Any], ...],
)

assert_type(np.atleast_3d(_f32_2d), npt.Array3D[np.float32])
assert_type(np.atleast_3d(_f32_3d), npt.Array3D[np.float32])
assert_type(np.atleast_3d(A), npt.NDArray[np.float64])
assert_type(np.atleast_3d(_x_bool), npt.Array3D[np.bool])
assert_type(np.atleast_3d(_x_int), npt.Array3D[np.int_ | Any])
assert_type(np.atleast_3d(_x_float), npt.Array3D[np.float64 | Any])
assert_type(np.atleast_3d(_py_i_1d), npt.Array3D[np.int_])
assert_type(np.atleast_3d(_py_f_3d), npt.Array3D[np.float64])
assert_type(np.atleast_3d(_py_c_2d), npt.Array3D[np.complex128])
assert_type(
    np.atleast_3d(_f32_3d, B),
    tuple[npt.Array3D[np.float32], SubClass[np.float64]],
)
assert_type(
    np.atleast_3d(_f32_0d, A),
    tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]],
)
assert_type(
    np.atleast_3d(A, C),
    tuple[npt.NDArray[Any], npt.NDArray[Any]],
)
assert_type(
    np.atleast_3d(A, A, A),
    tuple[npt.NDArray[np.float64], ...],
)
assert_type(
    np.atleast_3d(C, C, C),
    tuple[npt.NDArray[Any], ...],
)

assert_type(np.vstack([A, A]), npt.NDArray[np.float64])
assert_type(np.vstack([A, A], dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.vstack([A, C]), npt.NDArray[Any])
assert_type(np.vstack([C, C]), npt.NDArray[Any])
assert_type(np.vstack([_f32_0d, _f32_0d]), npt.Array2D[np.float32])
assert_type(np.vstack([_f32_1d, _f32_1d]), npt.Array2D[np.float32])
assert_type(np.vstack([_f32_2d, _f32_2d]), npt.Array2D[np.float32])
assert_type(np.vstack([_f32_3d, _f32_3d]), npt.Array3D[np.float32])
assert_type(np.vstack([_f32_3d, _f32_3d], dtype=np.int8), npt.Array3D[np.int8])

assert_type(np.hstack([A, A]), npt.NDArray[np.float64])
assert_type(np.hstack([A, A], dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.hstack([A, C]), npt.NDArray[Any])
assert_type(np.hstack([C, C]), npt.NDArray[Any])
assert_type(np.hstack([_f32_0d, _f32_0d]), npt.Array1D[np.float32])
assert_type(np.hstack([_f32_1d, _f32_1d]), npt.Array1D[np.float32])
assert_type(np.hstack([_f32_2d, _f32_2d]), npt.Array2D[np.float32])
assert_type(np.hstack([_f32_3d, _f32_3d]), npt.Array3D[np.float32])
assert_type(np.hstack([_f32_3d, _f32_3d], dtype=np.int8), npt.Array3D[np.int8])

assert_type(np.stack([A, A]), npt.NDArray[np.float64])
assert_type(np.stack([A, A], dtype=np.float32), npt.NDArray[np.float32])
assert_type(np.stack([A, C]), npt.NDArray[Any])
assert_type(np.stack([C, C]), npt.NDArray[Any])
assert_type(np.stack([A, A], axis=0), npt.NDArray[np.float64])
assert_type(np.stack([A, A], out=B), SubClass[np.float64])
assert_type(np.stack([_f32_0d, _f32_0d]), npt.Array1D[np.float32])
assert_type(np.stack([_f32_1d, _f32_1d]), npt.Array2D[np.float32])
assert_type(np.stack([_f32_2d, _f32_2d]), npt.Array3D[np.float32])
assert_type(np.stack([_f32_3d, _f32_3d]), npt.Array4D[np.float32])
assert_type(np.stack([_f32_2d, _f32_2d], axis=-1), npt.Array3D[np.float32])
assert_type(np.stack([_f32_2d, _f32_2d], dtype=np.int8), npt.Array3D[np.int8])
assert_type(np.stack([_f32_2d, _f32_2d], dtype="i1"), npt.Array3D[Any])

assert_type(np.block(_f32_2d), npt.Array2D[np.float32])
assert_type(np.block([[A, A], [A, A]]), npt.NDArray[np.float64])
assert_type(np.block([_f32_1d, _f32_1d]), npt.Array1D[np.float32])
assert_type(np.block([True, False]), npt.Array1D[np.bool])
assert_type(np.block([1, 2]), npt.Array1D[np.int_])
assert_type(np.block([1.0, 2]), npt.Array1D[np.float64])
assert_type(np.block([1j, 2]), npt.Array1D[np.complex128])
assert_type(np.block([_f32_1d, 1]), npt.Array1D[Any])
assert_type(np.block([[_f32_2d, _f32_2d], [_f32_2d, _f32_2d]]), npt.Array2D[np.float32])
assert_type(np.block([[True]]), npt.Array2D[np.bool])
assert_type(np.block([[1, 2], [3, 4]]), npt.Array2D[np.int_])
assert_type(np.block([[1.0, 2], [3, 4]]), npt.Array2D[np.float64])
assert_type(np.block([[1j, 2], [3, 4]]), npt.Array2D[np.complex128])
assert_type(np.block([[_f32_2d, _f32_2d], [_f32_1d, 1]]), npt.Array2D[Any])
assert_type(np.block([[[_f32_1d]], [[_f32_1d]]]), npt.Array3D[np.float32])
assert_type(np.block([[[True]]]), npt.Array3D[np.bool])
assert_type(np.block([[[1]]]), npt.Array3D[np.int_])
assert_type(np.block([[[1.0]]]), npt.Array3D[np.float64])
assert_type(np.block([[[1j]]]), npt.Array3D[np.complex128])
assert_type(np.block([[[_f32_1d, 1]]]), npt.Array3D[Any])
assert_type(np.block(_f32_4d_list), npt.NDArray[np.float32])
assert_type(np.block(["a", "b"]), npt.NDArray[Any])

from collections.abc import Buffer

def create_array(obj: npt.ArrayLike) -> npt.NDArray[Any]: ...

buffer: Buffer
assert_type(create_array(buffer), npt.NDArray[Any])
