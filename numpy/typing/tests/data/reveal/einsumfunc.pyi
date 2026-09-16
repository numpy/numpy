from typing import Any, assert_type

import numpy as np
import numpy.typing as npt

type _Array0D[ST: np.generic] = np.ndarray[tuple[()], np.dtype[ST]]
type _Array1D[ST: np.generic] = np.ndarray[tuple[int], np.dtype[ST]]
type _Array2D[ST: np.generic] = np.ndarray[tuple[int, int], np.dtype[ST]]
type _Array3D[ST: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[ST]]
type _Array4D[ST: np.generic] = np.ndarray[tuple[int, int, int, int], np.dtype[ST]]

_py_U: str
_py_b_1d: list[bool]
_py_i_1d: list[int]
_py_f_1d: list[float]
_py_c_1d: list[complex]
_py_U_1d: list[str]

_u32_1d_list: list[np.uint32]
_f64_1d: _Array1D[np.float64]
_f64_2d: _Array2D[np.float64]
_f64_3d: _Array3D[np.float64]
_f64_4d: _Array4D[np.float64]
_f64_nd: npt.NDArray[np.float64]
_c128_nd: npt.NDArray[np.complex128]

###
# einsum

# 0d

assert_type(np.einsum("i->", _f64_1d), np.float64)
assert_type(np.einsum("ii", _f64_2d), np.float64)
assert_type(np.einsum("i,i", _f64_1d, _f64_1d), np.float64)
assert_type(np.einsum("ij,ij->", _f64_nd, _f64_nd), np.float64)
assert_type(np.einsum("i,i", _f64_1d, _f64_1d, optimize=True), np.float64 | _Array0D[np.float64])

# 1d

assert_type(np.einsum("ii->i", _f64_2d), _Array1D[np.float64])
assert_type(np.einsum("ij,j", _f64_2d, _f64_1d), _Array1D[np.float64])
assert_type(np.einsum("ij,ij->i", _f64_nd, _f64_nd), _Array1D[np.float64])
assert_type(np.einsum("i,i->i", _u32_1d_list, _u32_1d_list), _Array1D[np.uint32])

# 2d

assert_type(np.einsum("ij->ji", _f64_2d), _Array2D[np.float64])
assert_type(np.einsum("i,j->ij", _f64_1d, _f64_1d), _Array2D[np.float64])
assert_type(np.einsum("ij,jk->ik", _f64_2d, _f64_2d), _Array2D[np.float64])
assert_type(np.einsum("ij,jk", _f64_nd, _f64_nd, optimize=True), _Array2D[np.float64])
assert_type(np.einsum("ijk,ij->ik", _f64_3d, _f64_2d), _Array2D[np.float64])

# 3d

assert_type(np.einsum("bij,bjk->bik", _f64_3d, _f64_3d), _Array3D[np.float64])
assert_type(np.einsum("i,j,k->ijk", _f64_1d, _f64_1d, _f64_1d), _Array3D[np.float64])

# 4d

assert_type(np.einsum("abcd,cdjk->abjk", _f64_4d, _f64_4d), _Array4D[np.float64])

# ?d

assert_type(np.einsum(_py_U, _f64_nd, _f64_nd), Any)
assert_type(np.einsum("ij,jk->ik", _f64_nd, _f64_nd, dtype=np.float32), Any)
assert_type(np.einsum("ij,jk->ik", _f64_nd, _f64_nd, casting="unsafe"), Any)
assert_type(np.einsum("ij,jk->ik", _f64_nd, _f64_nd, out=_f64_nd), npt.NDArray[np.float64])

assert_type(np.einsum("i,i->i", _py_b_1d, _py_b_1d), Any)
assert_type(np.einsum("i,i->i", _py_i_1d, _py_i_1d), Any)
assert_type(np.einsum("i,i->i", _py_f_1d, _py_f_1d), Any)
assert_type(np.einsum("i,i->i", _py_c_1d, _py_c_1d), Any)
assert_type(np.einsum("i,i->i", _py_b_1d, _py_i_1d), Any)
assert_type(np.einsum("i,i,i,i->i", _py_b_1d, _u32_1d_list, _py_i_1d, _py_c_1d), Any)
assert_type(np.einsum("i,i->i", _py_f_1d, _py_f_1d, dtype="c16"), Any)
assert_type(np.einsum("i,i->i", _py_c_1d, _py_c_1d, out=_f64_nd), npt.NDArray[np.float64])
assert_type(np.einsum("i,i->i", _py_U_1d, _py_U_1d, dtype=bool, casting="unsafe"), Any)
assert_type(np.einsum("i,i->i", _py_U_1d, _py_U_1d, dtype=bool, casting="unsafe", out=_f64_nd), npt.NDArray[np.float64])

# sublist format

assert_type(np.einsum([[1, 1], [1, 1]], _py_i_1d, _py_i_1d), Any)
assert_type(np.einsum(_f64_nd, [0, 1], _f64_nd, [1, 0], [0]), Any)
assert_type(np.einsum(_c128_nd, [0, 1], _c128_nd, [1, 0], [0]), Any)

###
#  einsum_path

assert_type(np.einsum_path("i,i->i", _py_b_1d, _py_b_1d), tuple[list[Any], str])
assert_type(np.einsum_path("i,i->i", _py_i_1d, _py_i_1d), tuple[list[Any], str])
assert_type(np.einsum_path("i,i->i", _py_f_1d, _py_f_1d), tuple[list[Any], str])
assert_type(np.einsum_path("i,i->i", _py_c_1d, _py_c_1d), tuple[list[Any], str])
assert_type(np.einsum_path("i,i->i", _py_b_1d, _py_i_1d), tuple[list[Any], str])
assert_type(np.einsum_path("i,i,i,i->i", _py_b_1d, _u32_1d_list, _py_i_1d, _py_c_1d), tuple[list[Any], str])
assert_type(np.einsum_path([[1, 1], [1, 1]], _py_i_1d, _py_i_1d), tuple[list[Any], str])
