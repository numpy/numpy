from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any, assert_type

import numpy as np
import numpy.polynomial.polyutils as pu
import numpy.typing as npt
from numpy.polynomial._polytypes import _Tuple2

num_int: int
num_float: float
num_complex: complex
# will result in an `object_` dtype
num_object: Decimal | Fraction

sct_int: np.int_
sct_float: np.float64
sct_complex: np.complex128
sct_object: np.object_[int]  # doesn't exist at runtime

arr_int: npt.NDArray[np.int_]
arr_float: npt.NDArray[np.float64]
arr_complex: npt.NDArray[np.complex128]
arr_object: npt.NDArray[np.object_[int]]
arr_float_2d: npt.Array2D[np.float64]
arr_complex_2d: npt.Array2D[np.complex128]
arr_object_2d: npt.Array2D[np.object_[int]]

seq_num_int: Sequence[int]
seq_num_float: Sequence[float]
seq_num_complex: Sequence[complex]
seq_num_object: Sequence[Decimal | Fraction]
list_num_complex: list[complex]

seq_sct_int: Sequence[np.int_]
seq_sct_float: Sequence[np.float64]
seq_sct_complex: Sequence[np.complex128]
seq_sct_object: Sequence[np.object_]

seq_arr_int: Sequence[npt.NDArray[np.int_]]
seq_arr_float: Sequence[npt.NDArray[np.float64]]
seq_arr_complex: Sequence[npt.NDArray[np.complex128]]
seq_arr_object: Sequence[npt.NDArray[np.object_]]
seq_arr_object_int: Sequence[npt.NDArray[np.object_[int]]]

seq_seq_num_int: Sequence[Sequence[int]]
seq_seq_num_float: Sequence[Sequence[float]]
seq_seq_num_complex: Sequence[Sequence[complex]]
seq_seq_num_object: Sequence[Sequence[Decimal | Fraction]]

seq_seq_sct_int: Sequence[Sequence[np.int_]]
seq_seq_sct_float: Sequence[Sequence[np.float64]]
seq_seq_sct_complex: Sequence[Sequence[np.complex128]]
seq_seq_sct_object: Sequence[Sequence[np.object_]]  # doesn't exist at runtime

# as_series

assert_type(pu.as_series(arr_int), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(arr_float), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(arr_complex), list[npt.Array1D[np.complexfloating]])
assert_type(pu.as_series(arr_object), list[npt.Array1D[np.object_]])

assert_type(pu.as_series(seq_num_int), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(seq_num_float), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(seq_num_complex), list[npt.Array1D[np.complexfloating]])
assert_type(pu.as_series(seq_num_object), list[npt.Array1D[np.object_]])

assert_type(pu.as_series(seq_sct_int), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(seq_sct_float), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(seq_sct_complex), list[npt.Array1D[np.complexfloating]])
assert_type(pu.as_series(seq_sct_object), list[npt.Array1D[np.object_]])

assert_type(pu.as_series(seq_arr_int), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(seq_arr_float), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(seq_arr_complex), list[npt.Array1D[np.complexfloating]])
assert_type(pu.as_series(seq_arr_object), list[npt.Array1D[np.object_]])

assert_type(pu.as_series(seq_seq_num_int), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(seq_seq_num_float), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(seq_seq_num_complex), list[npt.Array1D[np.complexfloating]])
assert_type(pu.as_series(seq_seq_num_object), list[npt.Array1D[np.object_]])

assert_type(pu.as_series(seq_seq_sct_int), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(seq_seq_sct_float), list[npt.Array1D[np.floating]])
assert_type(pu.as_series(seq_seq_sct_complex), list[npt.Array1D[np.complexfloating]])
assert_type(pu.as_series(seq_seq_sct_object), list[npt.Array1D[np.object_]])

# trimcoef

assert_type(pu.trimcoef(num_int), npt.Array1D[np.floating])
assert_type(pu.trimcoef(num_float), npt.Array1D[np.floating])
assert_type(pu.trimcoef(num_complex), npt.Array1D[np.complexfloating])
assert_type(pu.trimcoef(num_object), npt.Array1D[np.object_])
assert_type(pu.trimcoef(num_object), npt.Array1D[np.object_])

assert_type(pu.trimcoef(sct_int), npt.Array1D[np.floating])
assert_type(pu.trimcoef(sct_float), npt.Array1D[np.floating])
assert_type(pu.trimcoef(sct_complex), npt.Array1D[np.complexfloating])
assert_type(pu.trimcoef(sct_object), npt.Array1D[np.object_])

assert_type(pu.trimcoef(arr_int), npt.Array1D[np.floating])
assert_type(pu.trimcoef(arr_float), npt.Array1D[np.floating])
assert_type(pu.trimcoef(arr_complex), npt.Array1D[np.complexfloating])
assert_type(pu.trimcoef(arr_object), npt.Array1D[np.object_])

assert_type(pu.trimcoef(seq_num_int), npt.Array1D[np.floating])
assert_type(pu.trimcoef(seq_num_float), npt.Array1D[np.floating])
assert_type(pu.trimcoef(seq_num_complex), npt.Array1D[np.complexfloating])
assert_type(pu.trimcoef(seq_num_object), npt.Array1D[np.object_])

assert_type(pu.trimcoef(seq_sct_int), npt.Array1D[np.floating])
assert_type(pu.trimcoef(seq_sct_float), npt.Array1D[np.floating])
assert_type(pu.trimcoef(seq_sct_complex), npt.Array1D[np.complexfloating])
assert_type(pu.trimcoef(seq_sct_object), npt.Array1D[np.object_])

# getdomain

assert_type(pu.getdomain(num_int), npt.Array1D[np.float64])
assert_type(pu.getdomain(num_float), npt.Array1D[np.float64])
assert_type(pu.getdomain(num_complex), npt.Array1D[np.complex128])
assert_type(pu.getdomain(num_object), npt.Array1D[np.object_])
assert_type(pu.getdomain(num_object), npt.Array1D[np.object_])

assert_type(pu.getdomain(sct_int), npt.Array1D[np.float64])
assert_type(pu.getdomain(sct_float), npt.Array1D[np.float64])
assert_type(pu.getdomain(sct_complex), npt.Array1D[np.complex128])
assert_type(pu.getdomain(sct_object), npt.Array1D[np.object_])

assert_type(pu.getdomain(arr_int), npt.Array1D[np.float64])
assert_type(pu.getdomain(arr_float), npt.Array1D[np.float64])
assert_type(pu.getdomain(arr_complex), npt.Array1D[np.complex128])
assert_type(pu.getdomain(arr_object), npt.Array1D[np.object_])

assert_type(pu.getdomain(seq_num_int), npt.Array1D[np.float64])
assert_type(pu.getdomain(seq_num_float), npt.Array1D[np.float64])
assert_type(pu.getdomain(seq_num_complex), npt.Array1D[np.complex128])
assert_type(pu.getdomain(seq_num_object), npt.Array1D[np.object_])

assert_type(pu.getdomain(seq_sct_int), npt.Array1D[np.float64])
assert_type(pu.getdomain(seq_sct_float), npt.Array1D[np.float64])
assert_type(pu.getdomain(seq_sct_complex), npt.Array1D[np.complex128])
assert_type(pu.getdomain(seq_sct_object), npt.Array1D[np.object_])

# mapparms

assert_type(pu.mapparms(seq_num_int, seq_num_int), _Tuple2[float])
assert_type(pu.mapparms(seq_num_int, seq_num_float), _Tuple2[float])
assert_type(pu.mapparms(seq_num_float, seq_num_float), _Tuple2[float])
assert_type(pu.mapparms(seq_num_float, seq_num_complex), _Tuple2[complex])
assert_type(pu.mapparms(seq_num_complex, seq_num_complex), _Tuple2[complex])
assert_type(pu.mapparms(seq_num_complex, seq_num_object), _Tuple2[object])
assert_type(pu.mapparms(seq_num_object, seq_num_object), _Tuple2[object])

assert_type(pu.mapparms(seq_sct_int, seq_sct_int), _Tuple2[np.floating])
assert_type(pu.mapparms(seq_sct_int, seq_sct_float), _Tuple2[np.floating])
assert_type(pu.mapparms(seq_sct_float, seq_sct_float), _Tuple2[float])
assert_type(pu.mapparms(seq_sct_float, seq_sct_complex), _Tuple2[complex])
assert_type(pu.mapparms(seq_sct_complex, seq_sct_complex), _Tuple2[complex])
assert_type(pu.mapparms(seq_sct_complex, seq_sct_object), _Tuple2[object])
assert_type(pu.mapparms(seq_sct_object, seq_sct_object), _Tuple2[object])

assert_type(pu.mapparms(arr_int, arr_int), _Tuple2[np.floating])
assert_type(pu.mapparms(arr_int, arr_float), _Tuple2[np.floating])
assert_type(pu.mapparms(arr_float, arr_float), _Tuple2[np.floating])
assert_type(pu.mapparms(arr_float, arr_complex), _Tuple2[np.complexfloating])
assert_type(pu.mapparms(arr_complex, arr_complex), _Tuple2[np.complexfloating])
assert_type(pu.mapparms(arr_complex, arr_object), _Tuple2[object])
assert_type(pu.mapparms(arr_object, arr_object), _Tuple2[object])

# mapdomain

assert_type(pu.mapdomain(num_int, seq_num_int, seq_num_int), float)
assert_type(pu.mapdomain(num_int, seq_num_int, seq_num_float), float)
assert_type(pu.mapdomain(num_int, seq_num_float, seq_num_float), float)
assert_type(pu.mapdomain(num_float, seq_num_float, seq_num_float), float)
assert_type(pu.mapdomain(num_float, seq_num_float, seq_num_complex), complex)
assert_type(pu.mapdomain(num_float, seq_num_complex, seq_num_complex), complex)
assert_type(pu.mapdomain(num_complex, seq_num_complex, seq_num_complex), complex)
assert_type(pu.mapdomain(num_complex, seq_num_complex, seq_num_object), complex)
assert_type(pu.mapdomain(num_complex, seq_num_object, seq_num_object), complex)
assert_type(pu.mapdomain(num_object, seq_num_object, seq_num_object), Decimal | Fraction)

assert_type(pu.mapdomain(seq_num_int, seq_num_int, seq_num_int), npt.Array1D[np.float64])
assert_type(pu.mapdomain(seq_num_int, seq_num_int, seq_num_float), npt.Array1D[np.float64])
assert_type(pu.mapdomain(seq_num_int, seq_num_float, seq_num_float), npt.Array1D[np.float64])
assert_type(pu.mapdomain(seq_num_float, seq_num_float, seq_num_float), npt.Array1D[np.float64])
assert_type(pu.mapdomain(seq_num_float, seq_num_float, seq_num_complex), npt.Array1D[Any])
assert_type(pu.mapdomain(seq_num_float, seq_num_complex, seq_num_complex), npt.Array1D[Any])
assert_type(pu.mapdomain(seq_num_complex, seq_num_complex, seq_num_complex), npt.Array1D[Any])
assert_type(pu.mapdomain(list_num_complex, seq_num_complex, seq_num_complex), npt.Array1D[np.complex128])
assert_type(pu.mapdomain(seq_num_complex, seq_num_complex, seq_num_object), npt.Array1D[Any])
assert_type(pu.mapdomain(seq_num_complex, seq_num_object, seq_num_object), npt.Array1D[Any])
assert_type(pu.mapdomain(seq_num_object, seq_num_object, seq_num_object), npt.Array1D[np.object_])

assert_type(pu.mapdomain(seq_sct_int, seq_sct_int, seq_sct_int), npt.Array1D[np.float64])
assert_type(pu.mapdomain(seq_sct_int, seq_sct_int, seq_sct_float), npt.Array1D[np.float64])
assert_type(pu.mapdomain(seq_sct_int, seq_sct_float, seq_sct_float), npt.Array1D[np.float64])
assert_type(pu.mapdomain(seq_sct_float, seq_sct_float, seq_sct_float), npt.Array1D[np.float64])
assert_type(pu.mapdomain(seq_sct_float, seq_sct_float, seq_sct_complex), npt.Array1D[Any])
assert_type(pu.mapdomain(seq_sct_float, seq_sct_complex, seq_sct_complex), npt.Array1D[Any])
assert_type(pu.mapdomain(seq_sct_complex, seq_sct_complex, seq_sct_complex), npt.Array1D[Any])
assert_type(pu.mapdomain(seq_sct_complex, seq_sct_complex, seq_sct_object), npt.Array1D[Any])
assert_type(pu.mapdomain(seq_sct_complex, seq_sct_object, seq_sct_object), npt.Array1D[Any])

assert_type(pu.mapdomain(arr_int, arr_int, arr_int), npt.NDArray[np.float64])
assert_type(pu.mapdomain(arr_int, arr_int, arr_float), npt.NDArray[np.float64])
assert_type(pu.mapdomain(arr_int, arr_float, arr_float), npt.NDArray[np.float64])
assert_type(pu.mapdomain(arr_float, arr_float, arr_float), npt.NDArray[np.float64])
assert_type(pu.mapdomain(arr_float, arr_float, arr_complex), npt.NDArray[Any])
assert_type(pu.mapdomain(arr_float, arr_complex, arr_complex), npt.NDArray[Any])
assert_type(pu.mapdomain(arr_complex, arr_complex, arr_complex), npt.NDArray[np.complex128])
assert_type(pu.mapdomain(arr_complex, arr_complex, arr_object), npt.NDArray[Any])
assert_type(pu.mapdomain(arr_complex, arr_object, arr_object), npt.NDArray[Any])
assert_type(pu.mapdomain(arr_object, arr_object, arr_object), npt.NDArray[np.object_])

assert_type(pu.mapdomain(arr_float_2d, seq_num_float, seq_num_float), npt.Array2D[np.float64])
assert_type(pu.mapdomain(arr_complex_2d, seq_num_complex, seq_num_complex), npt.Array2D[np.complex128])
assert_type(pu.mapdomain(arr_object_2d, seq_num_object, seq_num_object), npt.Array2D[np.object_])
assert_type(pu.mapdomain(seq_arr_float, seq_num_float, seq_num_float), npt.NDArray[Any] | Any)
assert_type(pu.mapdomain(seq_arr_object_int, seq_num_float, seq_num_float), npt.NDArray[Any] | Any)
