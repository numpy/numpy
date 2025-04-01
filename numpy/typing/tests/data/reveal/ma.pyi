import numpy as np
from typing_extensions import assert_type
from typing import Any, TypeAlias, TypeVar
from numpy._typing import _Shape
from numpy import dtype, generic


_ScalarType_co = TypeVar("_ScalarType_co", bound=generic, covariant=True)
MaskedNDArray: TypeAlias = np.ma.MaskedArray[_Shape, dtype[_ScalarType_co]]

class MaskedNDArraySubclass(MaskedNDArray[np.complex128]): ...

MAR_b: MaskedNDArray[np.bool]
MAR_f4: MaskedNDArray[np.float32]
MAR_i8: MaskedNDArray[np.int64]
MAR_subclass: MaskedNDArraySubclass
MAR_1d: np.ma.MaskedArray[tuple[int], np.dtype[Any]]

assert_type(MAR_1d.shape, tuple[int])

assert_type(MAR_f4.dtype, np.dtype[np.float32])

assert_type(int(MAR_i8), int)
assert_type(float(MAR_f4), float)

assert_type(np.ma.min(MAR_b), np.bool)
assert_type(np.ma.min(MAR_f4), np.float32)
assert_type(np.ma.min(MAR_b, axis=0), Any)
assert_type(np.ma.min(MAR_f4, axis=0), Any)
assert_type(np.ma.min(MAR_b, keepdims=True), Any)
assert_type(np.ma.min(MAR_f4, keepdims=True), Any)
assert_type(np.ma.min(MAR_f4, out=MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.min(MAR_f4, None, MAR_subclass), MaskedNDArraySubclass)

assert_type(MAR_b.min(), np.bool)
assert_type(MAR_f4.min(), np.float32)
assert_type(MAR_b.min(axis=0), Any)
assert_type(MAR_f4.min(axis=0), Any)
assert_type(MAR_b.min(keepdims=True), Any)
assert_type(MAR_f4.min(keepdims=True), Any)
assert_type(MAR_f4.min(out=MAR_subclass), MaskedNDArraySubclass)
assert_type(MAR_f4.min(None, MAR_subclass), MaskedNDArraySubclass)

assert_type(np.ma.max(MAR_b), np.bool)
assert_type(np.ma.max(MAR_f4), np.float32)
assert_type(np.ma.max(MAR_b, axis=0), Any)
assert_type(np.ma.max(MAR_f4, axis=0), Any)
assert_type(np.ma.max(MAR_b, keepdims=True), Any)
assert_type(np.ma.max(MAR_f4, keepdims=True), Any)
assert_type(np.ma.max(MAR_f4, out=MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.max(MAR_f4, None, MAR_subclass), MaskedNDArraySubclass)

assert_type(np.ma.ptp(MAR_b), np.bool)
assert_type(np.ma.ptp(MAR_f4), np.float32)
assert_type(np.ma.ptp(MAR_b, axis=0), Any)
assert_type(np.ma.ptp(MAR_f4, axis=0), Any)
assert_type(np.ma.ptp(MAR_b, keepdims=True), Any)
assert_type(np.ma.ptp(MAR_f4, keepdims=True), Any)
assert_type(np.ma.ptp(MAR_f4, out=MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.ptp(MAR_f4, None, MAR_subclass), MaskedNDArraySubclass)
