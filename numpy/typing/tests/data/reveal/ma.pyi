import numpy as np
from typing_extensions import assert_type
from typing import Any, TypeAlias, TypeVar
from numpy._typing import _Shape, NDArray
from numpy import dtype, generic


_ScalarT_co = TypeVar("_ScalarT_co", bound=generic, covariant=True)
MaskedNDArray: TypeAlias = np.ma.MaskedArray[_Shape, dtype[_ScalarT_co]]

class MaskedNDArraySubclass(MaskedNDArray[np.complex128]): ...

AR_f4: NDArray[np.float32]

MAR_b: MaskedNDArray[np.bool]
MAR_f4: MaskedNDArray[np.float32]
MAR_f8: MaskedNDArray[np.float64]
MAR_i8: MaskedNDArray[np.int64]
MAR_subclass: MaskedNDArraySubclass
MAR_1d: np.ma.MaskedArray[tuple[int], np.dtype[Any]]

f4: np.float32
f: float

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
assert_type(np.ma.min(MAR_f4, 0, MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.min(MAR_f4, None, MAR_subclass), MaskedNDArraySubclass)

assert_type(MAR_b.min(), np.bool)
assert_type(MAR_f4.min(), np.float32)
assert_type(MAR_b.min(axis=0), Any)
assert_type(MAR_f4.min(axis=0), Any)
assert_type(MAR_b.min(keepdims=True), Any)
assert_type(MAR_f4.min(keepdims=True), Any)
assert_type(MAR_f4.min(out=MAR_subclass), MaskedNDArraySubclass)
assert_type(MAR_f4.min(0, MAR_subclass), MaskedNDArraySubclass)
assert_type(MAR_f4.min(None, MAR_subclass), MaskedNDArraySubclass)

assert_type(np.ma.max(MAR_b), np.bool)
assert_type(np.ma.max(MAR_f4), np.float32)
assert_type(np.ma.max(MAR_b, axis=0), Any)
assert_type(np.ma.max(MAR_f4, axis=0), Any)
assert_type(np.ma.max(MAR_b, keepdims=True), Any)
assert_type(np.ma.max(MAR_f4, keepdims=True), Any)
assert_type(np.ma.max(MAR_f4, out=MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.max(MAR_f4, 0, MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.max(MAR_f4, None, MAR_subclass), MaskedNDArraySubclass)

assert_type(MAR_b.max(), np.bool)
assert_type(MAR_f4.max(), np.float32)
assert_type(MAR_b.max(axis=0), Any)
assert_type(MAR_f4.max(axis=0), Any)
assert_type(MAR_b.max(keepdims=True), Any)
assert_type(MAR_f4.max(keepdims=True), Any)
assert_type(MAR_f4.max(out=MAR_subclass), MaskedNDArraySubclass)
assert_type(MAR_f4.max(0, MAR_subclass), MaskedNDArraySubclass)
assert_type(MAR_f4.max(None, MAR_subclass), MaskedNDArraySubclass)

assert_type(np.ma.ptp(MAR_b), np.bool)
assert_type(np.ma.ptp(MAR_f4), np.float32)
assert_type(np.ma.ptp(MAR_b, axis=0), Any)
assert_type(np.ma.ptp(MAR_f4, axis=0), Any)
assert_type(np.ma.ptp(MAR_b, keepdims=True), Any)
assert_type(np.ma.ptp(MAR_f4, keepdims=True), Any)
assert_type(np.ma.ptp(MAR_f4, out=MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.ptp(MAR_f4, 0, MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.ptp(MAR_f4, None, MAR_subclass), MaskedNDArraySubclass)

assert_type(MAR_b.ptp(), np.bool)
assert_type(MAR_f4.ptp(), np.float32)
assert_type(MAR_b.ptp(axis=0), Any)
assert_type(MAR_f4.ptp(axis=0), Any)
assert_type(MAR_b.ptp(keepdims=True), Any)
assert_type(MAR_f4.ptp(keepdims=True), Any)
assert_type(MAR_f4.ptp(out=MAR_subclass), MaskedNDArraySubclass)
assert_type(MAR_f4.ptp(0, MAR_subclass), MaskedNDArraySubclass)
assert_type(MAR_f4.ptp(None, MAR_subclass), MaskedNDArraySubclass)

assert_type(MAR_b.argmin(), np.intp)
assert_type(MAR_f4.argmin(), np.intp)
assert_type(MAR_f4.argmax(fill_value=6.28318, keepdims=False), np.intp)
assert_type(MAR_b.argmin(axis=0), Any)
assert_type(MAR_f4.argmin(axis=0), Any)
assert_type(MAR_b.argmin(keepdims=True), Any)
assert_type(MAR_f4.argmin(out=MAR_subclass), MaskedNDArraySubclass)
assert_type(MAR_f4.argmin(None, None, out=MAR_subclass), MaskedNDArraySubclass)

assert_type(np.ma.argmin(MAR_b), np.intp)
assert_type(np.ma.argmin(MAR_f4), np.intp)
assert_type(np.ma.argmin(MAR_f4, fill_value=6.28318, keepdims=False), np.intp)
assert_type(np.ma.argmin(MAR_b, axis=0), Any)
assert_type(np.ma.argmin(MAR_f4, axis=0), Any)
assert_type(np.ma.argmin(MAR_b, keepdims=True), Any)
assert_type(np.ma.argmin(MAR_f4, out=MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.argmin(MAR_f4, None, None, out=MAR_subclass), MaskedNDArraySubclass)

assert_type(MAR_b.argmax(), np.intp)
assert_type(MAR_f4.argmax(), np.intp)
assert_type(MAR_f4.argmax(fill_value=6.28318, keepdims=False), np.intp)
assert_type(MAR_b.argmax(axis=0), Any)
assert_type(MAR_f4.argmax(axis=0), Any)
assert_type(MAR_b.argmax(keepdims=True), Any)
assert_type(MAR_f4.argmax(out=MAR_subclass), MaskedNDArraySubclass)
assert_type(MAR_f4.argmax(None, None, out=MAR_subclass), MaskedNDArraySubclass)

assert_type(np.ma.argmax(MAR_b), np.intp)
assert_type(np.ma.argmax(MAR_f4), np.intp)
assert_type(np.ma.argmax(MAR_f4, fill_value=6.28318, keepdims=False), np.intp)
assert_type(np.ma.argmax(MAR_b, axis=0), Any)
assert_type(np.ma.argmax(MAR_f4, axis=0), Any)
assert_type(np.ma.argmax(MAR_b, keepdims=True), Any)
assert_type(np.ma.argmax(MAR_f4, out=MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.argmax(MAR_f4, None, None, out=MAR_subclass), MaskedNDArraySubclass)

assert_type(MAR_f4.sort(), None)
assert_type(MAR_f4.sort(axis=0, kind='quicksort', order='K', endwith=False, fill_value=42., stable=False), None)

assert_type(np.ma.sort(MAR_f4), MaskedNDArray[np.float32])
assert_type(np.ma.sort(MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.sort([[0, 1], [2, 3]]), NDArray[Any])
assert_type(np.ma.sort(AR_f4), NDArray[np.float32])

assert_type(MAR_f8.take(0), np.float64)
assert_type(MAR_1d.take(0), Any)
assert_type(MAR_f8.take([0]), MaskedNDArray[np.float64])
assert_type(MAR_f8.take(0, out=MAR_subclass), MaskedNDArraySubclass)
assert_type(MAR_f8.take([0], out=MAR_subclass), MaskedNDArraySubclass)

assert_type(np.ma.take(f, 0), Any)
assert_type(np.ma.take(f4, 0), np.float32)
assert_type(np.ma.take(MAR_f8, 0), np.float64)
assert_type(np.ma.take(AR_f4, 0), np.float32)
assert_type(np.ma.take(MAR_1d, 0), Any)
assert_type(np.ma.take(MAR_f8, [0]), MaskedNDArray[np.float64])
assert_type(np.ma.take(AR_f4, [0]), MaskedNDArray[np.float32])
assert_type(np.ma.take(MAR_f8, 0, out=MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.take(MAR_f8, [0], out=MAR_subclass), MaskedNDArraySubclass)
assert_type(np.ma.take([1], [0]), MaskedNDArray[Any])
assert_type(np.ma.take(np.eye(2), 1, axis=0), MaskedNDArray[np.float64])

assert_type(MAR_f4.partition(1), None)
assert_type(MAR_f4.partition(1, axis=0, kind='introselect', order='K'), None)

assert_type(MAR_f4.argpartition(1), MaskedNDArray[np.intp])
assert_type(MAR_1d.argpartition(1, axis=0, kind='introselect', order='K'), MaskedNDArray[np.intp])
