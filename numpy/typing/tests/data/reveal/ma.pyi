from datetime import datetime, timedelta
from typing import Any, TypeAlias, TypeVar, assert_type

import numpy as np
from numpy import dtype, generic
from numpy._typing import NDArray, _Shape

_ScalarT_co = TypeVar("_ScalarT_co", bound=generic, covariant=True)
MaskedNDArray: TypeAlias = np.ma.MaskedArray[_Shape, dtype[_ScalarT_co]]

class MaskedNDArraySubclass(MaskedNDArray[np.complex128]): ...

AR_f4: NDArray[np.float32]
AR_dt64: NDArray[np.datetime64]
AR_td64: NDArray[np.timedelta64]
AR_o: NDArray[np.timedelta64]

MAR_b: MaskedNDArray[np.bool]
MAR_f4: MaskedNDArray[np.float32]
MAR_f8: MaskedNDArray[np.float64]
MAR_i8: MaskedNDArray[np.int64]
MAR_dt64: MaskedNDArray[np.datetime64]
MAR_td64: MaskedNDArray[np.timedelta64]
MAR_o: MaskedNDArray[np.object_]
MAR_s: MaskedNDArray[np.str_]
MAR_byte: MaskedNDArray[np.bytes_]
MAR_V: MaskedNDArray[np.void]

MAR_subclass: MaskedNDArraySubclass

MAR_1d: np.ma.MaskedArray[tuple[int], np.dtype]

b: np.bool
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
assert_type(MAR_V.partition(1, axis=0, kind='introselect', order='K'), None)

assert_type(MAR_f4.argpartition(1), MaskedNDArray[np.intp])
assert_type(MAR_1d.argpartition(1, axis=0, kind='introselect', order='K'), MaskedNDArray[np.intp])

assert_type(np.ma.ndim(f4), int)
assert_type(np.ma.ndim(MAR_b), int)
assert_type(np.ma.ndim(AR_f4), int)

assert_type(np.ma.size(b), int)
assert_type(np.ma.size(MAR_f4, axis=0), int)
assert_type(np.ma.size(AR_f4), int)

assert_type(np.ma.is_masked(MAR_f4), bool)

assert_type(MAR_f4.ids(), tuple[int, int])

assert_type(MAR_f4.iscontiguous(), bool)

assert_type(MAR_f4 >= 3, MaskedNDArray[np.bool])
assert_type(MAR_i8 >= AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_b >= AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_td64 >= AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_dt64 >= AR_dt64, MaskedNDArray[np.bool])
assert_type(MAR_o >= AR_o, MaskedNDArray[np.bool])
assert_type(MAR_1d >= 0, MaskedNDArray[np.bool])
assert_type(MAR_s >= MAR_s, MaskedNDArray[np.bool])
assert_type(MAR_byte >= MAR_byte, MaskedNDArray[np.bool])

assert_type(MAR_f4 > 3, MaskedNDArray[np.bool])
assert_type(MAR_i8 > AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_b > AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_td64 > AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_dt64 > AR_dt64, MaskedNDArray[np.bool])
assert_type(MAR_o > AR_o, MaskedNDArray[np.bool])
assert_type(MAR_1d > 0, MaskedNDArray[np.bool])
assert_type(MAR_s > MAR_s, MaskedNDArray[np.bool])
assert_type(MAR_byte > MAR_byte, MaskedNDArray[np.bool])

assert_type(MAR_f4 <= 3, MaskedNDArray[np.bool])
assert_type(MAR_i8 <= AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_b <= AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_td64 <= AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_dt64 <= AR_dt64, MaskedNDArray[np.bool])
assert_type(MAR_o <= AR_o, MaskedNDArray[np.bool])
assert_type(MAR_1d <= 0, MaskedNDArray[np.bool])
assert_type(MAR_s <= MAR_s, MaskedNDArray[np.bool])
assert_type(MAR_byte <= MAR_byte, MaskedNDArray[np.bool])

assert_type(MAR_f4 < 3, MaskedNDArray[np.bool])
assert_type(MAR_i8 < AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_b < AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_td64 < AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_dt64 < AR_dt64, MaskedNDArray[np.bool])
assert_type(MAR_o < AR_o, MaskedNDArray[np.bool])
assert_type(MAR_1d < 0, MaskedNDArray[np.bool])
assert_type(MAR_s < MAR_s, MaskedNDArray[np.bool])
assert_type(MAR_byte < MAR_byte, MaskedNDArray[np.bool])

assert_type(MAR_f4 <= 3, MaskedNDArray[np.bool])
assert_type(MAR_i8 <= AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_b <= AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_td64 <= AR_td64, MaskedNDArray[np.bool])
assert_type(MAR_dt64 <= AR_dt64, MaskedNDArray[np.bool])
assert_type(MAR_o <= AR_o, MaskedNDArray[np.bool])
assert_type(MAR_1d <= 0, MaskedNDArray[np.bool])
assert_type(MAR_s <= MAR_s, MaskedNDArray[np.bool])
assert_type(MAR_byte <= MAR_byte, MaskedNDArray[np.bool])

assert_type(MAR_byte.count(), int)
assert_type(MAR_f4.count(axis=None), int)
assert_type(MAR_f4.count(axis=0), NDArray[np.int_])
assert_type(MAR_b.count(axis=(0,1)), NDArray[np.int_])
assert_type(MAR_o.count(keepdims=True), NDArray[np.int_])
assert_type(MAR_o.count(axis=None, keepdims=True), NDArray[np.int_])
assert_type(MAR_o.count(None, True), NDArray[np.int_])

assert_type(np.ma.count(MAR_byte), int)
assert_type(np.ma.count(MAR_byte, axis=None), int)
assert_type(np.ma.count(MAR_f4, axis=0), NDArray[np.int_])
assert_type(np.ma.count(MAR_b, axis=(0,1)), NDArray[np.int_])
assert_type(np.ma.count(MAR_o, keepdims=True), NDArray[np.int_])
assert_type(np.ma.count(MAR_o, axis=None, keepdims=True), NDArray[np.int_])
assert_type(np.ma.count(MAR_o, None, True), NDArray[np.int_])

assert_type(MAR_f4.compressed(), np.ndarray[tuple[int], np.dtype[np.float32]])

assert_type(np.ma.compressed(MAR_i8), np.ndarray[tuple[int], np.dtype[np.int64]])
assert_type(np.ma.compressed([[1,2,3]]), np.ndarray[tuple[int], np.dtype])

assert_type(MAR_f4.put([0,4,8], [10,20,30]), None)
assert_type(MAR_f4.put(4, 999), None)
assert_type(MAR_f4.put(4, 999, mode='clip'), None)

assert_type(np.ma.put(MAR_f4, [0,4,8], [10,20,30]), None)
assert_type(np.ma.put(MAR_f4, 4, 999), None)
assert_type(np.ma.put(MAR_f4, 4, 999, mode='clip'), None)

assert_type(np.ma.putmask(MAR_f4, [True, False], [0, 1]), None)

assert_type(MAR_f4.filled(float('nan')), NDArray[np.float32])
assert_type(MAR_i8.filled(), NDArray[np.int64])
assert_type(MAR_1d.filled(), np.ndarray[tuple[int], np.dtype])

assert_type(np.ma.filled(MAR_f4, float('nan')), NDArray[np.float32])
assert_type(np.ma.filled([[1,2,3]]), NDArray[Any])
# PyRight detects this one correctly, but mypy doesn't.
# https://github.com/numpy/numpy/pull/28742#discussion_r2048968375
assert_type(np.ma.filled(MAR_1d), np.ndarray[tuple[int], np.dtype])  # type: ignore[assert-type]
