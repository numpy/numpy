"""
Tests for miscellaneous (non-magic) ``np.ndarray``/``np.generic`` methods.

More extensive tests are performed for the methods'
function-based counterpart in `../from_numeric.py`.

"""

import ctypes as ct
import datetime as dt
import operator
from collections.abc import Iterator
from types import ModuleType
from typing import Any, Literal, assert_type
from typing_extensions import CapsuleType

import numpy as np
import numpy.typing as npt

class SubClass(np.ndarray[tuple[Any, ...], np.dtype[np.object_]]): ...

f8: np.float64
i8: np.int64
b1: np.bool
m8_ns: np.timedelta64[int]
m8_ms: np.timedelta64[dt.timedelta]
m8_na: np.timedelta64[None]

B: SubClass

AR_f8: npt.NDArray[np.float64]
AR_i8: npt.NDArray[np.int64]
AR_u1: npt.NDArray[np.uint8]
AR_c8: npt.NDArray[np.complex64]
AR_m: npt.NDArray[np.timedelta64]
AR_U: npt.NDArray[np.str_]
AR_V: npt.NDArray[np.void]

AR_f8_1d: np.ndarray[tuple[int], np.dtype[np.float64]]
AR_f8_2d: np.ndarray[tuple[int, int], np.dtype[np.float64]]
AR_f8_3d: np.ndarray[tuple[int, int, int], np.dtype[np.float64]]

AR_any: np.ndarray

ctypes_obj = AR_f8.ctypes

assert_type(AR_f8.__dlpack__(), CapsuleType)
assert_type(AR_f8.__dlpack_device__(), tuple[Literal[1], Literal[0]])

assert_type(ctypes_obj.data, int)
assert_type(ctypes_obj.shape, ct.Array[np.ctypeslib.c_intp])
assert_type(ctypes_obj.strides, ct.Array[np.ctypeslib.c_intp])
assert_type(ctypes_obj._as_parameter_, ct.c_void_p)

assert_type(ctypes_obj.data_as(ct.c_void_p), ct.c_void_p)
assert_type(ctypes_obj.shape_as(ct.c_longlong), ct.Array[ct.c_longlong])
assert_type(ctypes_obj.strides_as(ct.c_ubyte), ct.Array[ct.c_ubyte])

assert_type(f8.all(), np.bool)
assert_type(AR_f8.all(), np.bool)
assert_type(AR_f8.all(axis=0), npt.NDArray[np.bool])
assert_type(AR_f8.all(keepdims=True), npt.NDArray[np.bool])
assert_type(AR_f8_1d.all(keepdims=True), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(AR_f8_2d.all(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(AR_f8_3d.all(keepdims=True), np.ndarray[tuple[int, int, int], np.dtype[np.bool]])
assert_type(AR_f8.all(out=B), SubClass)

assert_type(f8.any(), np.bool)
assert_type(AR_f8.any(), np.bool)
assert_type(AR_f8.any(axis=0), npt.NDArray[np.bool])
assert_type(AR_f8.any(keepdims=True), npt.NDArray[np.bool])
assert_type(AR_f8_1d.any(keepdims=True), np.ndarray[tuple[int], np.dtype[np.bool]])
assert_type(AR_f8_2d.any(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.bool]])
assert_type(AR_f8_3d.any(keepdims=True), np.ndarray[tuple[int, int, int], np.dtype[np.bool]])
assert_type(AR_f8.any(out=B), SubClass)

# same as below
assert_type(f8.argmax(), np.intp)
assert_type(AR_f8.argmax(), np.intp)
assert_type(AR_f8.argmax(axis=0), npt.NDArray[np.intp])
assert_type(AR_f8.argmax(out=AR_i8), npt.NDArray[np.int64])
assert_type(AR_f8.argmax(keepdims=True), npt.NDArray[np.intp])
assert_type(AR_f8_1d.argmax(keepdims=True), np.ndarray[tuple[int], np.dtype[np.intp]])
assert_type(AR_f8_2d.argmax(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.intp]])
assert_type(AR_f8_3d.argmax(keepdims=True), np.ndarray[tuple[int, int, int], np.dtype[np.intp]])

# same as above
assert_type(f8.argmin(), np.intp)
assert_type(AR_f8.argmin(), np.intp)
assert_type(AR_f8.argmin(axis=0), npt.NDArray[np.intp])
assert_type(AR_f8.argmin(out=AR_i8), npt.NDArray[np.int64])
assert_type(AR_f8.argmin(keepdims=True), npt.NDArray[np.intp])
assert_type(AR_f8_1d.argmin(keepdims=True), np.ndarray[tuple[int], np.dtype[np.intp]])
assert_type(AR_f8_2d.argmin(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.intp]])
assert_type(AR_f8_3d.argmin(keepdims=True), np.ndarray[tuple[int, int, int], np.dtype[np.intp]])

assert_type(f8.argsort(), npt.NDArray[np.intp])
assert_type(AR_f8.argsort(), npt.NDArray[np.intp])

assert_type(f8.astype(np.int64).choose([()]), npt.NDArray[Any])
assert_type(AR_f8.choose([0]), npt.NDArray[Any])
assert_type(AR_f8.choose([0], out=B), SubClass)

assert_type(f8.clip(1), npt.NDArray[Any])
assert_type(AR_f8.clip(1), npt.NDArray[Any])
assert_type(AR_f8.clip(None, 1), npt.NDArray[Any])
assert_type(AR_f8.clip(1, out=B), SubClass)
assert_type(AR_f8.clip(None, 1, out=B), SubClass)

assert_type(f8.compress([0]), npt.NDArray[Any])
assert_type(AR_f8.compress([0]), npt.NDArray[Any])
assert_type(AR_f8.compress([0], out=B), SubClass)

assert_type(f8.conj(), np.float64)
assert_type(AR_f8.conj(), npt.NDArray[np.float64])
assert_type(B.conj(), SubClass)

assert_type(f8.conjugate(), np.float64)
assert_type(AR_f8.conjugate(), npt.NDArray[np.float64])
assert_type(B.conjugate(), SubClass)

assert_type(AR_i8.cumprod(), np.ndarray[tuple[int], np.dtype[np.int64]])
assert_type(AR_i8.cumprod(axis=0), npt.NDArray[np.int64])
assert_type(AR_f8.cumprod(), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(AR_f8.cumprod(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8.cumprod(dtype=np.float32), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(AR_f8.cumprod(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8.cumprod(out=B), SubClass)
assert_type(AR_f8_2d.cumprod(), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(AR_f8_2d.cumprod(axis=0), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.cumprod(dtype=np.float32), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(AR_f8_2d.cumprod(dtype=np.float32, axis=0), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(AR_any.cumprod(), np.ndarray[tuple[int]])
assert_type(AR_any.cumprod(axis=0), np.ndarray)

assert_type(AR_i8.cumsum(), np.ndarray[tuple[int], np.dtype[np.int64]])
assert_type(AR_i8.cumsum(axis=0), npt.NDArray[np.int64])
assert_type(AR_f8.cumsum(), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(AR_f8.cumsum(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8.cumsum(dtype=np.float32), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(AR_f8.cumsum(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8.cumsum(out=B), SubClass)
assert_type(AR_f8_2d.cumsum(), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(AR_f8_2d.cumsum(axis=0), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.cumsum(dtype=np.float32), np.ndarray[tuple[int], np.dtype[np.float32]])
assert_type(AR_f8_2d.cumsum(dtype=np.float32, axis=0), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(AR_any.cumsum(), np.ndarray[tuple[int]])
assert_type(AR_any.cumsum(axis=0), np.ndarray)

# same as below (but without `timedelta64`)
assert_type(b1.prod(), np.int_)
assert_type(i8.prod(), np.int64)
assert_type(f8.prod(), np.float64)
assert_type(AR_i8.prod(), np.int64)
assert_type(AR_i8.prod(keepdims=True), npt.NDArray[np.int64])
assert_type(AR_i8.prod(axis=0), npt.NDArray[np.int64])
assert_type(AR_i8.prod(axis=0, keepdims=True), npt.NDArray[np.int64])
assert_type(AR_f8.prod(), np.float64)
assert_type(AR_f8.prod(keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.prod(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8.prod(axis=0, keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.prod(dtype=np.float32), np.float32)
assert_type(AR_f8.prod(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8.prod(dtype=np.float32, keepdims=True), npt.NDArray[np.float32])
assert_type(AR_f8.prod(dtype=np.float32, axis=0, keepdims=True), npt.NDArray[np.float32])
assert_type(AR_f8.prod(out=B), SubClass)
assert_type(AR_f8_2d.prod(), np.float64)
assert_type(AR_f8_2d.prod(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.prod(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8_2d.prod(axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.prod(dtype=np.float32), np.float32)
assert_type(AR_f8_2d.prod(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8_2d.prod(dtype=np.float32, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(AR_f8_2d.prod(dtype=np.float32, axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(AR_any.prod(), Any)

# same as above (but also accept `timedelta64`)
assert_type(b1.sum(), np.int_)
assert_type(i8.sum(), np.int64)
assert_type(f8.sum(), np.float64)
assert_type(m8_ns.sum(), np.timedelta64[int])
assert_type(m8_ms.sum(), np.timedelta64[dt.timedelta])
assert_type(m8_na.sum(), np.timedelta64[None])
assert_type(AR_i8.sum(), np.int64)
assert_type(AR_i8.sum(keepdims=True), npt.NDArray[np.int64])
assert_type(AR_i8.sum(axis=0), npt.NDArray[np.int64])
assert_type(AR_i8.sum(axis=0, keepdims=True), npt.NDArray[np.int64])
assert_type(AR_f8.sum(), np.float64)
assert_type(AR_f8.sum(keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.sum(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8.sum(axis=0, keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.sum(dtype=np.float32), np.float32)
assert_type(AR_f8.sum(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8.sum(dtype=np.float32, keepdims=True), npt.NDArray[np.float32])
assert_type(AR_f8.sum(dtype=np.float32, axis=0, keepdims=True), npt.NDArray[np.float32])
assert_type(AR_f8.sum(out=B), SubClass)
assert_type(AR_f8_2d.sum(), np.float64)
assert_type(AR_f8_2d.sum(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.sum(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8_2d.sum(axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.sum(dtype=np.float32), np.float32)
assert_type(AR_f8_2d.sum(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8_2d.sum(dtype=np.float32, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(AR_f8_2d.sum(dtype=np.float32, axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(AR_any.sum(), Any)

# same as below
assert_type(f8.mean(), Any)
assert_type(AR_f8.mean(), np.float64)
assert_type(AR_f8.mean(keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.mean(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8.mean(axis=0, keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.mean(dtype=np.float32), np.float32)
assert_type(AR_f8.mean(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8.mean(dtype=np.float32, keepdims=True), npt.NDArray[np.float32])
assert_type(AR_f8.mean(dtype=np.float32, axis=0, keepdims=True), npt.NDArray[np.float32])
assert_type(AR_f8.mean(out=B), SubClass)
assert_type(AR_f8_2d.mean(), np.float64)
assert_type(AR_f8_2d.mean(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.mean(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8_2d.mean(axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.mean(dtype=np.float32), np.float32)
assert_type(AR_f8_2d.mean(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8_2d.mean(dtype=np.float32, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(AR_f8_2d.mean(dtype=np.float32, axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# same as above
assert_type(f8.std(), Any)
assert_type(AR_f8.std(), np.float64)
assert_type(AR_f8.std(keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.std(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8.std(axis=0, keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.std(dtype=np.float32), np.float32)
assert_type(AR_f8.std(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8.std(dtype=np.float32, keepdims=True), npt.NDArray[np.float32])
assert_type(AR_f8.std(dtype=np.float32, axis=0, keepdims=True), npt.NDArray[np.float32])
assert_type(AR_f8.std(out=B), SubClass)
assert_type(AR_f8_2d.std(), np.float64)
assert_type(AR_f8_2d.std(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.std(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8_2d.std(axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.std(dtype=np.float32), np.float32)
assert_type(AR_f8_2d.std(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8_2d.std(dtype=np.float32, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(AR_f8_2d.std(dtype=np.float32, axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# same as above
assert_type(f8.var(), Any)
assert_type(AR_f8.var(), np.float64)
assert_type(AR_f8.var(keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.var(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8.var(axis=0, keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.var(dtype=np.float32), np.float32)
assert_type(AR_f8.var(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8.var(dtype=np.float32, keepdims=True), npt.NDArray[np.float32])
assert_type(AR_f8.var(dtype=np.float32, axis=0, keepdims=True), npt.NDArray[np.float32])
assert_type(AR_f8.var(out=B), SubClass)
assert_type(AR_f8_2d.var(), np.float64)
assert_type(AR_f8_2d.var(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.var(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8_2d.var(axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.var(dtype=np.float32), np.float32)
assert_type(AR_f8_2d.var(dtype=np.float32, axis=0), npt.NDArray[np.float32])
assert_type(AR_f8_2d.var(dtype=np.float32, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float32]])
assert_type(AR_f8_2d.var(dtype=np.float32, axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float32]])

# same as below
assert_type(f8.max(), Any)
assert_type(AR_i8.max(), np.int64)
assert_type(AR_i8.max(keepdims=True), npt.NDArray[np.int64])
assert_type(AR_i8.max(axis=0), npt.NDArray[np.int64])
assert_type(AR_i8.max(axis=0, keepdims=True), npt.NDArray[np.int64])
assert_type(AR_f8.max(), np.float64)
assert_type(AR_f8.max(keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.max(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8.max(axis=0, keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.max(out=B), SubClass)
assert_type(AR_f8_2d.max(), np.float64)
assert_type(AR_f8_2d.max(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.max(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8_2d.max(axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])

# same as above
assert_type(f8.min(), Any)
assert_type(AR_i8.min(), np.int64)
assert_type(AR_i8.min(keepdims=True), npt.NDArray[np.int64])
assert_type(AR_i8.min(axis=0), npt.NDArray[np.int64])
assert_type(AR_i8.min(axis=0, keepdims=True), npt.NDArray[np.int64])
assert_type(AR_f8.min(), np.float64)
assert_type(AR_f8.min(keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.min(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8.min(axis=0, keepdims=True), npt.NDArray[np.float64])
assert_type(AR_f8.min(out=B), SubClass)
assert_type(AR_f8_2d.min(), np.float64)
assert_type(AR_f8_2d.min(keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])
assert_type(AR_f8_2d.min(axis=0), npt.NDArray[np.float64])
assert_type(AR_f8_2d.min(axis=0, keepdims=True), np.ndarray[tuple[int, int], np.dtype[np.float64]])

assert_type(f8.round(), np.float64)
assert_type(AR_f8.round(), npt.NDArray[np.float64])
assert_type(AR_f8.round(out=B), SubClass)

assert_type(f8.repeat(1), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(f8.repeat(1, axis=0), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(AR_f8.repeat(1), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(AR_f8.repeat(1, axis=0), npt.NDArray[np.float64])
assert_type(B.repeat(1), np.ndarray[tuple[int], np.dtype[np.object_]])
assert_type(B.repeat(1, axis=0), npt.NDArray[np.object_])

assert_type(f8.take(0), np.float64)
assert_type(AR_f8.take(0), np.float64)
assert_type(AR_f8.take([0]), npt.NDArray[np.float64])
assert_type(AR_f8.take(0, out=B), SubClass)
assert_type(AR_f8.take([0], out=B), SubClass)

assert_type(AR_f8.argpartition(0), npt.NDArray[np.intp])
assert_type(AR_f8.argpartition([0]), npt.NDArray[np.intp])
assert_type(AR_f8.argpartition(0, axis=None), np.ndarray[tuple[int], np.dtype[np.intp]])

assert_type(AR_f8.diagonal(), npt.NDArray[np.float64])
assert_type(AR_f8_2d.diagonal(), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(AR_f8_3d.diagonal(), np.ndarray[tuple[int, int], np.dtype[np.float64]])

assert_type(AR_f8.dot(1), npt.NDArray[Any])
assert_type(AR_f8.dot([1]), Any)
assert_type(AR_f8.dot(1, out=B), SubClass)

type _Int1D = np.ndarray[tuple[int], np.dtype[np.intp]]

assert_type(AR_f8.nonzero(), tuple[_Int1D, ...])
assert_type(AR_f8_1d.nonzero(), tuple[_Int1D])
assert_type(AR_f8_2d.nonzero(), tuple[_Int1D, _Int1D])
assert_type(AR_f8_3d.nonzero(), tuple[_Int1D, _Int1D, _Int1D])

assert_type(AR_f8.searchsorted(1), np.intp)
assert_type(AR_f8.searchsorted([1]), npt.NDArray[np.intp])

assert_type(AR_f8.trace(), Any)
assert_type(AR_f8.trace(out=B), SubClass)

assert_type(AR_f8.item(), float)
assert_type(AR_U.item(), str)

assert_type(AR_f8.ravel(), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(AR_U.ravel(), np.ndarray[tuple[int], np.dtype[np.str_]])

assert_type(AR_f8.flatten(), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(AR_U.flatten(), np.ndarray[tuple[int], np.dtype[np.str_]])

assert_type(AR_i8.reshape(None), npt.NDArray[np.int64])
assert_type(AR_f8.reshape(-1), np.ndarray[tuple[int], np.dtype[np.float64]])
assert_type(AR_c8.reshape(2, 3, 4, 5), np.ndarray[tuple[int, int, int, int], np.dtype[np.complex64]])
assert_type(AR_m.reshape(()), np.ndarray[tuple[()], np.dtype[np.timedelta64]])
assert_type(AR_U.reshape([]), np.ndarray[tuple[()], np.dtype[np.str_]])
assert_type(AR_V.reshape((480, 720, 4)), np.ndarray[tuple[int, int, int], np.dtype[np.void]])

assert_type(int(AR_f8), int)
assert_type(int(AR_U), int)

assert_type(float(AR_f8), float)
assert_type(float(AR_U), float)

assert_type(complex(AR_f8), complex)

assert_type(operator.index(AR_i8), int)

assert_type(AR_f8.__array_wrap__(B), npt.NDArray[np.object_])

assert_type(AR_V[0], Any)
assert_type(AR_V[0, 0], Any)
assert_type(AR_V[AR_i8], npt.NDArray[np.void])
assert_type(AR_V[AR_i8, AR_i8], npt.NDArray[np.void])
assert_type(AR_V[AR_i8, None], npt.NDArray[np.void])
assert_type(AR_V[0, ...], npt.NDArray[np.void])
assert_type(AR_V[[0]], npt.NDArray[np.void])
assert_type(AR_V[[0], [0]], npt.NDArray[np.void])
assert_type(AR_V[:], npt.NDArray[np.void])
assert_type(AR_V["a"], npt.NDArray[Any])
assert_type(AR_V[["a", "b"]], npt.NDArray[np.void])

assert_type(AR_f8.dump("test_file"), None)
assert_type(AR_f8.dump(b"test_file"), None)
with open("test_file", "wb") as f:
    assert_type(AR_f8.dump(f), None)

assert_type(AR_f8.__array_finalize__(None), None)
assert_type(AR_f8.__array_finalize__(B), None)
assert_type(AR_f8.__array_finalize__(AR_f8), None)

assert_type(f8.device, Literal["cpu"])
assert_type(AR_f8.device, Literal["cpu"])

assert_type(f8.to_device("cpu"), np.float64)
assert_type(i8.to_device("cpu"), np.int64)
assert_type(AR_f8.to_device("cpu"), npt.NDArray[np.float64])
assert_type(AR_i8.to_device("cpu"), npt.NDArray[np.int64])
assert_type(AR_u1.to_device("cpu"), npt.NDArray[np.uint8])
assert_type(AR_c8.to_device("cpu"), npt.NDArray[np.complex64])
assert_type(AR_m.to_device("cpu"), npt.NDArray[np.timedelta64])

assert_type(f8.__array_namespace__(), ModuleType)
assert_type(AR_f8.__array_namespace__(), ModuleType)

assert_type(iter(AR_f8), Iterator[Any])  # any-D
assert_type(iter(AR_f8_1d), Iterator[np.float64])  # 1-D
assert_type(iter(AR_f8_2d), Iterator[npt.NDArray[np.float64]])  # 2-D
assert_type(iter(AR_f8_3d), Iterator[npt.NDArray[np.float64]])  # 3-D
