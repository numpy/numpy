import ctypes as ct
from typing import Any, assert_type

import numpy as np
import numpy.typing as npt
from numpy import ctypeslib

AR_bool: npt.NDArray[np.bool]
AR_i8: npt.NDArray[np.int8]
AR_u8: npt.NDArray[np.uint8]
AR_i16: npt.NDArray[np.int16]
AR_u16: npt.NDArray[np.uint16]
AR_i32: npt.NDArray[np.int32]
AR_u32: npt.NDArray[np.uint32]
AR_i64: npt.NDArray[np.int64]
AR_u64: npt.NDArray[np.uint64]
AR_f32: npt.NDArray[np.float32]
AR_f64: npt.NDArray[np.float64]
AR_f80: npt.NDArray[np.longdouble]
AR_void: npt.NDArray[np.void]

pointer: ct._Pointer[Any]

assert_type(np.ctypeslib.c_intp(), ctypeslib.c_intp)

assert_type(np.ctypeslib.ndpointer(), type[ctypeslib._ndptr[None]])
assert_type(np.ctypeslib.ndpointer(dtype=np.float64), type[ctypeslib._ndptr[np.dtype[np.float64]]])
assert_type(np.ctypeslib.ndpointer(dtype=float), type[ctypeslib._ndptr[np.dtype]])
assert_type(np.ctypeslib.ndpointer(shape=(10, 3)), type[ctypeslib._ndptr[None]])
assert_type(np.ctypeslib.ndpointer(np.int64, shape=(10, 3)), type[ctypeslib._concrete_ndptr[np.dtype[np.int64]]])
assert_type(np.ctypeslib.ndpointer(int, shape=(1,)), type[np.ctypeslib._concrete_ndptr[np.dtype]])

assert_type(np.ctypeslib.as_ctypes_type(np.bool), type[ct.c_bool])
assert_type(np.ctypeslib.as_ctypes_type(np.int8), type[ct.c_int8])
assert_type(np.ctypeslib.as_ctypes_type(np.uint8), type[ct.c_uint8])
assert_type(np.ctypeslib.as_ctypes_type(np.int16), type[ct.c_int16])
assert_type(np.ctypeslib.as_ctypes_type(np.uint16), type[ct.c_uint16])
assert_type(np.ctypeslib.as_ctypes_type(np.int32), type[ct.c_int32])
assert_type(np.ctypeslib.as_ctypes_type(np.uint32), type[ct.c_uint32])
assert_type(np.ctypeslib.as_ctypes_type(np.int64), type[ct.c_int64])
assert_type(np.ctypeslib.as_ctypes_type(np.uint64), type[ct.c_uint64])
assert_type(np.ctypeslib.as_ctypes_type(np.float32), type[ct.c_float])
assert_type(np.ctypeslib.as_ctypes_type(np.float64), type[ct.c_double])
assert_type(np.ctypeslib.as_ctypes_type(np.longdouble), type[ct.c_longdouble])
assert_type(np.ctypeslib.as_ctypes_type("?"), type[ct.c_bool])
assert_type(np.ctypeslib.as_ctypes_type("intp"), type[ct.c_ssize_t])
assert_type(np.ctypeslib.as_ctypes_type("q"), type[ct.c_longlong])
assert_type(np.ctypeslib.as_ctypes_type("i8"), type[ct.c_int64])
assert_type(np.ctypeslib.as_ctypes_type("f8"), type[ct.c_double])
assert_type(np.ctypeslib.as_ctypes_type([("i8", np.int64), ("f8", np.float64)]), type[Any])

assert_type(np.ctypeslib.as_ctypes(AR_bool.take(0)), ct.c_bool)
assert_type(np.ctypeslib.as_ctypes(AR_u8.take(0)), ct.c_uint8)
assert_type(np.ctypeslib.as_ctypes(AR_u16.take(0)), ct.c_uint16)
assert_type(np.ctypeslib.as_ctypes(AR_u32.take(0)), ct.c_uint32)

assert_type(np.ctypeslib.as_ctypes(np.bool()), ct.c_bool)
assert_type(np.ctypeslib.as_ctypes(np.int8()), ct.c_int8)
assert_type(np.ctypeslib.as_ctypes(np.uint8()), ct.c_uint8)
assert_type(np.ctypeslib.as_ctypes(np.int16()), ct.c_int16)
assert_type(np.ctypeslib.as_ctypes(np.uint16()), ct.c_uint16)
assert_type(np.ctypeslib.as_ctypes(np.int32()), ct.c_int32)
assert_type(np.ctypeslib.as_ctypes(np.uint32()), ct.c_uint32)
assert_type(np.ctypeslib.as_ctypes(np.int64()), ct.c_int64)
assert_type(np.ctypeslib.as_ctypes(np.uint64()), ct.c_uint64)
assert_type(np.ctypeslib.as_ctypes(np.float32()), ct.c_float)
assert_type(np.ctypeslib.as_ctypes(np.float64()), ct.c_double)
assert_type(np.ctypeslib.as_ctypes(np.longdouble()), ct.c_longdouble)
assert_type(np.ctypeslib.as_ctypes(np.void(b"")), Any)
assert_type(np.ctypeslib.as_ctypes(AR_bool), ct.Array[ct.c_bool])
assert_type(np.ctypeslib.as_ctypes(AR_i8), ct.Array[ct.c_int8])
assert_type(np.ctypeslib.as_ctypes(AR_u8), ct.Array[ct.c_uint8])
assert_type(np.ctypeslib.as_ctypes(AR_i16), ct.Array[ct.c_int16])
assert_type(np.ctypeslib.as_ctypes(AR_u16), ct.Array[ct.c_uint16])
assert_type(np.ctypeslib.as_ctypes(AR_i32), ct.Array[ct.c_int32])
assert_type(np.ctypeslib.as_ctypes(AR_u32), ct.Array[ct.c_uint32])
assert_type(np.ctypeslib.as_ctypes(AR_i64), ct.Array[ct.c_int64])
assert_type(np.ctypeslib.as_ctypes(AR_u64), ct.Array[ct.c_uint64])
assert_type(np.ctypeslib.as_ctypes(AR_f32), ct.Array[ct.c_float])
assert_type(np.ctypeslib.as_ctypes(AR_f64), ct.Array[ct.c_double])
assert_type(np.ctypeslib.as_ctypes(AR_f80), ct.Array[ct.c_longdouble])
assert_type(np.ctypeslib.as_ctypes(AR_void), ct.Array[Any])

assert_type(np.ctypeslib.as_array(AR_u8), npt.NDArray[np.ubyte])
assert_type(np.ctypeslib.as_array(1), npt.NDArray[Any])
assert_type(np.ctypeslib.as_array(pointer), npt.NDArray[Any])
