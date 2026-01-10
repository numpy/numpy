from typing import Any, Protocol, assert_type

import numpy as np
from numpy._typing import _64Bit

class CanAbs[T](Protocol):
    def __abs__(self, /) -> T: ...

class CanInvert[T](Protocol):
    def __invert__(self, /) -> T: ...

class CanNeg[T](Protocol):
    def __neg__(self, /) -> T: ...

class CanPos[T](Protocol):
    def __pos__(self, /) -> T: ...

def do_abs[T](x: CanAbs[T]) -> T: ...
def do_invert[T](x: CanInvert[T]) -> T: ...
def do_neg[T](x: CanNeg[T]) -> T: ...
def do_pos[T](x: CanPos[T]) -> T: ...

type _Bool_1d = np.ndarray[tuple[int], np.dtype[np.bool]]
type _UInt8_1d = np.ndarray[tuple[int], np.dtype[np.uint8]]
type _Int16_1d = np.ndarray[tuple[int], np.dtype[np.int16]]
type _LongLong_1d = np.ndarray[tuple[int], np.dtype[np.longlong]]
type _Float32_1d = np.ndarray[tuple[int], np.dtype[np.float32]]
type _Float64_1d = np.ndarray[tuple[int], np.dtype[np.float64]]
type _LongDouble_1d = np.ndarray[tuple[int], np.dtype[np.longdouble]]
type _Complex64_1d = np.ndarray[tuple[int], np.dtype[np.complex64]]
type _Complex128_1d = np.ndarray[tuple[int], np.dtype[np.complex128]]
type _CLongDouble_1d = np.ndarray[tuple[int], np.dtype[np.clongdouble]]
type _Void_1d = np.ndarray[tuple[int], np.dtype[np.void]]

b1_1d: _Bool_1d
u1_1d: _UInt8_1d
i2_1d: _Int16_1d
q_1d: _LongLong_1d
f4_1d: _Float32_1d
f8_1d: _Float64_1d
g_1d: _LongDouble_1d
c8_1d: _Complex64_1d
c16_1d: _Complex128_1d
G_1d: _CLongDouble_1d
V_1d: _Void_1d

assert_type(do_abs(b1_1d), _Bool_1d)
assert_type(do_abs(u1_1d), _UInt8_1d)
assert_type(do_abs(i2_1d), _Int16_1d)
assert_type(do_abs(q_1d), _LongLong_1d)
assert_type(do_abs(f4_1d), _Float32_1d)
assert_type(do_abs(f8_1d), _Float64_1d)
assert_type(do_abs(g_1d), _LongDouble_1d)

assert_type(do_abs(c8_1d), _Float32_1d)
# NOTE: Unfortunately it's not possible to have this return a `float64` sctype, see
# https://github.com/python/mypy/issues/14070
assert_type(do_abs(c16_1d), np.ndarray[tuple[int], np.dtype[np.floating[_64Bit]]])
assert_type(do_abs(G_1d), _LongDouble_1d)

assert_type(do_invert(b1_1d), _Bool_1d)
assert_type(do_invert(u1_1d), _UInt8_1d)
assert_type(do_invert(i2_1d), _Int16_1d)
assert_type(do_invert(q_1d), _LongLong_1d)

assert_type(do_neg(u1_1d), _UInt8_1d)
assert_type(do_neg(i2_1d), _Int16_1d)
assert_type(do_neg(q_1d), _LongLong_1d)
assert_type(do_neg(f4_1d), _Float32_1d)
assert_type(do_neg(c16_1d), _Complex128_1d)

assert_type(do_pos(u1_1d), _UInt8_1d)
assert_type(do_pos(i2_1d), _Int16_1d)
assert_type(do_pos(q_1d), _LongLong_1d)
assert_type(do_pos(f4_1d), _Float32_1d)
assert_type(do_pos(c16_1d), _Complex128_1d)

# this shape is effectively equivalent to `tuple[int, *tuple[Any, ...]]`, i.e. ndim >= 1
assert_type(V_1d["field"], np.ndarray[tuple[int] | tuple[Any, ...]])
