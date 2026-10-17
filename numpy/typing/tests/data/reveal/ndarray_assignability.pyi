from typing import Any, Protocol, assert_type

import numpy as np
import numpy.typing as npt
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

b1_1d: npt.Array1D[np.bool]
u1_1d: npt.Array1D[np.uint8]
i2_1d: npt.Array1D[np.int16]
q_1d: npt.Array1D[np.longlong]
f4_1d: npt.Array1D[np.float32]
f8_1d: npt.Array1D[np.float64]
g_1d: npt.Array1D[np.longdouble]
c8_1d: npt.Array1D[np.complex64]
c16_1d: npt.Array1D[np.complex128]
G_1d: npt.Array1D[np.clongdouble]
V_1d: npt.Array1D[np.void]

assert_type(do_abs(b1_1d), npt.Array1D[np.bool])
assert_type(do_abs(u1_1d), npt.Array1D[np.uint8])
assert_type(do_abs(i2_1d), npt.Array1D[np.int16])
assert_type(do_abs(q_1d), npt.Array1D[np.longlong])
assert_type(do_abs(f4_1d), npt.Array1D[np.float32])
assert_type(do_abs(f8_1d), npt.Array1D[np.float64])
assert_type(do_abs(g_1d), npt.Array1D[np.longdouble])

assert_type(do_abs(c8_1d), npt.Array1D[np.float32])
# NOTE: Unfortunately it's not possible to have this return a `float64` sctype, see
# https://github.com/python/mypy/issues/14070
assert_type(do_abs(c16_1d), np.ndarray[tuple[int], np.dtype[np.floating[_64Bit]]])
assert_type(do_abs(G_1d), npt.Array1D[np.longdouble])

assert_type(do_invert(b1_1d), npt.Array1D[np.bool])
assert_type(do_invert(u1_1d), npt.Array1D[np.uint8])
assert_type(do_invert(i2_1d), npt.Array1D[np.int16])
assert_type(do_invert(q_1d), npt.Array1D[np.longlong])

assert_type(do_neg(u1_1d), npt.Array1D[np.uint8])
assert_type(do_neg(i2_1d), npt.Array1D[np.int16])
assert_type(do_neg(q_1d), npt.Array1D[np.longlong])
assert_type(do_neg(f4_1d), npt.Array1D[np.float32])
assert_type(do_neg(c16_1d), npt.Array1D[np.complex128])

assert_type(do_pos(u1_1d), npt.Array1D[np.uint8])
assert_type(do_pos(i2_1d), npt.Array1D[np.int16])
assert_type(do_pos(q_1d), npt.Array1D[np.longlong])
assert_type(do_pos(f4_1d), npt.Array1D[np.float32])
assert_type(do_pos(c16_1d), npt.Array1D[np.complex128])

# this shape is effectively equivalent to `tuple[int, *tuple[Any, ...]]`, i.e. ndim >= 1
assert_type(V_1d["field"], np.ndarray[tuple[int] | tuple[Any, ...]])
