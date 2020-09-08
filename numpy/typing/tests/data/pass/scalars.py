import sys
import datetime as dt

import numpy as np


# Construction
class D:
    def __index__(self) -> int:
        return 0


class C:
    def __complex__(self) -> complex:
        return 3j


class B:
    def __int__(self) -> int:
        return 4


class A:
    def __float__(self) -> float:
        return 4.0


np.complex64(3j)
np.complex64(A())
np.complex64(C())
np.complex128(3j)
np.complex128(C())
np.complex128(None)
np.complex64("1.2")
np.complex128(b"2j")

np.int8(4)
np.int16(3.4)
np.int32(4)
np.int64(-1)
np.uint8(B())
np.uint32()
np.int32("1")
np.int64(b"2")

np.float16(A())
np.float32(16)
np.float64(3.0)
np.float64(None)
np.float32("1")
np.float16(b"2.5")

if sys.version_info >= (3, 8):
    np.uint64(D())
    np.float32(D())
    np.complex64(D())

np.bytes_(b"hello")
np.bytes_("hello", 'utf-8')
np.bytes_("hello", encoding='utf-8')
np.str_("hello")
np.str_(b"hello", 'utf-8')
np.str_(b"hello", encoding='utf-8')

# Protocols
float(np.int8(4))
int(np.int16(5))
np.int8(np.float32(6))

# TODO(alan): test after https://github.com/python/typeshed/pull/2004
# complex(np.int32(8))

abs(np.int8(4))

# Array-ish semantics
np.int8().real
np.int16().imag
np.int32().data
np.int64().flags

np.uint8().itemsize * 2
np.uint16().ndim + 1
np.uint32().strides
np.uint64().shape

# Time structures
np.datetime64()
np.datetime64(0, "D")
np.datetime64(0, b"D")
np.datetime64(0, ('ms', 3))
np.datetime64("2019")
np.datetime64(b"2019")
np.datetime64("2019", "D")
np.datetime64(np.datetime64())
np.datetime64(dt.datetime(2000, 5, 3))
np.datetime64(None)
np.datetime64(None, "D")

np.timedelta64()
np.timedelta64(0)
np.timedelta64(0, "D")
np.timedelta64(0, ('ms', 3))
np.timedelta64(0, b"D")
np.timedelta64("3")
np.timedelta64(b"5")
np.timedelta64(np.timedelta64(2))
np.timedelta64(dt.timedelta(2))
np.timedelta64(None)
np.timedelta64(None, "D")

np.void(1)
np.void(np.int64(1))
np.void(True)
np.void(np.bool_(True))
np.void(b"test")
np.void(np.bytes_("test"))
