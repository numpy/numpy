import numpy as np


# Construction
class C:
    def __complex__(self):
        return 3j


class B:
    def __int__(self):
        return 4


class A:
    def __float__(self):
        return 4.0


np.complex64(3j)
np.complex64(C())
np.complex128(3j)
np.complex128(C())
np.complex128(None)

np.int8(4)
np.int16(3.4)
np.int32(4)
np.int64(-1)
np.uint8(B())
np.uint32()

np.float16(A())
np.float32(16)
np.float64(3.0)
np.float64(None)

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
np.datetime64("2019")
np.datetime64("2019", "D")
np.datetime64(None)
np.datetime64(None, "D")

np.timedelta64()
np.timedelta64(0)
np.timedelta64(0, "D")
np.timedelta64(None)
np.timedelta64(None, "D")

dt_64 = np.datetime64(0, "D")
td_64 = np.timedelta64(1, "h")

dt_64 + td_64
dt_64 - dt_64
dt_64 - td_64

td_64 + td_64
td_64 - td_64
td_64 / 1.0
td_64 / td_64
td_64 % td_64

np.void(1)
np.void(np.int64(1))
np.void(True)
np.void(np.bool_(True))
np.void(b"test")
np.void(np.bytes_("test"))
