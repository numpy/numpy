import numpy as np

c16 = np.complex128()
f8 = np.float64()
i8 = np.int64()
u8 = np.uint64()

c8 = np.complex64()
f4 = np.float32()
i4 = np.int32()
u4 = np.uint32()

dt = np.datetime64(0, "D")
td = np.timedelta64(0, "D")

b_ = np.bool_()

b = bool()
c = complex()
f = float()
i = int()

AR = np.array([0], dtype=np.float64)
AR.setflags(write=False)

# Time structures

reveal_type(dt + td)  # E: numpy.datetime64
reveal_type(dt + i)  # E: numpy.datetime64
reveal_type(dt + i4)  # E: numpy.datetime64
reveal_type(dt + i8)  # E: numpy.datetime64
reveal_type(dt - dt)  # E: numpy.timedelta64
reveal_type(dt - i)  # E: numpy.datetime64
reveal_type(dt - i4)  # E: numpy.datetime64
reveal_type(dt - i8)  # E: numpy.datetime64

reveal_type(td + td)  # E: numpy.timedelta64
reveal_type(td + i)  # E: numpy.timedelta64
reveal_type(td + i4)  # E: numpy.timedelta64
reveal_type(td + i8)  # E: numpy.timedelta64
reveal_type(td - td)  # E: numpy.timedelta64
reveal_type(td - i)  # E: numpy.timedelta64
reveal_type(td - i4)  # E: numpy.timedelta64
reveal_type(td - i8)  # E: numpy.timedelta64
reveal_type(td / f)  # E: numpy.timedelta64
reveal_type(td / f4)  # E: numpy.timedelta64
reveal_type(td / f8)  # E: numpy.timedelta64
reveal_type(td / td)  # E: float64
reveal_type(td // td)  # E: signedinteger
reveal_type(td % td)  # E: numpy.timedelta64

# boolean

reveal_type(b_ / b)  # E: float64
reveal_type(b_ / b_)  # E: float64
reveal_type(b_ / i)  # E: float64
reveal_type(b_ / i8)  # E: float64
reveal_type(b_ / i4)  # E: float64
reveal_type(b_ / u8)  # E: float64
reveal_type(b_ / u4)  # E: float64
reveal_type(b_ / f)  # E: float64
reveal_type(b_ / f8)  # E: float64
reveal_type(b_ / f4)  # E: float32
reveal_type(b_ / c)  # E: complex128
reveal_type(b_ / c16)  # E: complex128
reveal_type(b_ / c8)  # E: complex64

reveal_type(b / b_)  # E: float64
reveal_type(b_ / b_)  # E: float64
reveal_type(i / b_)  # E: float64
reveal_type(i8 / b_)  # E: float64
reveal_type(i4 / b_)  # E: float64
reveal_type(u8 / b_)  # E: float64
reveal_type(u4 / b_)  # E: float64
reveal_type(f / b_)  # E: float64
reveal_type(f8 / b_)  # E: float64
reveal_type(f4 / b_)  # E: float32
reveal_type(c / b_)  # E: complex128
reveal_type(c16 / b_)  # E: complex128
reveal_type(c8 / b_)  # E: complex64

# Complex

reveal_type(c16 + c16)  # E: complexfloating
reveal_type(c16 + f8)  # E: complexfloating
reveal_type(c16 + i8)  # E: complexfloating
reveal_type(c16 + c8)  # E: complexfloating
reveal_type(c16 + f4)  # E: complexfloating
reveal_type(c16 + i4)  # E: complexfloating
reveal_type(c16 + b_)  # E: complex128
reveal_type(c16 + b)  # E: complexfloating
reveal_type(c16 + c)  # E: complexfloating
reveal_type(c16 + f)  # E: complexfloating
reveal_type(c16 + i)  # E: complexfloating
reveal_type(c16 + AR)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(c16 + c16)  # E: complexfloating
reveal_type(f8 + c16)  # E: complexfloating
reveal_type(i8 + c16)  # E: complexfloating
reveal_type(c8 + c16)  # E: complexfloating
reveal_type(f4 + c16)  # E: complexfloating
reveal_type(i4 + c16)  # E: complexfloating
reveal_type(b_ + c16)  # E: complex128
reveal_type(b + c16)  # E: complexfloating
reveal_type(c + c16)  # E: complexfloating
reveal_type(f + c16)  # E: complexfloating
reveal_type(i + c16)  # E: complexfloating
reveal_type(AR + c16)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(c8 + c16)  # E: complexfloating
reveal_type(c8 + f8)  # E: complexfloating
reveal_type(c8 + i8)  # E: complexfloating
reveal_type(c8 + c8)  # E: complexfloating
reveal_type(c8 + f4)  # E: complexfloating
reveal_type(c8 + i4)  # E: complexfloating
reveal_type(c8 + b_)  # E: complex64
reveal_type(c8 + b)  # E: complexfloating
reveal_type(c8 + c)  # E: complexfloating
reveal_type(c8 + f)  # E: complexfloating
reveal_type(c8 + i)  # E: complexfloating
reveal_type(c8 + AR)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(c16 + c8)  # E: complexfloating
reveal_type(f8 + c8)  # E: complexfloating
reveal_type(i8 + c8)  # E: complexfloating
reveal_type(c8 + c8)  # E: complexfloating
reveal_type(f4 + c8)  # E: complexfloating
reveal_type(i4 + c8)  # E: complexfloating
reveal_type(b_ + c8)  # E: complex64
reveal_type(b + c8)  # E: complexfloating
reveal_type(c + c8)  # E: complexfloating
reveal_type(f + c8)  # E: complexfloating
reveal_type(i + c8)  # E: complexfloating
reveal_type(AR + c8)  # E: Union[numpy.ndarray, numpy.generic]

# Float

reveal_type(f8 + f8)  # E: floating
reveal_type(f8 + i8)  # E: floating
reveal_type(f8 + f4)  # E: floating
reveal_type(f8 + i4)  # E: floating
reveal_type(f8 + b_)  # E: float64
reveal_type(f8 + b)  # E: floating
reveal_type(f8 + c)  # E: complexfloating
reveal_type(f8 + f)  # E: floating
reveal_type(f8 + i)  # E: floating
reveal_type(f8 + AR)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(f8 + f8)  # E: floating
reveal_type(i8 + f8)  # E: floating
reveal_type(f4 + f8)  # E: floating
reveal_type(i4 + f8)  # E: floating
reveal_type(b_ + f8)  # E: float64
reveal_type(b + f8)  # E: floating
reveal_type(c + f8)  # E: complexfloating
reveal_type(f + f8)  # E: floating
reveal_type(i + f8)  # E: floating
reveal_type(AR + f8)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(f4 + f8)  # E: floating
reveal_type(f4 + i8)  # E: floating
reveal_type(f4 + f4)  # E: floating
reveal_type(f4 + i4)  # E: floating
reveal_type(f4 + b_)  # E: float32
reveal_type(f4 + b)  # E: floating
reveal_type(f4 + c)  # E: complexfloating
reveal_type(f4 + f)  # E: floating
reveal_type(f4 + i)  # E: floating
reveal_type(f4 + AR)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(f8 + f4)  # E: floating
reveal_type(i8 + f4)  # E: floating
reveal_type(f4 + f4)  # E: floating
reveal_type(i4 + f4)  # E: floating
reveal_type(b_ + f4)  # E: float32
reveal_type(b + f4)  # E: floating
reveal_type(c + f4)  # E: complexfloating
reveal_type(f + f4)  # E: floating
reveal_type(i + f4)  # E: floating
reveal_type(AR + f4)  # E: Union[numpy.ndarray, numpy.generic]

# Int

reveal_type(i8 + i8)  # E: signedinteger
reveal_type(i8 + u8)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(i8 + i4)  # E: signedinteger
reveal_type(i8 + u4)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(i8 + b_)  # E: int64
reveal_type(i8 + b)  # E: signedinteger
reveal_type(i8 + c)  # E: complexfloating
reveal_type(i8 + f)  # E: floating
reveal_type(i8 + i)  # E: signedinteger
reveal_type(i8 + AR)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(u8 + u8)  # E: unsignedinteger
reveal_type(u8 + i4)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(u8 + u4)  # E: unsignedinteger
reveal_type(u8 + b_)  # E: uint64
reveal_type(u8 + b)  # E: unsignedinteger
reveal_type(u8 + c)  # E: complexfloating
reveal_type(u8 + f)  # E: floating
reveal_type(u8 + i)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(u8 + AR)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(i8 + i8)  # E: signedinteger
reveal_type(u8 + i8)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(i4 + i8)  # E: signedinteger
reveal_type(u4 + i8)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(b_ + i8)  # E: int64
reveal_type(b + i8)  # E: signedinteger
reveal_type(c + i8)  # E: complexfloating
reveal_type(f + i8)  # E: floating
reveal_type(i + i8)  # E: signedinteger
reveal_type(AR + i8)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(u8 + u8)  # E: unsignedinteger
reveal_type(i4 + u8)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(u4 + u8)  # E: unsignedinteger
reveal_type(b_ + u8)  # E: uint64
reveal_type(b + u8)  # E: unsignedinteger
reveal_type(c + u8)  # E: complexfloating
reveal_type(f + u8)  # E: floating
reveal_type(i + u8)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(AR + u8)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(i4 + i8)  # E: signedinteger
reveal_type(i4 + i4)  # E: signedinteger
reveal_type(i4 + i)  # E: signedinteger
reveal_type(i4 + b_)  # E: int32
reveal_type(i4 + b)  # E: signedinteger
reveal_type(i4 + AR)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(u4 + i8)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(u4 + i4)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(u4 + u8)  # E: unsignedinteger
reveal_type(u4 + u4)  # E: unsignedinteger
reveal_type(u4 + i)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(u4 + b_)  # E: uint32
reveal_type(u4 + b)  # E: unsignedinteger
reveal_type(u4 + AR)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(i8 + i4)  # E: signedinteger
reveal_type(i4 + i4)  # E: signedinteger
reveal_type(i + i4)  # E: signedinteger
reveal_type(b_ + i4)  # E: int32
reveal_type(b + i4)  # E: signedinteger
reveal_type(AR + i4)  # E: Union[numpy.ndarray, numpy.generic]

reveal_type(i8 + u4)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(i4 + u4)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(u8 + u4)  # E: unsignedinteger
reveal_type(u4 + u4)  # E: unsignedinteger
reveal_type(b_ + u4)  # E: uint32
reveal_type(b + u4)  # E: unsignedinteger
reveal_type(i + u4)  # E: Union[numpy.signedinteger, numpy.float64]
reveal_type(AR + u4)  # E: Union[numpy.ndarray, numpy.generic]
