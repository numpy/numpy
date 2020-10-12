import numpy as np

i8 = np.int64(1)
u8 = np.uint64(1)

i4 = np.int32(1)
u4 = np.uint32(1)

b_ = np.bool_(1)

b = bool(1)
i = int(1)

AR = np.array([0, 1, 2], dtype=np.int32)
AR.setflags(write=False)


reveal_type(i8 << i8)  # E: numpy.signedinteger
reveal_type(i8 >> i8)  # E: numpy.signedinteger
reveal_type(i8 | i8)  # E: numpy.signedinteger
reveal_type(i8 ^ i8)  # E: numpy.signedinteger
reveal_type(i8 & i8)  # E: numpy.signedinteger

reveal_type(i8 << AR)  # E: Union[numpy.ndarray, numpy.integer]
reveal_type(i8 >> AR)  # E: Union[numpy.ndarray, numpy.integer]
reveal_type(i8 | AR)  # E: Union[numpy.ndarray, numpy.integer, numpy.bool_]
reveal_type(i8 ^ AR)  # E: Union[numpy.ndarray, numpy.integer, numpy.bool_]
reveal_type(i8 & AR)  # E: Union[numpy.ndarray, numpy.integer, numpy.bool_]

reveal_type(i4 << i4)  # E: numpy.signedinteger
reveal_type(i4 >> i4)  # E: numpy.signedinteger
reveal_type(i4 | i4)  # E: numpy.signedinteger
reveal_type(i4 ^ i4)  # E: numpy.signedinteger
reveal_type(i4 & i4)  # E: numpy.signedinteger

reveal_type(i8 << i4)  # E: numpy.signedinteger
reveal_type(i8 >> i4)  # E: numpy.signedinteger
reveal_type(i8 | i4)  # E: numpy.signedinteger
reveal_type(i8 ^ i4)  # E: numpy.signedinteger
reveal_type(i8 & i4)  # E: numpy.signedinteger

reveal_type(i8 << i)  # E: numpy.signedinteger
reveal_type(i8 >> i)  # E: numpy.signedinteger
reveal_type(i8 | i)  # E: numpy.signedinteger
reveal_type(i8 ^ i)  # E: numpy.signedinteger
reveal_type(i8 & i)  # E: numpy.signedinteger

reveal_type(i8 << b_)  # E: numpy.int64
reveal_type(i8 >> b_)  # E: numpy.int64
reveal_type(i8 | b_)  # E: numpy.int64
reveal_type(i8 ^ b_)  # E: numpy.int64
reveal_type(i8 & b_)  # E: numpy.int64

reveal_type(i8 << b)  # E: numpy.signedinteger
reveal_type(i8 >> b)  # E: numpy.signedinteger
reveal_type(i8 | b)  # E: numpy.signedinteger
reveal_type(i8 ^ b)  # E: numpy.signedinteger
reveal_type(i8 & b)  # E: numpy.signedinteger

reveal_type(u8 << u8)  # E: numpy.unsignedinteger
reveal_type(u8 >> u8)  # E: numpy.unsignedinteger
reveal_type(u8 | u8)  # E: numpy.unsignedinteger
reveal_type(u8 ^ u8)  # E: numpy.unsignedinteger
reveal_type(u8 & u8)  # E: numpy.unsignedinteger

reveal_type(u8 << AR)  # E: Union[numpy.ndarray, numpy.integer]
reveal_type(u8 >> AR)  # E: Union[numpy.ndarray, numpy.integer]
reveal_type(u8 | AR)  # E: Union[numpy.ndarray, numpy.integer, numpy.bool_]
reveal_type(u8 ^ AR)  # E: Union[numpy.ndarray, numpy.integer, numpy.bool_]
reveal_type(u8 & AR)  # E: Union[numpy.ndarray, numpy.integer, numpy.bool_]

reveal_type(u4 << u4)  # E: numpy.unsignedinteger
reveal_type(u4 >> u4)  # E: numpy.unsignedinteger
reveal_type(u4 | u4)  # E: numpy.unsignedinteger
reveal_type(u4 ^ u4)  # E: numpy.unsignedinteger
reveal_type(u4 & u4)  # E: numpy.unsignedinteger

reveal_type(u4 << i4)  # E: numpy.signedinteger
reveal_type(u4 >> i4)  # E: numpy.signedinteger
reveal_type(u4 | i4)  # E: numpy.signedinteger
reveal_type(u4 ^ i4)  # E: numpy.signedinteger
reveal_type(u4 & i4)  # E: numpy.signedinteger

reveal_type(u4 << i)  # E: numpy.signedinteger
reveal_type(u4 >> i)  # E: numpy.signedinteger
reveal_type(u4 | i)  # E: numpy.signedinteger
reveal_type(u4 ^ i)  # E: numpy.signedinteger
reveal_type(u4 & i)  # E: numpy.signedinteger

reveal_type(u8 << b_)  # E: numpy.uint64
reveal_type(u8 >> b_)  # E: numpy.uint64
reveal_type(u8 | b_)  # E: numpy.uint64
reveal_type(u8 ^ b_)  # E: numpy.uint64
reveal_type(u8 & b_)  # E: numpy.uint64

reveal_type(u8 << b)  # E: numpy.unsignedinteger
reveal_type(u8 >> b)  # E: numpy.unsignedinteger
reveal_type(u8 | b)  # E: numpy.unsignedinteger
reveal_type(u8 ^ b)  # E: numpy.unsignedinteger
reveal_type(u8 & b)  # E: numpy.unsignedinteger

reveal_type(b_ << b_)  # E: numpy.int8
reveal_type(b_ >> b_)  # E: numpy.int8
reveal_type(b_ | b_)  # E: numpy.bool_
reveal_type(b_ ^ b_)  # E: numpy.bool_
reveal_type(b_ & b_)  # E: numpy.bool_

reveal_type(b_ << AR)  # E: Union[numpy.ndarray, numpy.integer]
reveal_type(b_ >> AR)  # E: Union[numpy.ndarray, numpy.integer]
reveal_type(b_ | AR)  # E: Union[numpy.ndarray, numpy.integer, numpy.bool_]
reveal_type(b_ ^ AR)  # E: Union[numpy.ndarray, numpy.integer, numpy.bool_]
reveal_type(b_ & AR)  # E: Union[numpy.ndarray, numpy.integer, numpy.bool_]

reveal_type(b_ << b)  # E: numpy.int8
reveal_type(b_ >> b)  # E: numpy.int8
reveal_type(b_ | b)  # E: numpy.bool_
reveal_type(b_ ^ b)  # E: numpy.bool_
reveal_type(b_ & b)  # E: numpy.bool_

reveal_type(b_ << i)  # E: Union[numpy.int32, numpy.int64]
reveal_type(b_ >> i)  # E: Union[numpy.int32, numpy.int64]
reveal_type(b_ | i)  # E: Union[numpy.int32, numpy.int64]
reveal_type(b_ ^ i)  # E: Union[numpy.int32, numpy.int64]
reveal_type(b_ & i)  # E: Union[numpy.int32, numpy.int64]

reveal_type(~i8)  # E: numpy.int64
reveal_type(~i4)  # E: numpy.int32
reveal_type(~u8)  # E: numpy.uint64
reveal_type(~u4)  # E: numpy.uint32
reveal_type(~b_)  # E: numpy.bool_
reveal_type(~AR)  # E: Union[numpy.ndarray*, numpy.integer, numpy.bool_]
