from typing import Any
import numpy as np

f8 = np.float64()
i8 = np.int64()
u8 = np.uint64()

f4 = np.float32()
i4 = np.int32()
u4 = np.uint32()

td = np.timedelta64(0, "D")
b_ = np.bool_()

b = bool()
f = float()
i = int()

AR_b: np.ndarray[Any, np.dtype[np.bool_]]
AR_m: np.ndarray[Any, np.dtype[np.timedelta64]]

# Time structures

reveal_type(td % td)  # E: timedelta64
reveal_type(AR_m % td)  # E: Any
reveal_type(td % AR_m)  # E: Any

reveal_type(divmod(td, td))  # E: tuple[{int64}, timedelta64]
reveal_type(divmod(AR_m, td))  # E: tuple[ndarray[Any, dtype[signedinteger[typing._64Bit]]], ndarray[Any, dtype[timedelta64]]]
reveal_type(divmod(td, AR_m))  # E: tuple[ndarray[Any, dtype[signedinteger[typing._64Bit]]], ndarray[Any, dtype[timedelta64]]]

# Bool

reveal_type(b_ % b)  # E: {int8}
reveal_type(b_ % i)  # E: {int_}
reveal_type(b_ % f)  # E: {float64}
reveal_type(b_ % b_)  # E: {int8}
reveal_type(b_ % i8)  # E: {int64}
reveal_type(b_ % u8)  # E: {uint64}
reveal_type(b_ % f8)  # E: {float64}
reveal_type(b_ % AR_b)  # E: ndarray[Any, dtype[{int8}]]

reveal_type(divmod(b_, b))  # E: tuple[{int8}, {int8}]
reveal_type(divmod(b_, i))  # E: tuple[{int_}, {int_}]
reveal_type(divmod(b_, f))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(b_, b_))  # E: tuple[{int8}, {int8}]
reveal_type(divmod(b_, i8))  # E: tuple[{int64}, {int64}]
reveal_type(divmod(b_, u8))  # E: tuple[{uint64}, {uint64}]
reveal_type(divmod(b_, f8))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(b_, AR_b))  # E: ndarray[Any, dtype[{int8}]], ndarray[Any, dtype[{int8}]]]

reveal_type(b % b_)  # E: {int8}
reveal_type(i % b_)  # E: {int_}
reveal_type(f % b_)  # E: {float64}
reveal_type(b_ % b_)  # E: {int8}
reveal_type(i8 % b_)  # E: {int64}
reveal_type(u8 % b_)  # E: {uint64}
reveal_type(f8 % b_)  # E: {float64}
reveal_type(AR_b % b_)  # E: ndarray[Any, dtype[{int8}]]

reveal_type(divmod(b, b_))  # E: tuple[{int8}, {int8}]
reveal_type(divmod(i, b_))  # E: tuple[{int_}, {int_}]
reveal_type(divmod(f, b_))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(b_, b_))  # E: tuple[{int8}, {int8}]
reveal_type(divmod(i8, b_))  # E: tuple[{int64}, {int64}]
reveal_type(divmod(u8, b_))  # E: tuple[{uint64}, {uint64}]
reveal_type(divmod(f8, b_))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(AR_b, b_))  # E: ndarray[Any, dtype[{int8}]], ndarray[Any, dtype[{int8}]]]

# int

reveal_type(i8 % b)  # E: {int64}
reveal_type(i8 % f)  # E: {float64}
reveal_type(i8 % i8)  # E: {int64}
reveal_type(i8 % f8)  # E: {float64}
reveal_type(i4 % i8)  # E: signedinteger[Union[_32Bit, _64Bit]]
reveal_type(i4 % f8)  # E: floating[Union[_64Bit, _32Bit]]
reveal_type(i4 % i4)  # E: {int32}
reveal_type(i4 % f4)  # E: {float32}
reveal_type(i8 % AR_b)  # E: ndarray[Any, dtype[signedinteger[Any]]]

reveal_type(divmod(i8, b))  # E: tuple[{int64}, {int64}]
reveal_type(divmod(i8, f))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(i8, i8))  # E: tuple[{int64}, {int64}]
reveal_type(divmod(i8, f8))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(i8, i4))  # E: tuple[signedinteger[Union[_64Bit, _32Bit]], signedinteger[Union[_64Bit, _32Bit]]]
reveal_type(divmod(i8, f4))  # E: tuple[floating[Union[_32Bit, _64Bit]], floating[Union[_32Bit, _64Bit]]]
reveal_type(divmod(i4, i4))  # E: tuple[{int32}, {int32}]
reveal_type(divmod(i4, f4))  # E: tuple[{float32}, {float32}]
reveal_type(divmod(i8, AR_b))  # E: tuple[ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]]]

reveal_type(b % i8)  # E: {int64}
reveal_type(f % i8)  # E: {float64}
reveal_type(i8 % i8)  # E: {int64}
reveal_type(f8 % i8)  # E: {float64}
reveal_type(i8 % i4)  # E: signedinteger[Union[_64Bit, _32Bit]]
reveal_type(f8 % i4)  # E: floating[Union[_64Bit, _32Bit]]
reveal_type(i4 % i4)  # E: {int32}
reveal_type(f4 % i4)  # E: {float32}
reveal_type(AR_b % i8)  # E: ndarray[Any, dtype[signedinteger[Any]]]

reveal_type(divmod(b, i8))  # E: tuple[{int64}, {int64}]
reveal_type(divmod(f, i8))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(i8, i8))  # E: tuple[{int64}, {int64}]
reveal_type(divmod(f8, i8))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(i4, i8))  # E: tuple[signedinteger[Union[_32Bit, _64Bit]], signedinteger[Union[_32Bit, _64Bit]]]
reveal_type(divmod(f4, i8))  # E: tuple[floating[Union[_32Bit, _64Bit]], floating[Union[_32Bit, _64Bit]]]
reveal_type(divmod(i4, i4))  # E: tuple[{int32}, {int32}]
reveal_type(divmod(f4, i4))  # E: tuple[{float32}, {float32}]
reveal_type(divmod(AR_b, i8))  # E: tuple[ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]]]

# float

reveal_type(f8 % b)  # E: {float64}
reveal_type(f8 % f)  # E: {float64}
reveal_type(i8 % f4)  # E: floating[Union[_32Bit, _64Bit]]
reveal_type(f4 % f4)  # E: {float32}
reveal_type(f8 % AR_b)  # E: ndarray[Any, dtype[floating[Any]]]

reveal_type(divmod(f8, b))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(f8, f))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(f8, f8))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(f8, f4))  # E: tuple[floating[Union[_64Bit, _32Bit]], floating[Union[_64Bit, _32Bit]]]
reveal_type(divmod(f4, f4))  # E: tuple[{float32}, {float32}]
reveal_type(divmod(f8, AR_b))  # E: tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]

reveal_type(b % f8)  # E: {float64}
reveal_type(f % f8)  # E: {float64}
reveal_type(f8 % f8)  # E: {float64}
reveal_type(f8 % f8)  # E: {float64}
reveal_type(f4 % f4)  # E: {float32}
reveal_type(AR_b % f8)  # E: ndarray[Any, dtype[floating[Any]]]

reveal_type(divmod(b, f8))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(f, f8))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(f8, f8))  # E: tuple[{float64}, {float64}]
reveal_type(divmod(f4, f8))  # E: tuple[floating[Union[_32Bit, _64Bit]], floating[Union[_32Bit, _64Bit]]]
reveal_type(divmod(f4, f4))  # E: tuple[{float32}, {float32}]
reveal_type(divmod(AR_b, f8))  # E: tuple[ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
