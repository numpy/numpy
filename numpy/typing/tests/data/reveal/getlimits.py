import numpy as np
from numpy.typing import _32Bit

f: float
f8: np.float64
c8: np.complex64

i: int
i8: np.int64
u4: np.uint32

finfo_f8: np.finfo[np.float64]
iinfo_i8: np.iinfo[np.int64]
machar_f4: np.core.getlimits.MachArLike[_32Bit]

reveal_type(np.finfo(f))  # E: numpy.finfo[{double}]
reveal_type(np.finfo(f8))  # E: numpy.finfo[{float64}]
reveal_type(np.finfo(c8))  # E: numpy.finfo[{float32}]
reveal_type(np.finfo("f2"))  # E: numpy.finfo[numpy.floating[Any]]

reveal_type(finfo_f8.dtype)  # E: numpy.dtype[{float64}]
reveal_type(finfo_f8.bits)  # E: int
reveal_type(finfo_f8.eps)  # E: {float64}
reveal_type(finfo_f8.epsneg)  # E: {float64}
reveal_type(finfo_f8.iexp)  # E: int
reveal_type(finfo_f8.machep)  # E: int
reveal_type(finfo_f8.max)  # E: {float64}
reveal_type(finfo_f8.maxexp)  # E: int
reveal_type(finfo_f8.min)  # E: {float64}
reveal_type(finfo_f8.minexp)  # E: int
reveal_type(finfo_f8.negep)  # E: int
reveal_type(finfo_f8.nexp)  # E: int
reveal_type(finfo_f8.nmant)  # E: int
reveal_type(finfo_f8.precision)  # E: int
reveal_type(finfo_f8.resolution)  # E: {float64}
reveal_type(finfo_f8.tiny)  # E: {float64}
reveal_type(finfo_f8.machar)  # E: MachArLike[numpy.typing._64Bit]

reveal_type(np.iinfo(i))  # E: iinfo[{int_}]
reveal_type(np.iinfo(i8))  # E: iinfo[{int64}]
reveal_type(np.iinfo(u4))  # E: iinfo[{uint32}]
reveal_type(np.iinfo("i2"))  # E: iinfo[Any]

reveal_type(iinfo_i8.dtype)  # E: numpy.dtype[{int64}]
reveal_type(iinfo_i8.kind)  # E: str
reveal_type(iinfo_i8.bits)  # E: int
reveal_type(iinfo_i8.key)  # E: str
reveal_type(iinfo_i8.min)  # E: int
reveal_type(iinfo_i8.max)  # E: int

reveal_type(machar_f4.eps)  # E: numpy.ndarray[Any, numpy.dtype[{float32}]]
reveal_type(machar_f4.epsilon)  # E: numpy.ndarray[Any, numpy.dtype[{float32}]]
reveal_type(machar_f4.epsneg)  # E: numpy.ndarray[Any, numpy.dtype[{float32}]]
reveal_type(machar_f4.huge)  # E: numpy.ndarray[Any, numpy.dtype[{float32}]]
reveal_type(machar_f4.resolution)  # E: numpy.ndarray[Any, numpy.dtype[{float32}]]
reveal_type(machar_f4.tiny)  # E: numpy.ndarray[Any, numpy.dtype[{float32}]]
reveal_type(machar_f4.xmax)  # E: numpy.ndarray[Any, numpy.dtype[{float32}]]
reveal_type(machar_f4.xmin)  # E: numpy.ndarray[Any, numpy.dtype[{float32}]]
reveal_type(machar_f4.iexp)  # E: int
reveal_type(machar_f4.irnd)  # E: int
reveal_type(machar_f4.it)  # E: int
reveal_type(machar_f4.machep)  # E: int
reveal_type(machar_f4.maxexp)  # E: int
reveal_type(machar_f4.minexp)  # E: int
reveal_type(machar_f4.negep)  # E: int
reveal_type(machar_f4.ngrd)  # E: int
reveal_type(machar_f4.precision)  # E: int
reveal_type(machar_f4.ibeta)  # E: {int32}
reveal_type(machar_f4.title)  # E: str
