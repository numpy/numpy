import numpy as np

x = np.complex64(3 + 2j)

reveal_type(x.real)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(x.imag)  # E: numpy.floating[numpy.typing._32Bit]

reveal_type(x.real.real)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(x.real.imag)  # E: numpy.floating[numpy.typing._32Bit]

reveal_type(x.itemsize)  # E: int
reveal_type(x.shape)  # E: Tuple[]
reveal_type(x.strides)  # E: Tuple[]

reveal_type(x.ndim)  # E: Literal[0]
reveal_type(x.size)  # E: Literal[1]

reveal_type(x.squeeze())  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(x.byteswap())  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]
reveal_type(x.transpose())  # E: numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]

reveal_type(x.dtype)  # E: numpy.dtype[numpy.complexfloating[numpy.typing._32Bit, numpy.typing._32Bit]]

reveal_type(np.complex64().real)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(np.complex128().imag)  # E: numpy.floating[numpy.typing._64Bit]

reveal_type(np.unicode_('foo'))  # E: numpy.str_
reveal_type(np.str0('foo'))  # E: numpy.str_

# Aliases
reveal_type(np.unicode_())  # E: numpy.str_
reveal_type(np.str0())  # E: numpy.str_

reveal_type(np.byte())  # E: numpy.signedinteger[numpy.typing._
reveal_type(np.short())  # E: numpy.signedinteger[numpy.typing._
reveal_type(np.intc())  # E: numpy.signedinteger[numpy.typing._
reveal_type(np.intp())  # E: numpy.signedinteger[numpy.typing._
reveal_type(np.int0())  # E: numpy.signedinteger[numpy.typing._
reveal_type(np.int_())  # E: numpy.signedinteger[numpy.typing._
reveal_type(np.longlong())  # E: numpy.signedinteger[numpy.typing._

reveal_type(np.ubyte())  # E: numpy.unsignedinteger[numpy.typing._
reveal_type(np.ushort())  # E: numpy.unsignedinteger[numpy.typing._
reveal_type(np.uintc())  # E: numpy.unsignedinteger[numpy.typing._
reveal_type(np.uintp())  # E: numpy.unsignedinteger[numpy.typing._
reveal_type(np.uint0())  # E: numpy.unsignedinteger[numpy.typing._
reveal_type(np.uint())  # E: numpy.unsignedinteger[numpy.typing._
reveal_type(np.ulonglong())  # E: numpy.unsignedinteger[numpy.typing._

reveal_type(np.half())  # E: numpy.floating[numpy.typing._
reveal_type(np.single())  # E: numpy.floating[numpy.typing._
reveal_type(np.double())  # E: numpy.floating[numpy.typing._
reveal_type(np.float_())  # E: numpy.floating[numpy.typing._
reveal_type(np.longdouble())  # E: numpy.floating[numpy.typing._
reveal_type(np.longfloat())  # E: numpy.floating[numpy.typing._

reveal_type(np.csingle())  # E: numpy.complexfloating[numpy.typing._
reveal_type(np.singlecomplex())  # E: numpy.complexfloating[numpy.typing._
reveal_type(np.cdouble())  # E: numpy.complexfloating[numpy.typing._
reveal_type(np.complex_())  # E: numpy.complexfloating[numpy.typing._
reveal_type(np.cfloat())  # E: numpy.complexfloating[numpy.typing._
reveal_type(np.clongdouble())  # E: numpy.complexfloating[numpy.typing._
reveal_type(np.clongfloat())  # E: numpy.complexfloating[numpy.typing._
reveal_type(np.longcomplex())  # E: numpy.complexfloating[numpy.typing._
