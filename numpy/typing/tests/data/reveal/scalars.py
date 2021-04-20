import numpy as np

x = np.complex64(3 + 2j)

reveal_type(x.real)  # E: {float32}
reveal_type(x.imag)  # E: {float32}

reveal_type(x.real.real)  # E: {float32}
reveal_type(x.real.imag)  # E: {float32}

reveal_type(x.itemsize)  # E: int
reveal_type(x.shape)  # E: Tuple[]
reveal_type(x.strides)  # E: Tuple[]

reveal_type(x.ndim)  # E: Literal[0]
reveal_type(x.size)  # E: Literal[1]

reveal_type(x.squeeze())  # E: {complex64}
reveal_type(x.byteswap())  # E: {complex64}
reveal_type(x.transpose())  # E: {complex64}

reveal_type(x.dtype)  # E: numpy.dtype[{complex64}]

reveal_type(np.complex64().real)  # E: {float32}
reveal_type(np.complex128().imag)  # E: {float64}

reveal_type(np.unicode_('foo'))  # E: numpy.str_
reveal_type(np.str0('foo'))  # E: numpy.str_

# Aliases
reveal_type(np.unicode_())  # E: numpy.str_
reveal_type(np.str0())  # E: numpy.str_
reveal_type(np.bool8())  # E: numpy.bool_
reveal_type(np.bytes0())  # E: numpy.bytes_
reveal_type(np.string_())  # E: numpy.bytes_
reveal_type(np.object0())  # E: numpy.object_
reveal_type(np.void0(0))  # E: numpy.void

reveal_type(np.byte())  # E: {byte}
reveal_type(np.short())  # E: {short}
reveal_type(np.intc())  # E: {intc}
reveal_type(np.intp())  # E: {intp}
reveal_type(np.int0())  # E: {intp}
reveal_type(np.int_())  # E: {int_}
reveal_type(np.longlong())  # E: {longlong}

reveal_type(np.ubyte())  # E: {ubyte}
reveal_type(np.ushort())  # E: {ushort}
reveal_type(np.uintc())  # E: {uintc}
reveal_type(np.uintp())  # E: {uintp}
reveal_type(np.uint0())  # E: {uintp}
reveal_type(np.uint())  # E: {uint}
reveal_type(np.ulonglong())  # E: {ulonglong}

reveal_type(np.half())  # E: {half}
reveal_type(np.single())  # E: {single}
reveal_type(np.double())  # E: {double}
reveal_type(np.float_())  # E: {double}
reveal_type(np.longdouble())  # E: {longdouble}
reveal_type(np.longfloat())  # E: {longdouble}

reveal_type(np.csingle())  # E: {csingle}
reveal_type(np.singlecomplex())  # E: {csingle}
reveal_type(np.cdouble())  # E: {cdouble}
reveal_type(np.complex_())  # E: {cdouble}
reveal_type(np.cfloat())  # E: {cdouble}
reveal_type(np.clongdouble())  # E: {clongdouble}
reveal_type(np.clongfloat())  # E: {clongdouble}
reveal_type(np.longcomplex())  # E: {clongdouble}
