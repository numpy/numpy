import numpy as np

x = np.complex64(3 + 2j)

reveal_type(x.real)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(x.imag)  # E: numpy.floating[numpy.typing._32Bit]

reveal_type(x.real.real)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(x.real.imag)  # E: numpy.floating[numpy.typing._32Bit]

reveal_type(x.itemsize)  # E: int
reveal_type(x.shape)  # E: tuple[builtins.int]
reveal_type(x.strides)  # E: tuple[builtins.int]

reveal_type(np.complex64().real)  # E: numpy.floating[numpy.typing._32Bit]
reveal_type(np.complex128().imag)  # E: numpy.floating[numpy.typing._64Bit]

reveal_type(np.unicode_('foo'))  # E: numpy.str_
reveal_type(np.str0('foo'))  # E: numpy.str_
