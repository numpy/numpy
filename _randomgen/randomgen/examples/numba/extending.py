import datetime as dt

import numpy as np
import numba as nb

from randomgen import Xoroshiro128

x = Xoroshiro128()
f = x.ctypes.next_uint32
s = x.ctypes.state


@nb.jit(nopython=True)
def bounded_uint(lb, ub, state):
    mask = delta = ub - lb
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16

    val = f(state) & mask
    while val > delta:
        val = f(state) & mask

    return lb + val


bounded_uint(323, 2394691, s.value)


@nb.jit(nopython=True)
def bounded_uints(lb, ub, n, state):
    out = np.empty(n, dtype=np.uint32)
    for i in range(n):
        out[i] = bounded_uint(lb, ub, state)


bounded_uints(323, 2394691, 10000000, s.value)

g = x.cffi.next_double
cffi_state = x.cffi.state
state_addr = x.cffi.state_address


def normals(n, state):
    out = np.empty(n)
    for i in range((n + 1) // 2):
        x1 = 2.0 * g(state) - 1.0
        x2 = 2.0 * g(state) - 1.0
        r2 = x1 * x1 + x2 * x2
        while r2 >= 1.0 or r2 == 0.0:
            x1 = 2.0 * g(state) - 1.0
            x2 = 2.0 * g(state) - 1.0
            r2 = x1 * x1 + x2 * x2
        f = np.sqrt(-2.0 * np.log(r2) / r2)
        out[2 * i] = f * x1
        if 2 * i + 1 < n:
            out[2 * i + 1] = f * x2
    return out


print(normals(10, cffi_state).var())
# Warm up
normalsj = nb.jit(normals, nopython=True)
normalsj(1, state_addr)

start = dt.datetime.now()
normalsj(1000000, state_addr)
ms = 1000 * (dt.datetime.now() - start).total_seconds()
print('1,000,000 Polar-transform (numba/Xoroshiro128) randoms in '
      '{ms:0.1f}ms'.format(ms=ms))

start = dt.datetime.now()
np.random.standard_normal(1000000)
ms = 1000 * (dt.datetime.now() - start).total_seconds()
print('1,000,000 Polar-transform (NumPy) randoms in {ms:0.1f}ms'.format(ms=ms))
