from core_prng import Xoroshiro128
import numpy as np
import numba as nb

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
        bounded_uint(lb, ub, state)


bounded_uints(323, 2394691, 10000000, s.value)
