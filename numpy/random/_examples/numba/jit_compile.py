import numpy as np
import numba as nb
from numpy.random import PCG64
from timeit import timeit

bit_gen = PCG64()
rg = np.random.Generator(bit_gen)

def numpycall(rg):
    return rg.random(size=n)


# JIT compile Python function using Numba
numbacall = nb.njit(numpycall)

# Must use state address not state with numba
n = 10000

# Users can directly pass NumPy Generator objects into Numba
# compiled functions and access it's supported methods.
r1 = numbacall(rg)
r2 = numpycall(rg)

t1 = timeit(lambda: numbacall(rg), number=n)
print(f'{t1:.2f} secs for {n} PCG64 (Numba/PCG64) gaussian randoms')

t2 = timeit(lambda: numpycall(rg), number=n)
print(f'{t2:.2f} secs for {n} PCG64 (NumPy/PCG64) gaussian randoms')
