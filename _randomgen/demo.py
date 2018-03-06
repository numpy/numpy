import timeit

from core_prng import Xoroshiro128, ThreeFry, MT19937, \
    Xorshift1024, PCG64, Philox, DSFMT
from core_prng.generator import RandomGenerator

print(RandomGenerator().random_integer(32))
print(RandomGenerator(Xoroshiro128()).random_integer())
print(RandomGenerator(ThreeFry()).random_integer())

print('\n' * 3)
print('Check random_sample')
rg = RandomGenerator()
print(rg.state)
print(rg.random_sample())
print(rg.state)
print(rg.random_sample())
print(rg.random_sample((3)))
print(rg.random_sample((3, 1)))
print(rg.state)
import numpy as np

a = rg.random_sample((1, 1), dtype=np.float32)
print(a)
print(a.dtype)
print(rg.state)

print('\n' * 3)
print('Check set/get state')
state = rg.state
print(rg.state)
print(rg.random_integer())
print(rg.state)
rg.state = state
print(rg.random_integer())

print(RandomGenerator(Xoroshiro128()).state)
rg = RandomGenerator(ThreeFry())
print(rg.state)
rg.random_integer()
print(rg.state)
rg = RandomGenerator(MT19937())
state = rg.state
print(state)
rg.state = state
print(rg.random_integer())
print(rg.random_integer(32))
print(rg.random_sample())

rg = RandomGenerator(Xorshift1024())
state = rg.state
print(state)
rg.state = state

rg = RandomGenerator(PCG64())
state = rg.state
print(state)
rg.state = state

rg = RandomGenerator(Philox())
state = rg.state
print(state)
rg.state = state

rg = RandomGenerator(DSFMT())
state = rg.state
print(state)
rg.state = state


PRNGS = [MT19937, PCG64, Philox, ThreeFry, Xoroshiro128, Xorshift1024, DSFMT]

setup = """
from core_prng import {module}
m = {module}()
m._benchmark(701)
"""
import pandas as pd
res = []
for p in PRNGS:
    module = p.__name__
    print(module)
    t = timeit.timeit("m._benchmark(10000000)", setup.format(module=module),
                      number=10)
    res.append(pd.Series({'module': module, 'ms': 1000 *
                          t / 10, 'rps': int(10000000 / (t/10))}))
    #print('{:0.2f} ms'.format())
    # print('{:,} randoms per second'.format()))
res = pd.DataFrame(res)
print(res.set_index('module').sort_values('ms'))

res = []
for p in PRNGS:
    module = p.__name__
    print(module)
    t = timeit.timeit("m._benchmark(10000000, 'double')", setup.format(module=module),
                      number=10)
    res.append(pd.Series({'module': module, 'ms': 1000 *
                          t / 10, 'rps': int(10000000 / (t/10))}))
    #print('{:0.2f} ms'.format())
    # print('{:,} randoms per second'.format()))
res = pd.DataFrame(res)
print(res.set_index('module').sort_values('ms'))
