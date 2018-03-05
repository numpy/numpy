from core_prng import Xoroshiro128, ThreeFry, MT19937, \
    Xorshift1024, PCG64, Philox
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
