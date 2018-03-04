from core_prng.generator import RandomGenerator

from core_prng import SplitMix64, Xoroshiro128, ThreeFry, MT19937, XorShift1024, PCG64

print(RandomGenerator().random_integer(32))
print(RandomGenerator(Xoroshiro128()).random_integer())
print(RandomGenerator(ThreeFry()).random_integer())
print(RandomGenerator(SplitMix64()).random_integer())
print(RandomGenerator(SplitMix64()).random_integer())
print(RandomGenerator(SplitMix64(1)).random_integer())
print(RandomGenerator(SplitMix64([1.0, 2.0])).random_integer())

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

rg = RandomGenerator(XorShift1024())
state = rg.state
print(state)
rg.state = state

rg = RandomGenerator(PCG64())
state = rg.state
print(state)
rg.state = state
