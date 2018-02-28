from core_prng.generator import RandomGenerator
from core_prng.splitmix64 import SplitMix64
from core_prng.xoroshiro128 import Xoroshiro128

print(RandomGenerator().random_integer())
print(RandomGenerator(Xoroshiro128()).random_integer())
print(RandomGenerator(SplitMix64()).random_integer())
print(RandomGenerator(SplitMix64()).random_integer())
print(RandomGenerator(SplitMix64(1)).random_integer())
print(RandomGenerator(SplitMix64([1.0, 2.0])).random_integer())


print('\n'*3)
print('Check random_sample')
rg = RandomGenerator()
print(rg.random_sample())
print(rg.random_sample((3)))
print(rg.random_sample((3,1)))

print('\n'*3)
print('Check set/get state')
state = rg.state
print(rg.state)
print(rg.random_integer())
rg.state = state
print(rg.random_integer())
