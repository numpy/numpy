from core_prng.generator import RandomGenerator
from core_prng.splitmix64 import SplitMix64
from core_prng.xoroshiro128 import Xoroshiro128

print(RandomGenerator().random_integer())
print(RandomGenerator(Xoroshiro128()).random_integer())
x = Xoroshiro128()
print(x.get_state())
print(RandomGenerator(SplitMix64()).random_integer())
print(RandomGenerator(SplitMix64()).random_integer())
print(RandomGenerator(SplitMix64(1)).random_integer())
print(RandomGenerator(SplitMix64([1.0, 2.0])).random_integer())
