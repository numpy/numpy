from core_prng.generator import RandomGenerator
from core_prng.splitmix64 import SplitMix64
from core_prng.xoroshiro128 import Xoroshiro128

print(RandomGenerator().random_integer())
print(RandomGenerator(Xoroshiro128()).random_integer())
print(RandomGenerator(SplitMix64()).random_integer())
