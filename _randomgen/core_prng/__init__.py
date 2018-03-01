from .generator import RandomGenerator
from .xoroshiro128 import Xoroshiro128
from .splitmix64 import SplitMix64

__all__ = ['RandomGenerator', 'SplitMix64', 'Xoroshiro128']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
