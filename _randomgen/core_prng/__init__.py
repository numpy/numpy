from .generator import RandomGenerator
from .mt19937 import MT19937
from .pcg64 import PCG64
from .splitmix64 import SplitMix64
from .threefry import ThreeFry
from .xoroshiro128 import Xoroshiro128
from .xorshift1024 import XorShift1024

__all__ = ['RandomGenerator', 'SplitMix64', 'PCG64', 'Xoroshiro128',
           'ThreeFry', 'MT19937', 'XorShift1024']

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
