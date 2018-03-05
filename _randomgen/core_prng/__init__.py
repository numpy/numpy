from .generator import RandomGenerator
from .mt19937 import MT19937
from .pcg64 import PCG64
from .threefry import ThreeFry
from .xoroshiro128 import Xoroshiro128
from .xorshift1024 import Xorshift1024
from .philox import Philox

__all__ = ['RandomGenerator', 'PCG64', 'Xoroshiro128',
           'ThreeFry', 'MT19937', 'Xorshift1024']

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
