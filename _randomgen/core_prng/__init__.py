from .dsfmt import DSFMT
from .generator import RandomGenerator
from .mt19937 import MT19937
from .pcg64 import PCG64
from .philox import Philox
from .threefry import ThreeFry
from .threefry32 import ThreeFry32
from .xoroshiro128 import Xoroshiro128
from .xorshift1024 import Xorshift1024

__all__ = ['RandomGenerator', 'DSFMT', 'MT19937', 'PCG64', 'Philox',
           'ThreeFry', 'ThreeFry32', 'Xoroshiro128', 'Xorshift1024']

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
