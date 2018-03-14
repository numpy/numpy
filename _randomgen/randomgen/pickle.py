from .generator import RandomGenerator
from .dsfmt import DSFMT
from .mt19937 import MT19937
from .pcg32 import PCG32
from .pcg64 import PCG64
from .philox import Philox
from .threefry import ThreeFry
from .threefry32 import ThreeFry32
from .xoroshiro128 import Xoroshiro128
from .xorshift1024 import Xorshift1024

PRNGS = {'MT19937': MT19937,
         'DSFMT': DSFMT,
         'PCG32': PCG32,
         'PCG64': PCG64,
         'Philox': Philox,
         'ThreeFry': ThreeFry,
         'ThreeFry32': ThreeFry32,
         'Xorshift1024': Xorshift1024,
         'Xoroshiro128': Xoroshiro128}


def __generator_ctor(brng_name='mt19937'):
    """
    Pickling helper function that returns a RandomGenerator object

    Parameters
    ----------
    brng_name: str
        String containing the core PRNG

    Returns
    -------
    rg: RandomGenerator
        RandomGenerator using the named core PRNG
    """
    try:
        brng_name = brng_name.decode('ascii')
    except AttributeError:
        pass
    if brng_name in PRNGS:
        brng = PRNGS[brng_name]
    else:
        raise ValueError(str(brng_name) + ' is not a known PRNG module.')

    return RandomGenerator(brng())


def __brng_ctor(brng_name='mt19937'):
    """
    Pickling helper function that returns a basic RNG object

    Parameters
    ----------
    brng_name: str
        String containing the name of the Basic RNG

    Returns
    -------
    brng: BasicRNG
        Basic RNG instance
    """
    try:
        brng_name = brng_name.decode('ascii')
    except AttributeError:
        pass
    if brng_name in PRNGS:
        brng = PRNGS[brng_name]
    else:
        raise ValueError(str(brng_name) + ' is not a known PRNG module.')

    return brng()
