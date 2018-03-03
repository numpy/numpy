from .generator import RandomGenerator
from .mt19937 import MT19937
from .splitmix64 import SplitMix64
from .threefry import ThreeFry
from .xoroshiro128 import Xoroshiro128

PRNGS = {'SplitMix64': SplitMix64,
         'ThreeFry': ThreeFry,
         'MT19937': MT19937,
         'Xoroshiro128': Xoroshiro128}


def __generator_ctor(prng_name='mt19937'):
    """
    Pickling helper function that returns a mod_name.RandomState object

    Parameters
    ----------
    prng_name: str
        String containing the core PRNG

    Returns
    -------
    rg: RandomGenerator
        RandomGenerator using the named core PRNG
    """
    try:
        prng_name = prng_name.decode('ascii')
    except AttributeError:
        pass
    if prng_name in PRNGS:
        prng = PRNGS[prng_name]
    else:
        raise ValueError(str(prng_name) + ' is not a known PRNG module.')

    return RandomGenerator(prng())


def __prng_ctor(prng_name='mt19937'):
    """
    Pickling helper function that returns a mod_name.RandomState object

    Parameters
    ----------
    prng_name: str
        String containing the core PRNG

    Returns
    -------
    prng: CorePRNG
        Core PRNG instance
    """
    try:
        prng_name = prng_name.decode('ascii')
    except AttributeError:
        pass
    if prng_name in PRNGS:
        prng = PRNGS[prng_name]
    else:
        raise ValueError(str(prng_name) + ' is not a known PRNG module.')

    return prng()
