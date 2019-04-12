from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np

from numpy.random import RandomState, RandomGenerator

class Random(Benchmark):
    params = ['normal', 'uniform', 'weibull 1', 'binomial 10 0.5',
              'poisson 10']

    def setup(self, name):
        items = name.split()
        name = items.pop(0)
        params = [float(x) for x in items]

        self.func = getattr(np.random, name)
        self.params = tuple(params) + ((100, 100),)

    def time_rng(self, name):
        self.func(*self.params)


class Shuffle(Benchmark):
    def setup(self):
        self.a = np.arange(100000)

    def time_100000(self):
        np.random.shuffle(self.a)


class Randint(Benchmark):

    def time_randint_fast(self):
        """Compare to uint32 below"""
        np.random.randint(0, 2**30, size=10**5)

    def time_randint_slow(self):
        """Compare to uint32 below"""
        np.random.randint(0, 2**30 + 1, size=10**5)


class Randint_dtype(Benchmark):
    high = {
        'bool': 1,
        'uint8': 2**7,
        'uint16': 2**15,
        'uint32': 2**31,
        'uint64': 2**63
        }

    param_names = ['dtype']
    params = ['bool', 'uint8', 'uint16', 'uint32', 'uint64']

    def setup(self, name):
        from numpy.lib import NumpyVersion
        if NumpyVersion(np.__version__) < '1.11.0.dev0':
            raise NotImplementedError

    def time_randint_fast(self, name):
        high = self.high[name]
        np.random.randint(0, high, size=10**5, dtype=name)

    def time_randint_slow(self, name):
        high = self.high[name]
        np.random.randint(0, high + 1, size=10**5, dtype=name)


class Permutation(Benchmark):
    def setup(self):
        self.n = 10000
        self.a_1d = np.random.random_sample(self.n)
        self.a_2d = np.random.random_sample((self.n, 2))
    
    def time_permutation_1d(self):
        np.random.permutation(self.a_1d)

    def time_permutation_2d(self):
        np.random.permutation(self.a_2d)        

    def time_permutation_int(self):
        np.random.permutation(self.n)

nom_size = 100000

class RNG(Benchmark):
    param_names = ['rng']
    params = ['DSFMT', 'PCG64', 'PCG32', 'MT19937', 'Xoroshiro128',
              'Xorshift1024', 'Xoshiro256StarStar', 'Xoshiro512StarStar',
              'Philox', 'ThreeFry', 'ThreeFry32', 'numpy']

    def setup(self, brng):
        if brng == 'numpy':
            self.rg = np.random.RandomState()
        else:
            self.rg = RandomGenerator(getattr(np.random, brng)())
        self.rg.random_sample()
        self.int32info = np.iinfo(np.int32)
        self.uint32info = np.iinfo(np.uint32)
        self.uint64info = np.iinfo(np.uint64)
    
    def time_raw(self, brng):
        if brng == 'numpy':
            self.rg.random_integers(self.int32info.max, size=nom_size)
        else:
            self.rg.random_integers(self.int32info.max, size=nom_size)

    def time_32bit(self, brng):
        min, max = self.uint32info.min, self.uint32info.max
        self.rg.randint(min, max + 1, nom_size, dtype=np.uint32)

    def time_64bit(self, brng):
        min, max = self.uint64info.min, self.uint64info.max
        self.rg.randint(min, max + 1, nom_size, dtype=np.uint64)

    def time_normal_zig(self, brng):
        self.rg.standard_normal(nom_size)

class Bounded(Benchmark):
    u8 = np.uint8
    u16 = np.uint16
    u32 = np.uint32
    u64 = np.uint64
    param_names = ['rng', 'dt_max_masked']
    params = [['DSFMT', 'PCG64', 'PCG32', 'MT19937', 'Xoroshiro128',
               'Xorshift1024', 'Xoshiro256StarStar', 'Xoshiro512StarStar',
               'Philox', 'ThreeFry', 'ThreeFry32', 'numpy'],
              [[u8,  95, True],
               [u8,  64, False],  # Worst case for legacy
               [u8,  95, False],   # Typ. avg. case for legacy
               [u8, 127, False],  # Best case for legacy
               [u16,   95, True],
               [u16, 1024, False],  # Worst case for legacy
               [u16, 1535, False],   # Typ. avg. case for legacy
               [u16, 2047, False],  # Best case for legacy
               [u32,   95, True],
               [u32, 1024, False],  # Worst case for legacy
               [u32, 1535, False],   # Typ. avg. case for legacy
               [u32, 2047, False],  # Best case for legacy
               [u64,   95, True],
               [u64, 1024, False],  # Worst case for legacy
               [u64, 1535, False],   # Typ. avg. case for legacy
               [u64, 2047, False],  # Best case for legacy
             ]]

    def setup(self, brng, args):
        if brng == 'numpy':
            self.rg = np.random.RandomState()
        else:
            self.rg = RandomGenerator(getattr(np.random, brng)())
        self.rg.random_sample()

    def time_bounded(self, brng, args):
            """
            Timer for 8-bit bounded values.

            Parameters (packed as args)
            ----------
            dt : {uint8, uint16, uint32, unit64}
                output dtype
            max : int
                Upper bound for range. Lower is always 0.  Must be <= 2**bits.
            use_masked: bool
                If True, masking and rejection sampling is used to generate a random
                number in an interval. If False, Lemire's algorithm is used if
                available to generate a random number in an interval.

            Notes
            -----
            Lemire's algorithm has improved performance when max+1 is not a
            power of two.
            """
            dt, max, use_masked = args
            if brng == 'numpy':
                self.rg.randint(0, max + 1, nom_size, dtype=dt)
            else:
                self.rg.randint(0, max + 1, nom_size, dtype=dt,
                                use_masked=use_masked)
        
