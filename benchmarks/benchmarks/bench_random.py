import numpy as np

from .common import Benchmark

try:
    from numpy.random import Generator
except ImportError:
    pass


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
        self.a_1d = np.random.random(self.n)
        self.a_2d = np.random.random((self.n, 2))

    def time_permutation_1d(self):
        np.random.permutation(self.a_1d)

    def time_permutation_2d(self):
        np.random.permutation(self.a_2d)

    def time_permutation_int(self):
        np.random.permutation(self.n)


nom_size = 100000

class RNG(Benchmark):
    param_names = ['rng']
    params = ['PCG64', 'MT19937', 'Philox', 'SFC64', 'numpy']

    def setup(self, bitgen):
        if bitgen == 'numpy':
            self.rg = np.random.RandomState()
        else:
            self.rg = Generator(getattr(np.random, bitgen)())
        self.rg.random()
        self.int32info = np.iinfo(np.int32)
        self.uint32info = np.iinfo(np.uint32)
        self.uint64info = np.iinfo(np.uint64)

    def time_raw(self, bitgen):
        if bitgen == 'numpy':
            self.rg.random_integers(self.int32info.max, size=nom_size)
        else:
            self.rg.integers(self.int32info.max, size=nom_size, endpoint=True)

    def time_32bit(self, bitgen):
        min, max = self.uint32info.min, self.uint32info.max
        if bitgen == 'numpy':
            self.rg.randint(min, max + 1, nom_size, dtype=np.uint32)
        else:
            self.rg.integers(min, max + 1, nom_size, dtype=np.uint32)

    def time_64bit(self, bitgen):
        min, max = self.uint64info.min, self.uint64info.max
        if bitgen == 'numpy':
            self.rg.randint(min, max + 1, nom_size, dtype=np.uint64)
        else:
            self.rg.integers(min, max + 1, nom_size, dtype=np.uint64)

    def time_normal_zig(self, bitgen):
        self.rg.standard_normal(nom_size)

class Bounded(Benchmark):
    u8 = np.uint8
    u16 = np.uint16
    u32 = np.uint32
    u64 = np.uint64
    param_names = ['rng', 'dt_max']
    params = [['PCG64', 'MT19937', 'Philox', 'SFC64', 'numpy'],
              [[u8,    95],
               [u8,    64],  # Worst case for legacy
               [u8,   127],  # Best case for legacy
               [u16,   95],
               [u16, 1024],  # Worst case for legacy
               [u16, 1535],   # Typ. avg. case for legacy
               [u16, 2047],  # Best case for legacy
               [u32, 1024],  # Worst case for legacy
               [u32, 1535],   # Typ. avg. case for legacy
               [u32, 2047],  # Best case for legacy
               [u64,   95],
               [u64, 1024],  # Worst case for legacy
               [u64, 1535],   # Typ. avg. case for legacy
               [u64, 2047],  # Best case for legacy
             ]]

    def setup(self, bitgen, args):
        seed = 707250673
        if bitgen == 'numpy':
            self.rg = np.random.RandomState(seed)
        else:
            self.rg = Generator(getattr(np.random, bitgen)(seed))
        self.rg.random()

    def time_bounded(self, bitgen, args):
        """
        Timer for 8-bit bounded values.

        Parameters (packed as args)
        ----------
        dt : {uint8, uint16, uint32, unit64}
              output dtype
        max : int
              Upper bound for range. Lower is always 0.  Must be <= 2**bits.
        """
        dt, max = args
        if bitgen == 'numpy':
            self.rg.randint(0, max + 1, nom_size, dtype=dt)
        else:
            self.rg.integers(0, max + 1, nom_size, dtype=dt)

class Choice(Benchmark):
    params = [1e3, 1e6, 1e8]

    def setup(self, v):
        self.a = np.arange(v)
        self.rng = np.random.default_rng()

    def time_legacy_choice(self, v):
        np.random.choice(self.a, 1000, replace=False)

    def time_choice(self, v):
        self.rng.choice(self.a, 1000, replace=False)
