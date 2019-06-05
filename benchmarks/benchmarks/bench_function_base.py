from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np


class Histogram1D(Benchmark):
    def setup(self):
        self.d = np.linspace(0, 100, 100000)

    def time_full_coverage(self):
        np.histogram(self.d, 200, (0, 100))

    def time_small_coverage(self):
        np.histogram(self.d, 200, (50, 51))

    def time_fine_binning(self):
        np.histogram(self.d, 10000, (0, 100))


class Histogram2D(Benchmark):
    def setup(self):
        self.d = np.linspace(0, 100, 200000).reshape((-1,2))

    def time_full_coverage(self):
        np.histogramdd(self.d, (200, 200), ((0, 100), (0, 100)))

    def time_small_coverage(self):
        np.histogramdd(self.d, (200, 200), ((50, 51), (50, 51)))

    def time_fine_binning(self):
        np.histogramdd(self.d, (10000, 10000), ((0, 100), (0, 100)))


class Bincount(Benchmark):
    def setup(self):
        self.d = np.arange(80000, dtype=np.intp)
        self.e = self.d.astype(np.float64)

    def time_bincount(self):
        np.bincount(self.d)

    def time_weights(self):
        np.bincount(self.d, weights=self.e)


class Median(Benchmark):
    def setup(self):
        self.e = np.arange(10000, dtype=np.float32)
        self.o = np.arange(10001, dtype=np.float32)

    def time_even(self):
        np.median(self.e)

    def time_odd(self):
        np.median(self.o)

    def time_even_inplace(self):
        np.median(self.e, overwrite_input=True)

    def time_odd_inplace(self):
        np.median(self.o, overwrite_input=True)

    def time_even_small(self):
        np.median(self.e[:500], overwrite_input=True)

    def time_odd_small(self):
        np.median(self.o[:500], overwrite_input=True)


class Percentile(Benchmark):
    def setup(self):
        self.e = np.arange(10000, dtype=np.float32)
        self.o = np.arange(10001, dtype=np.float32)

    def time_quartile(self):
        np.percentile(self.e, [25, 75])

    def time_percentile(self):
        np.percentile(self.e, [25, 35, 55, 65, 75])


class Select(Benchmark):
    def setup(self):
        self.d = np.arange(20000)
        self.e = self.d.copy()
        self.cond = [(self.d > 4), (self.d < 2)]
        self.cond_large = [(self.d > 4), (self.d < 2)] * 10

    def time_select(self):
        np.select(self.cond, [self.d, self.e])

    def time_select_larger(self):
        np.select(self.cond_large, ([self.d, self.e] * 10))


def memoize(f):
    _memoized = {}
    def wrapped(*args):
        if args not in _memoized:
            _memoized[args] = f(*args)
        
        return _memoized[args].copy()

    return f


class SortGenerator(object):
    # The size of the unsorted area in the "random unsorted area"
    # benchmarks
    AREA_SIZE = 100
    # The size of the "partially ordered" sub-arrays
    BUBBLE_SIZE = 100

    @staticmethod
    @memoize
    def random(size, dtype):
        """
        Returns a randomly-shuffled array.
        """
        arr = np.arange(size, dtype=dtype)
        np.random.shuffle(arr)
        return arr
    
    @staticmethod
    @memoize
    def ordered(size, dtype):
        """
        Returns an ordered array.
        """
        return np.arange(size, dtype=dtype)

    @staticmethod
    @memoize
    def reversed(size, dtype):
        """
        Returns an array that's in descending order.
        """
        return np.arange(size-1, -1, -1, dtype=dtype)

    @staticmethod
    @memoize
    def uniform(size, dtype):
        """
        Returns an array that has the same value everywhere.
        """
        return np.ones(size, dtype=dtype)

    @staticmethod
    @memoize
    def swapped_pair(size, dtype, swap_frac):
        """
        Returns an ordered array, but one that has ``swap_frac * size``
        pairs swapped.
        """
        a = np.arange(size, dtype=dtype)
        for _ in range(int(size * swap_frac)):
            x, y = np.random.randint(0, size, 2)
            a[x], a[y] = a[y], a[x]
        return a

    @staticmethod
    @memoize
    def sorted_block(size, dtype, block_size):
        """
        Returns an array with blocks that are all sorted.
        """
        a = np.arange(size, dtype=dtype)
        b = []
        if size < block_size:
            return a
        block_num = size // block_size
        for i in range(block_num):
            b.extend(a[i::block_num])
        return np.array(b)

    @classmethod
    @memoize
    def random_unsorted_area(cls, size, dtype, frac, area_size=None):
        """
        This type of array has random unsorted areas such that they
        compose the fraction ``frac`` of the original array.
        """
        if area_size is None:
            area_size = cls.AREA_SIZE

        area_num = int(size * frac / area_size)
        a = np.arange(size, dtype=dtype)
        for _ in range(area_num):
            start = np.random.randint(size-area_size)
            end = start + area_size
            np.random.shuffle(a[start:end])
        return a

    @classmethod
    @memoize
    def random_bubble(cls, size, dtype, bubble_num, bubble_size=None):
        """
        This type of array has ``bubble_num`` random unsorted areas.
        """
        if bubble_size is None:
            bubble_size = cls.BUBBLE_SIZE
        frac = bubble_size * bubble_num / size

        return cls.random_unsorted_area(size, dtype, frac, bubble_size)


class Sort(Benchmark):
    """
    This benchmark tests sorting performance with several
    different types of arrays that are likely to appear in
    real-world applications.
    """
    params = [
        # In NumPy 1.17 and newer, 'merge' can be one of several
        # stable sorts, it isn't necessarily merge sort.
        ['quick', 'merge', 'heap'],
        ['float64', 'int64', 'int16'],
        [
            ('random',),
            ('ordered',),
            ('reversed',),
            ('uniform',),
            ('sorted_block', 10),
            ('sorted_block', 100),
            ('sorted_block', 1000),
            # ('swapped_pair', 0.01),
            # ('swapped_pair', 0.1),
            # ('swapped_pair', 0.5),
            # ('random_unsorted_area', 0.5),
            # ('random_unsorted_area', 0.1),
            # ('random_unsorted_area', 0.01),
            # ('random_bubble', 1),
            # ('random_bubble', 5),
            # ('random_bubble', 10),
        ],
    ]
    param_names = ['kind', 'dtype', 'array_type']

    # The size of the benchmarked arrays.
    ARRAY_SIZE = 10000

    def setup(self, kind, dtype, array_type):
        np.random.seed(1234)
        array_class = array_type[0]
        self.arr = getattr(SortGenerator, array_class)(self.ARRAY_SIZE, dtype, *array_type[1:])

    def time_sort(self, kind, dtype, array_type):
        # Using np.sort(...) instead of arr.sort(...) because it makes a copy.
        # This is important because the data is prepared once per benchmark, but
        # used across multiple runs.
        np.sort(self.arr, kind=kind)

    def time_argsort(self, kind, dtype, array_type):
        np.argsort(self.arr, kind=kind)


class SortWorst(Benchmark):
    def setup(self):
        # quicksort median of 3 worst case
        self.worst = np.arange(1000000)
        x = self.worst
        while x.size > 3:
            mid = x.size // 2
            x[mid], x[-2] = x[-2], x[mid]
            x = x[:-2]

    def time_sort_worst(self):
        np.sort(self.worst)

    # Retain old benchmark name for backward compatibility
    time_sort_worst.benchmark_name = "bench_function_base.Sort.time_sort_worst"


class Where(Benchmark):
    def setup(self):
        self.d = np.arange(20000)
        self.e = self.d.copy()
        self.cond = (self.d > 5000)

    def time_1(self):
        np.where(self.cond)

    def time_2(self):
        np.where(self.cond, self.d, self.e)

    def time_2_broadcast(self):
        np.where(self.cond, self.d, 0)
