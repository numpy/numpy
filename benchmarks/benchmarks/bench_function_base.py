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


class SortGenerator(object):
    # The size of the unsorted area in the "random unsorted area"
    # benchmarks
    AREA_SIZE = 10
    # The size of the "partially ordered" sub-arrays
    BUBBLE_SIZE = 10

    @staticmethod
    def random(size, dtype):
        arr = np.arange(size, dtype=dtype)
        np.random.shuffle(arr)
        return arr
    
    @staticmethod
    def ordered(size, dtype):
        return np.arange(size, dtype=dtype)

    @staticmethod
    def reversed(size, dtype):
        return np.arange(size-1, -1, -1, dtype=dtype)

    @staticmethod
    def uniform(size, dtype):
        return np.ones(size, dtype=dtype)

    @staticmethod
    def _type_swapped_pair(size, dtype, swap_frac):
        a = np.arange(size, dtype=dtype)
        for _ in range(int(size * swap_frac)):
            x, y = np.random.randint(0, size, 2)
            a[x], a[y] = a[y], a[x]
        return a

    @staticmethod
    def _type_sorted_block(size, dtype, block_size):
        a = np.arange(size, dtype=dtype)
        b = []
        if size < block_size:
            return a
        block_num = size // block_size
        for i in range(block_num):
            b.extend(a[i::block_num])
        return np.array(b)

    @staticmethod
    def _type_random_unsorted_area(size, dtype, frac, area_num):
        area_num = int(area_num)
        a = np.arange(size, dtype=dtype)
        unsorted_len = int(size * frac / area_num)
        for _ in range(area_num):
            start = np.random.randint(size-unsorted_len)
            end = start + unsorted_len
            np.random.shuffle(a[start:end])
        return a

    @classmethod
    def get_array(cls, array_type, size, dtype):
        import re
        import functools
        token_specification = [
            (r'sorted\_block\_([0-9]+)',
                lambda size, dtype, x: cls._type_sorted_block(size, dtype, x)),
            (r'swapped\_pair\_([0-9]+)\_percent',
                lambda size, dtype, x: cls._type_swapped_pair(size, dtype, x / 100.0)),
            (r'random\_unsorted\_area\_([0-9]+)\_percent',
                lambda size, dtype, x: cls._type_random_unsorted_area(size, dtype, x / 100.0, cls.AREA_SIZE)),
            (r'random_bubble\_([0-9]+)\_fold',
                lambda size, dtype, x: cls._type_random_unsorted_area(size, dtype, x * cls.BUBBLE_SIZE / size, cls.BUBBLE_SIZE)),
        ]

        for pattern, function in token_specification:
            match = re.fullmatch(pattern, array_type)

            if match is not None:
                return function(size, dtype, int(match.group(1)))

        raise ValueError("Incorrect array_type specified.")


class Sort(Benchmark):
    """
    This benchmark tests sorting performance with several
    different types of arrays that are likely to appear in
    real-world applications.

    ordered
        Perfectly ordered array
    reversed
        A reversed array
    uniform
        A array where all values are the same
    sorted_block_X
        An array that is entirely composed of sorted blocks
        of size X
    swapped_pair_X_percent
        An array which has random pairs swapped. The number
        of swapped pairs is X% of the size of the array
    random_unsorted_area_X_percent
        This kind of array has random unsorted areas which
        take up X% of the original array, the size of one
        area is SortGenerator.AREA_SIZE
    random_bubble_X_fold
        Same as random_unsorted_area, except that it has
        X areas of size SortGenerator.BUBBLE_SIZE
    """
    params = [
        # In NumPy 1.17 and newer, 'merge' can be one of several
        # stable sorts, it isn't necessarily merge sort.
        ['quick', 'merge', 'heap'],
        ['float64', 'int64', 'uint64'],
        [
            'ordered',
            'reversed',
            'uniform',
            'sorted_block_10',
            'sorted_block_100',
            'sorted_block_1000',
            'swapped_pair_1_percent',
            'swapped_pair_10_percent',
            'swapped_pair_50_percent',
            'random_unsorted_area_50_percent',
            'random_unsorted_area_10_percent',
            'random_unsorted_area_1_percent',
            'random_bubble_1_fold',
            'random_bubble_5_fold',
            'random_bubble_10_fold',
        ],
    ]
    param_names = ['kind', 'dtype', 'array_type']

    # The size of the benchmarked arrays.
    ARRAY_SIZE = 10000

    def setup(self, kind, dtype, array_type):
        np.random.seed(1234)
        self.arr = SortGenerator.get_array(array_type, self.ARRAY_SIZE, dtype)

    def time_sort_inplace(self, kind, dtype, array_type):
        self.arr.sort(kind=kind)

    def time_sort(self, kind, dtype, array_type):
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

    # Retain old benchmark name for backward compatability
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
