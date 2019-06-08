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


class Sort(Benchmark):
    def setup(self):
        self.e = np.arange(10000, dtype=np.float32)
        self.o = np.arange(10001, dtype=np.float32)
        np.random.seed(25)
        np.random.shuffle(self.o)
        # quicksort implementations can have issues with equal elements
        self.equal = np.ones(10000)
        self.many_equal = np.sort(np.arange(10000) % 10)

    def time_sort(self):
        np.sort(self.e)

    def time_sort_random(self):
        np.sort(self.o)

    def time_sort_inplace(self):
        self.e.sort()

    def time_sort_equal(self):
        self.equal.sort()

    def time_sort_many_equal(self):
        self.many_equal.sort()

    def time_argsort(self):
        self.e.argsort()

    def time_argsort_random(self):
        self.o.argsort()


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
