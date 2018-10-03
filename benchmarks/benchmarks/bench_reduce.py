from __future__ import absolute_import, division, print_function

from .common import Benchmark, TYPES1, get_squares

import numpy as np


class AddReduce(Benchmark):
    def setup(self):
        self.squares = get_squares().values()

    def time_axis_0(self):
        [np.add.reduce(a, axis=0) for a in self.squares]

    def time_axis_1(self):
        [np.add.reduce(a, axis=1) for a in self.squares]


class AddReduceSeparate(Benchmark):
    params = [[0, 1], TYPES1]
    param_names = ['axis', 'type']

    def setup(self, axis, typename):
        self.a = get_squares()[typename]

    def time_reduce(self, axis, typename):
        np.add.reduce(self.a, axis=axis)


class AnyAll(Benchmark):
    def setup(self):
        # avoid np.zeros's lazy allocation that would
        # cause page faults during benchmark
        self.zeros = np.full(100000, 0, bool)
        self.ones = np.full(100000, 0, bool)

    def time_all_fast(self):
        self.zeros.all()

    def time_all_slow(self):
        self.ones.all()

    def time_any_fast(self):
        self.ones.any()

    def time_any_slow(self):
        self.zeros.any()


class MinMax(Benchmark):
    params = [np.float32, np.float64, np.intp]
    param_names = ['dtype']

    def setup(self, dtype):
        self.d = np.ones(20000, dtype=dtype)

    def time_min(self, dtype):
        np.min(self.d)

    def time_max(self, dtype):
        np.max(self.d)


class SmallReduction(Benchmark):
    def setup(self):
        self.d = np.ones(100, dtype=np.float32)

    def time_small(self):
        np.sum(self.d)
