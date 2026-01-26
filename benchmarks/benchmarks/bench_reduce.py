import numpy as np

from .common import TYPES1, Benchmark, get_squares


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
        self.ones = np.full(100000, 1, bool)

    def time_all_fast(self):
        self.zeros.all()

    def time_all_slow(self):
        self.ones.all()

    def time_any_fast(self):
        self.ones.any()

    def time_any_slow(self):
        self.zeros.any()


class StatsReductions(Benchmark):
    params = ['int64', 'uint64', 'float32', 'float64', 'complex64', 'bool_'],
    param_names = ['dtype']

    def setup(self, dtype):
        self.data = np.ones(200, dtype=dtype)
        if dtype.startswith('complex'):
            self.data = self.data * self.data.T * 1j

    def time_min(self, dtype):
        np.min(self.data)

    def time_max(self, dtype):
        np.max(self.data)

    def time_mean(self, dtype):
        np.mean(self.data)

    def time_std(self, dtype):
        np.std(self.data)

    def time_prod(self, dtype):
        np.prod(self.data)

    def time_var(self, dtype):
        np.var(self.data)


class FMinMax(Benchmark):
    params = [np.float32, np.float64]
    param_names = ['dtype']

    def setup(self, dtype):
        self.d = np.ones(20000, dtype=dtype)

    def time_min(self, dtype):
        np.fmin.reduce(self.d)

    def time_max(self, dtype):
        np.fmax.reduce(self.d)


class ArgMax(Benchmark):
    params = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
              np.int64, np.uint64, np.float32, np.float64, bool]
    param_names = ['dtype']

    def setup(self, dtype):
        self.d = np.zeros(200000, dtype=dtype)

    def time_argmax(self, dtype):
        np.argmax(self.d)


class ArgMin(Benchmark):
    params = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
              np.int64, np.uint64, np.float32, np.float64, bool]
    param_names = ['dtype']

    def setup(self, dtype):
        self.d = np.ones(200000, dtype=dtype)

    def time_argmin(self, dtype):
        np.argmin(self.d)


class SmallReduction(Benchmark):
    def setup(self):
        self.d = np.ones(100, dtype=np.float32)

    def time_small(self):
        np.sum(self.d)
