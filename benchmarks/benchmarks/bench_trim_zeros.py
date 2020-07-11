from .common import Benchmark

import numpy as np


class TrimZeros(Benchmark):
    def setup(self):
        self.ar_float = np.hstack([
            np.zeros(100_000),
            np.random.uniform(size=100_000),
            np.zeros(100_000),
        ])

        dtype = [('a', 'int64'), ('b', 'float64'), ('c', bool)]
        self.ar_void = self.ar_float.astype(dtype)
        self.ar_str = self.ar_float.astype(str)

    def time_trim_zeros_float(self):
        np.trim_zeros(self.ar_float)

    def time_trim_zeros_void(self):
        np.trim_zeros(self.ar_void)

    def time_trim_zeros_str(self):
        np.trim_zeros(self.ar_str)
