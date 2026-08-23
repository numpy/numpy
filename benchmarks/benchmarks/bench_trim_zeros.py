import numpy as np

from .common import Benchmark

_FLOAT = np.dtype('float64')
_COMPLEX = np.dtype('complex128')
_INT = np.dtype('int64')
_BOOL = np.dtype('bool')


class TrimZeros(Benchmark):
    param_names = ["dtype", "size"]
    params = [
        [_INT, _FLOAT, _COMPLEX, _BOOL],
        [3000, 30_000, 300_000]
    ]

    def setup(self, dtype, size):
        n = size // 3
        self.array = np.hstack([
            np.zeros(n),
            np.random.uniform(size=n),
            np.zeros(n),
        ]).astype(dtype)

    def time_trim_zeros(self, dtype, size):
        np.trim_zeros(self.array)
