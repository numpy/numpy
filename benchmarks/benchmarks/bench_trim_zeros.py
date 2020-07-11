from .common import Benchmark

import numpy as np

_FLOAT = np.dtype('float64')
_COMPLEX = np.dtype('complex128')
_INT = np.dtype('int64')
_BOOL = np.dtype('bool')
_VOID = np.dtype([('a', 'int64'), ('b', 'float64'), ('c', bool)])
_STR = np.dtype('U32')
_BYTES = np.dtype('S32')


class TrimZeros(Benchmark):
    param_names = ["dtype", "size"]
    params = [
        [_INT, _FLOAT, _COMPLEX, _BOOL, _STR, _BYTES, _VOID],
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
