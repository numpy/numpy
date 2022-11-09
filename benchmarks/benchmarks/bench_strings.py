import operator

import numpy as np

from .common import Benchmark

_OPERATORS = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
}


class StringComparisons(Benchmark):
    # Basic string comparison speed tests
    params = [
        [100, 10000, (1000, 20)],
        ['U', 'S'],
        [True, False],
        ['==', '!=', '<', '<=', '>', '>=']]
    param_names = ['shape', 'dtype', 'contig', 'operator']
    int64 = np.dtype(np.int64)

    def setup(self, shape, dtype, contig, operator):
        self.arr = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
        self.arr_identical = self.arr.copy()
        self.arr_different = self.arr[::-1].copy()

        if not contig:
            self.arr = self.arr[..., ::2]
            self.arr_identical = self.arr_identical[..., ::2]
            self.arr_different = self.arr_different[..., ::2]

        self.operator = _OPERATORS[operator]

    def time_compare_identical(self, shape, dtype, contig, operator):
        self.operator(self.arr, self.arr_identical)

    def time_compare_different(self, shape, dtype, contig, operator):
        self.operator(self.arr, self.arr_different)
