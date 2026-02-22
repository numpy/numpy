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


class ExpandTabs(Benchmark):
    # Benchmark expandtabs for different string dtypes and tab layouts

    params = [
        [np.dtype("U256"), np.dtype("S256"), np.dtypes.StringDType()],
        [3, 11],
        [32, 2048],
    ]
    param_names = ["dtype", "tab_pattern", "size"]

    def setup(self, dtype, tab_pattern, size):
        base = self._make_sample(tab_pattern)
        if isinstance(dtype, np.dtypes.StringDType):
            fill_value = base
        elif dtype.kind == "S":
            fill_value = base.encode("ascii")
        else:
            fill_value = base
        self.arr = np.full(size, fill_value, dtype=dtype)
        self.tabsize = 4

    @staticmethod
    def _make_sample(tab_pattern):
        newline_pattern = 41
        letters = "abcdefghijklmnopqrstuvwxyz$%&."
        parts = []
        for idx in range(160):
            parts.append(letters[idx % len(letters)])
            if (idx + 1) % tab_pattern == 0:
                parts.append("\t")
            if (idx + 1) % newline_pattern == 0:
                parts.append("\n")
        return "".join(parts)

    def time_expandtabs(self, dtype, tab_pattern, size):
        np.strings.expandtabs(self.arr, tabsize=self.tabsize)
