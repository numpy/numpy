import numpy as np

from .common import Benchmark


class ArrayEqual(Benchmark):
    """Guard for the fused ``np.array_equal`` reduction of gh-32465."""

    param_names = ["dtype", "difference", "layout"]
    params = [
        ["int8", "int64", "float64", "complex128"],
        ["equal", "first", "middle"],
        ["1d", "2d", "strided"],
    ]

    def setup(self, dtype, difference, layout):
        size = 1_000_000
        a = (np.arange(size) % 251).astype(dtype)
        b = a.copy()
        if difference != "equal":
            b[0 if difference == "first" else size // 2] += 1
        if layout == "2d":
            a = a.reshape(1000, 1000)
            b = b.reshape(1000, 1000)
        elif layout == "strided":
            a = np.repeat(a, 2)[::2]
            b = np.repeat(b, 2)[::2]
        self.a, self.b = a, b

    def time_array_equal(self, dtype, difference, layout):
        np.array_equal(self.a, self.b)

    def time_array_equal_nan(self, dtype, difference, layout):
        np.array_equal(self.a, self.b, equal_nan=True)
