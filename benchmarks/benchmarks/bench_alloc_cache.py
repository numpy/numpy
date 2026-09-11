"""Benchmarks for the NumPy small-allocation cache.

NumPy caches data allocations smaller than 1024 bytes (up to 7 per size
bucket) to avoid repeated malloc/free calls.  For float64 arrays this
means arrays with fewer than 128 elements hit the cache.

These benchmarks measure tight create-and-discard loops so that the
allocator cache is exercised on every iteration after the first.
"""

import numpy as np

from .common import Benchmark


class SmallArrayCreation(Benchmark):
    # Sizes chosen so that data bytes = size * 8 (float64).
    # Cached:   1..127  →  8..1016 bytes  (< 1024, hits the cache)
    # Uncached: 128+    →  1024+ bytes    (bypasses the cache)
    params = [[1, 4, 16, 64, 127, 128, 256, 512]]
    param_names = ['size']
    timeout = 60

    def setup(self, size):
        self.dtype = np.float64

    def time_empty_loop(self, size):
        dt = self.dtype
        for _ in range(10_000):
            np.empty(size, dtype=dt)

    def time_full_loop(self, size):
        dt = self.dtype
        for _ in range(10_000):
            np.full(size, 1.0, dtype=dt)

    def time_ones_loop(self, size):
        dt = self.dtype
        for _ in range(10_000):
            np.ones(size, dtype=dt)

    def time_zeros_loop(self, size):
        dt = self.dtype
        for _ in range(10_000):
            np.zeros(size, dtype=dt)
