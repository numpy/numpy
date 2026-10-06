from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np

from .common import TYPES1, Benchmark


def _repeat_array(rows, width, dtype):
    value = {
        "float64": np.float64(3.14),
        "string": "repeat benchmark value",
        "object": {1, 2, 3, 4, 5, 6, 7},
    }[dtype]
    return np.full((rows, width), value)


class Repeat(Benchmark):
    # With width=1, rows=250 and 251 produce 500 and 502 output
    # elements, respectively, straddling the GIL-release threshold.
    # float64 widths 1 and 8 exercise 8-byte and 64-byte chunks,
    # covering both the specialized and fallback copy paths.
    params = [[16, 250, 251, 125000], [1, 8],
              ["float64", "string", "object"]]
    param_names = ["rows", "width", "dtype"]

    def setup(self, rows, width, dtype):
        self.arr = _repeat_array(rows, width, dtype)

    def time_repeat(self, rows, width, dtype):
        self.arr.repeat(2, axis=0)


class RepeatThreads(Benchmark):
    """Time the same two repeat calls using one or two workers."""

    params = [[1, 2], [1, 8], ["float64", "string", "object"]]
    param_names = ["workers", "width", "dtype"]

    def setup(self, workers, width, dtype):
        self.arrays = [_repeat_array(125000, width, dtype) for _ in range(2)]
        self.pool = ThreadPoolExecutor(max_workers=workers)
        # Force every worker to start before we begin timing.
        barrier = Barrier(workers)
        for _ in self.pool.map(lambda _: barrier.wait(), range(workers)):
            pass

    @staticmethod
    def _repeat(arr):
        arr.repeat(2, axis=0)

    def time_repeat(self, workers, width, dtype):
        for _ in self.pool.map(self._repeat, self.arrays):
            pass

    def teardown(self, workers, width, dtype):
        self.pool.shutdown(wait=True)


class Take(Benchmark):
    params = [
        [(1000, 1), (2, 1000, 1), (1000, 3)],
        ["raise", "wrap", "clip"],
        TYPES1 + ["O", "i,O"]]
    param_names = ["shape", "mode", "dtype"]

    def setup(self, shape, mode, dtype):
        self.arr = np.ones(shape, dtype)
        self.indices = np.arange(1000)

    def time_contiguous(self, shape, mode, dtype):
        self.arr.take(self.indices, axis=-2, mode=mode)


class PutMask(Benchmark):
    params = [
        [True, False],
        TYPES1 + ["O", "i,O"]]
    param_names = ["values_is_scalar", "dtype"]

    def setup(self, values_is_scalar, dtype):
        if values_is_scalar:
            self.vals = np.array(1., dtype=dtype)
        else:
            self.vals = np.ones(1000, dtype=dtype)

        self.arr = np.ones(1000, dtype=dtype)

        self.dense_mask = np.ones(1000, dtype="bool")
        self.sparse_mask = np.zeros(1000, dtype="bool")

    def time_dense(self, values_is_scalar, dtype):
        np.putmask(self.arr, self.dense_mask, self.vals)

    def time_sparse(self, values_is_scalar, dtype):
        np.putmask(self.arr, self.sparse_mask, self.vals)


class Put(Benchmark):
    params = [
        [True, False],
        TYPES1 + ["O", "i,O"]]
    param_names = ["values_is_scalar", "dtype"]

    def setup(self, values_is_scalar, dtype):
        if values_is_scalar:
            self.vals = np.array(1., dtype=dtype)
        else:
            self.vals = np.ones(1000, dtype=dtype)

        self.arr = np.ones(1000, dtype=dtype)
        self.indx = np.arange(1000, dtype=np.intp)

    def time_ordered(self, values_is_scalar, dtype):
        np.put(self.arr, self.indx, self.vals)
