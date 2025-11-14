import numpy as np

from .common import Benchmark


class ArrayCoercionSmall(Benchmark):
    # More detailed benchmarks for array coercion,
    # some basic benchmarks are in `bench_core.py`.
    params = [[range(3), [1], 1, np.array([5], dtype=np.int64), np.int64(5)]]
    param_names = ['array_like']
    int64 = np.dtype(np.int64)

    def time_array_invalid_kwarg(self, array_like):
        try:
            np.array(array_like, ndmin="not-integer")
        except TypeError:
            pass

    def time_array(self, array_like):
        np.array(array_like)

    def time_array_dtype_not_kwargs(self, array_like):
        np.array(array_like, self.int64)

    def time_array_no_copy(self, array_like):
        np.array(array_like, copy=None)

    def time_array_subok(self, array_like):
        np.array(array_like, subok=True)

    def time_array_all_kwargs(self, array_like):
        np.array(array_like, dtype=self.int64, copy=None, order="F",
                 subok=False, ndmin=2)

    def time_asarray(self, array_like):
        np.asarray(array_like)

    def time_asarray_dtype(self, array_like):
        np.asarray(array_like, dtype=self.int64)

    def time_asarray_dtype_order(self, array_like):
        np.asarray(array_like, dtype=self.int64, order="F")

    def time_asanyarray(self, array_like):
        np.asanyarray(array_like)

    def time_asanyarray_dtype(self, array_like):
        np.asanyarray(array_like, dtype=self.int64)

    def time_asanyarray_dtype_order(self, array_like):
        np.asanyarray(array_like, dtype=self.int64, order="F")

    def time_ascontiguousarray(self, array_like):
        np.ascontiguousarray(array_like)
