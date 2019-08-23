from __future__ import absolute_import, division, print_function

from .common import Benchmark

try:
    from numpy.core.overrides import array_function_dispatch
except ImportError:
    # Don't fail at import time with old Numpy versions
    def array_function_dispatch(*args, **kwargs):
        def wrap(*args, **kwargs):
            return None
        return wrap

import numpy as np


def _broadcast_to_dispatcher(array, shape, subok=None):
    return (array,)


@array_function_dispatch(_broadcast_to_dispatcher)
def mock_broadcast_to(array, shape, subok=False):
    pass


def _concatenate_dispatcher(arrays, axis=None, out=None):
    if out is not None:
        arrays = list(arrays)
        arrays.append(out)
    return arrays


@array_function_dispatch(_concatenate_dispatcher)
def mock_concatenate(arrays, axis=0, out=None):
    pass


class DuckArray(object):
    def __array_function__(self, func, types, args, kwargs):
        pass


class ArrayFunction(Benchmark):

    def setup(self):
        self.numpy_array = np.array(1)
        self.numpy_arrays = [np.array(1), np.array(2)]
        self.many_arrays = 500 * self.numpy_arrays
        self.duck_array = DuckArray()
        self.duck_arrays = [DuckArray(), DuckArray()]
        self.mixed_arrays = [np.array(1), DuckArray()]

    def time_mock_broadcast_to_numpy(self):
        mock_broadcast_to(self.numpy_array, ())

    def time_mock_broadcast_to_duck(self):
        mock_broadcast_to(self.duck_array, ())

    def time_mock_concatenate_numpy(self):
        mock_concatenate(self.numpy_arrays, axis=0)

    def time_mock_concatenate_many(self):
        mock_concatenate(self.many_arrays, axis=0)

    def time_mock_concatenate_duck(self):
        mock_concatenate(self.duck_arrays, axis=0)

    def time_mock_concatenate_mixed(self):
        mock_concatenate(self.mixed_arrays, axis=0)
