"""Benchmarks for `numpy.lib`."""


from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np


class Pad(Benchmark):
    """Benchmarks for `numpy.pad`."""

    param_names = ["shape", "pad_width", "mode"]
    params = [
        [(1000,),
         (10, 100),
         (10, 10, 10),
         (64, 64),
         (512, 512),
         (10, 512, 512)],
        [1, 3, (0, 5)],
        ["constant", "edge", "linear_ramp",
         "mean", "median", "minimum", "maximum",
         "reflect", "symmetric", 'wrap']
    ]

    def setup(self, shape, pad_width, mode):
        # This should not be zeros or empty as the OS
        # might defer page faulting to the time we pad the array
        self.array = np.ones(shape)

    def time_pad(self, shape, pad_width, mode):
        _ = np.pad(self.array, pad_width, mode)
