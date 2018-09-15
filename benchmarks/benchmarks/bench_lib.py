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
         # 50 * 512 * 512 = 13 million points = 46 MB. should be a good
         # out of cache describing a typical usecase
         (50, 512, 512)],
        [1,
         3,
         (0, 5)],
        ["constant",
         "edge", "linear_ramp",
         # mean/median/minimum/maximum all use the same code path
         "mean",
         # reflect/symmetric share alot of the code path
         "reflect",
         "wrap"],
    ]

    def setup(self, shape, pad_width, mode):
        # Make sure to fill the array to make the OS page fault
        # in the setup phase and not the timed phase
        self.array = np.full(shape, fill_value=1)

    def time_pad(self, shape, pad_width, mode):
        np.pad(self.array, pad_width, mode)
