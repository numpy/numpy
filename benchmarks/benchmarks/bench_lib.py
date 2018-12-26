"""Benchmarks for `numpy.lib`."""


from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np


class Pad(Benchmark):
    """Benchmarks for `numpy.pad`.

    When benchmarking the pad function it is useful to cover scenarios where
    the ratio between the size of the input array and the output array differs
    significantly (original area vs. padded area). This allows to evaluate for
    which scenario a padding algorithm is optimized. Furthermore involving
    large range of array sizes ensures that the effects of CPU-bound caching is
    visible.

    The table below shows the sizes of the arrays involved in this benchmark:

    +-----------------+----------+-----------+-----------+-----------------+
    | shape           | original | padded: 1 | padded: 8 | padded: (0, 32) |
    +=================+==========+===========+===========+=================+
    | (2 ** 22,)      | 32 MiB   | 32.0 MiB  | 32.0 MiB  | 32.0 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (1024, 1024)    | 8 MiB    | 8.03 MiB  | 8.25 MiB  | 8.51 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (256, 256, 1)   | 256 KiB  | 786 KiB   | 5.08 MiB  | 11.6 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (4, 4, 4, 4)    | 2 KiB    | 10.1 KiB  | 1.22 MiB  | 12.8 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (1, 1, 1, 1, 1) | 8 B      | 1.90 MiB  | 10.8 MiB  | 299 MiB         |
    +-----------------+----------+-----------+-----------+-----------------+
    """

    param_names = ["shape", "pad_width", "mode"]
    params = [
        # Shape of the input arrays
        [(2 ** 22,), (1024, 1024), (256, 128, 1),
         (4, 4, 4, 4), (1, 1, 1, 1, 1)],
        # Tested pad widths
        [1, 8, (0, 32)],
        # Tested modes: mean, median, minimum & maximum use the same code path
        #               reflect & symmetric share a lot of their code path
        ["constant", "edge", "linear_ramp", "mean", "reflect", "wrap"],
    ]

    def setup(self, shape, pad_width, mode):
        # Make sure to fill the array to make the OS page fault
        # in the setup phase and not the timed phase
        self.array = np.full(shape, fill_value=1, dtype=np.float64)

    def time_pad(self, shape, pad_width, mode):
        np.pad(self.array, pad_width, mode)
