from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np


class PadSuite(Benchmark):
    params = ([(64, 64),
               (256, 256),
               (512, 512),
               (10, 10, 10),
               (10, 64, 64),
               (50, 1024, 1024)],
              ['constant', 'edge', 'linear_ramp',
               'maximum', 'mean', 'median', 'minimum',
               'reflect', 'symmetric', 'wrap'])
    param_names = ['shape', 'mode']

    def setup(self, shape, mode):
        self.array = np.ones(shape=shape)

    def time_pad(self, shape, mode):
        _ = np.pad(self.array, pad_width=10, mode=mode)
