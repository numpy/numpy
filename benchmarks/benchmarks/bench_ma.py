from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np


class MA(Benchmark):
    def setup(self):
        self.l100 = range(100)
        self.t100 = ([True] * 100)

    def time_masked_array(self):
        np.ma.masked_array()

    def time_masked_array_l100(self):
        np.ma.masked_array(self.l100)

    def time_masked_array_l100_t100(self):
        np.ma.masked_array(self.l100, self.t100)
