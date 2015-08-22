from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np


class Random(Benchmark):
    params = ['normal', 'uniform', 'weibull 1', 'binomial 10 0.5',
              'poisson 10']

    def setup(self, name):
        items = name.split()
        name = items.pop(0)
        params = [float(x) for x in items]

        self.func = getattr(np.random, name)
        self.params = tuple(params) + ((100, 100),)

    def time_rng(self, name):
        self.func(*self.params)


class Shuffle(Benchmark):
    def setup(self):
        self.a = np.arange(100000)

    def time_100000(self):
        np.random.shuffle(self.a)
