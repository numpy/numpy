from .common import Benchmark

import numpy as np


class IsClose(Benchmark):
    params = [1, int(1e3), int(1e6)]

    def setup(self, v):
        self.a = np.random.default_rng().random(v)
        self.b = np.random.default_rng().random(v)

    def time_isclose(self, v):
        np.isclose(self.a, self.b, atol=1e-5, rtol=1e-7)
