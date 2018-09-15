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


class Indexing(Benchmark):
    param_names = ['masked', 'ndim', 'size']
    params = [[True, False],
              [1, 2],
              [10, 100, 1000]]
    def setup(self, masked, ndim, size):
        x = np.arange(size**ndim).reshape(ndim * (size,))

        if masked:
            self.m = np.ma.array(x, mask=x%2 == 0)
        else:
            self.m = np.ma.array(x)

        self.idx_scalar = (size//2,) * ndim
        self.idx_0d = (size//2,) * ndim + (Ellipsis,)
        self.idx_1d = (size//2,) * (ndim - 1)

    def time_scalar(self, masked, ndim, size):
        self.m[self.idx_scalar]

    def time_0d(self, masked, ndim, size):
        self.m[self.idx_0d]

    def time_1d(self, masked, ndim, size):
        self.m[self.idx_1d]


class UFunc(Benchmark):
    param_names = ['a_masked', 'b_masked', 'size']
    params = [[True, False],
              [True, False],
              [10, 100, 1000]]

    def setup(self, a_masked, b_masked, size):
        x = np.arange(size).astype(np.uint8)

        self.a_scalar = np.ma.masked if a_masked else 5
        self.b_scalar = np.ma.masked if b_masked else 3

        self.a_1d = np.ma.array(x, mask=x%2 == 0 if a_masked else np.ma.nomask)
        self.b_1d = np.ma.array(x, mask=x%3 == 0 if b_masked else np.ma.nomask)

        self.a_2d = self.a_1d.reshape(1, -1)
        self.b_2d = self.a_1d.reshape(-1, 1)

    def time_scalar(self, a_masked, b_masked, size):
        np.ma.add(self.a_scalar, self.b_scalar)

    def time_scalar_1d(self, a_masked, b_masked, size):
        np.ma.add(self.a_scalar, self.b_1d)

    def time_1d(self, a_masked, b_masked, size):
        np.ma.add(self.a_1d, self.b_1d)

    def time_2d(self, a_masked, b_masked, size):
        # broadcasting happens this time
        np.ma.add(self.a_2d, self.b_2d)


class Concatenate(Benchmark):
    param_names = ['mode', 'n']
    params = [
        ['ndarray', 'unmasked',
         'ndarray+masked', 'unmasked+masked',
         'masked'],
        [2, 100, 2000]
    ]

    def setup(self, mode, n):
        # avoid np.zeros's lazy allocation that cause page faults during benchmark.
        # np.fill will cause pagefaults to happen during setup.
        normal = np.full((n, n), 0, int)
        unmasked = np.ma.zeros((n, n), int)
        masked = np.ma.array(normal, mask=True)

        mode_parts = mode.split('+')
        base = mode_parts[0]
        promote = 'masked' in mode_parts[1:]

        if base == 'ndarray':
            args = 10 * (normal,)
        elif base == 'unmasked':
            args = 10 * (unmasked,)
        else:
            args = 10 * (masked,)

        if promote:
            args = args[:-1] + (masked,)

        self.args = args

    def time_it(self, mode, n):
        np.ma.concatenate(self.args)
