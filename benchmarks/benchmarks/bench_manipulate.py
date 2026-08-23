from collections import deque

import numpy as np

from .common import TYPES1, Benchmark


class BroadcastArrays(Benchmark):
    params = [[(16, 32), (128, 256), (512, 1024)],
              TYPES1]
    param_names = ['shape', 'ndtype']
    timeout = 10

    def setup(self, shape, ndtype):
        self.xarg = np.random.ranf(shape[0] * shape[1]).reshape(shape)
        self.xarg = self.xarg.astype(ndtype)
        if ndtype.startswith('complex'):
            self.xarg += np.random.ranf(1) * 1j

    def time_broadcast_arrays(self, shape, ndtype):
        np.broadcast_arrays(self.xarg, np.ones(1))


class BroadcastArraysTo(Benchmark):
    params = [[16, 64, 512],
              TYPES1]
    param_names = ['size', 'ndtype']
    timeout = 10

    def setup(self, size, ndtype):
        self.rng = np.random.default_rng()
        self.xarg = self.rng.random(size)
        self.xarg = self.xarg.astype(ndtype)
        if ndtype.startswith('complex'):
            self.xarg += self.rng.random(1) * 1j

    def time_broadcast_to(self, size, ndtype):
        np.broadcast_to(self.xarg, (size, size))


class ConcatenateStackArrays(Benchmark):
    params = [[(16, 32), (32, 64)],
              [2, 5],
              TYPES1]
    param_names = ['shape', 'narrays', 'ndtype']
    timeout = 10

    def setup(self, shape, narrays, ndtype):
        self.xarg = [np.random.ranf(shape[0] * shape[1]).reshape(shape)
                     for x in range(narrays)]
        self.xarg = [x.astype(ndtype) for x in self.xarg]
        if ndtype.startswith('complex'):
            [x + np.random.ranf(1) * 1j for x in self.xarg]

    def time_concatenate_ax0(self, size, narrays, ndtype):
        np.concatenate(self.xarg, axis=0)

    def time_concatenate_ax1(self, size, narrays, ndtype):
        np.concatenate(self.xarg, axis=1)

    def time_stack_ax0(self, size, narrays, ndtype):
        np.stack(self.xarg, axis=0)

    def time_stack_ax1(self, size, narrays, ndtype):
        np.stack(self.xarg, axis=1)


class ConcatenateNestedArrays(ConcatenateStackArrays):
    # Large number of small arrays to test GIL (non-)release
    params = [[(1, 1)], [1000, 100000], TYPES1]


class DimsManipulations(Benchmark):
    params = [
        [(2, 1, 4), (2, 1), (5, 2, 3, 1)],
    ]
    param_names = ['shape']
    timeout = 10

    def setup(self, shape):
        self.xarg = np.ones(shape=shape)
        self.reshaped = deque(shape)
        self.reshaped.rotate(1)
        self.reshaped = tuple(self.reshaped)

    def time_expand_dims(self, shape):
        np.expand_dims(self.xarg, axis=1)

    def time_expand_dims_neg(self, shape):
        np.expand_dims(self.xarg, axis=-1)

    def time_squeeze_dims(self, shape):
        np.squeeze(self.xarg)

    def time_flip_all(self, shape):
        np.flip(self.xarg, axis=None)

    def time_flip_one(self, shape):
        np.flip(self.xarg, axis=1)

    def time_flip_neg(self, shape):
        np.flip(self.xarg, axis=-1)

    def time_moveaxis(self, shape):
        np.moveaxis(self.xarg, [0, 1], [-1, -2])

    def time_roll(self, shape):
        np.roll(self.xarg, 3)

    def time_reshape(self, shape):
        np.reshape(self.xarg, self.reshaped)
