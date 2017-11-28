from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np


class Core(Benchmark):
    def setup(self):
        self.l100 = range(100)
        self.l50 = range(50)
        self.l = [np.arange(1000), np.arange(1000)]
        self.l10x10 = np.ones((10, 10))

    def time_array_1(self):
        np.array(1)

    def time_array_empty(self):
        np.array([])

    def time_array_l1(self):
        np.array([1])

    def time_array_l100(self):
        np.array(self.l100)

    def time_array_l(self):
        np.array(self.l)

    def time_vstack_l(self):
        np.vstack(self.l)

    def time_hstack_l(self):
        np.hstack(self.l)

    def time_dstack_l(self):
        np.dstack(self.l)

    def time_arange_100(self):
        np.arange(100)

    def time_zeros_100(self):
        np.zeros(100)

    def time_ones_100(self):
        np.ones(100)

    def time_empty_100(self):
        np.empty(100)

    def time_eye_100(self):
        np.eye(100)

    def time_identity_100(self):
        np.identity(100)

    def time_eye_3000(self):
        np.eye(3000)

    def time_identity_3000(self):
        np.identity(3000)

    def time_diag_l100(self):
        np.diag(self.l100)

    def time_diagflat_l100(self):
        np.diagflat(self.l100)

    def time_diagflat_l50_l50(self):
        np.diagflat([self.l50, self.l50])

    def time_triu_l10x10(self):
        np.triu(self.l10x10)

    def time_tril_l10x10(self):
        np.tril(self.l10x10)


class Temporaries(Benchmark):
    def setup(self):
        self.amid = np.ones(50000)
        self.bmid = np.ones(50000)
        self.alarge = np.ones(1000000)
        self.blarge = np.ones(1000000)

    def time_mid(self):
        (self.amid * 2) + self.bmid

    def time_mid2(self):
        (self.amid + self.bmid) - 2

    def time_large(self):
        (self.alarge * 2) + self.blarge

    def time_large2(self):
        (self.alarge + self.blarge) - 2


class CorrConv(Benchmark):
    params = [[50, 1000, 1e5],
              [10, 100, 1000, 1e4],
              ['valid', 'same', 'full']]
    param_names = ['size1', 'size2', 'mode']

    def setup(self, size1, size2, mode):
        self.x1 = np.linspace(0, 1, num=size1)
        self.x2 = np.cos(np.linspace(0, 2*np.pi, num=size2))

    def time_correlate(self, size1, size2, mode):
        np.correlate(self.x1, self.x2, mode=mode)

    def time_convolve(self, size1, size2, mode):
        np.convolve(self.x1, self.x2, mode=mode)


class CountNonzero(Benchmark):
    param_names = ['numaxes', 'size', 'dtype']
    params = [
        [1, 2, 3],
        [100, 10000, 1000000],
        [bool, int, str, object]
    ]

    def setup(self, numaxes, size, dtype):
        self.x = np.arange(numaxes * size).reshape(numaxes, size)
        self.x = (self.x % 3).astype(dtype)

    def time_count_nonzero(self, numaxes, size, dtype):
        np.count_nonzero(self.x)

    def time_count_nonzero_axis(self, numaxes, size, dtype):
        np.count_nonzero(self.x, axis=self.x.ndim - 1)

    def time_count_nonzero_multi_axis(self, numaxes, size, dtype):
        if self.x.ndim >= 2:
            np.count_nonzero(self.x, axis=(
                self.x.ndim - 1, self.x.ndim - 2))


class PackBits(Benchmark):
    param_names = ['dtype']
    params = [[bool, np.uintp]]
    def setup(self, dtype):
        self.d = np.ones(10000, dtype=dtype)
        self.d2 = np.ones((200, 1000), dtype=dtype)

    def time_packbits(self, dtype):
        np.packbits(self.d)

    def time_packbits_axis0(self, dtype):
        np.packbits(self.d2, axis=0)

    def time_packbits_axis1(self, dtype):
        np.packbits(self.d2, axis=1)


class UnpackBits(Benchmark):
    def setup(self):
        self.d = np.ones(10000, dtype=np.uint8)
        self.d2 = np.ones((200, 1000), dtype=np.uint8)

    def time_unpackbits(self):
        np.unpackbits(self.d)

    def time_unpackbits_axis0(self):
        np.unpackbits(self.d2, axis=0)

    def time_unpackbits_axis1(self):
        np.unpackbits(self.d2, axis=1)


class Indices(Benchmark):
    def time_indices(self):
        np.indices((1000, 500))
