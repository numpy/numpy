from __future__ import absolute_import, division, print_function

from .common import Benchmark, squares

import numpy as np


class Copy(Benchmark):
    params = ["int8", "int16", "float32", "float64",
              "complex64", "complex128"]
    param_names = ['type']

    def setup(self, typename):
        dtype = np.dtype(typename)
        self.d = np.arange((50 * 500), dtype=dtype).reshape((500, 50))
        self.e = np.arange((50 * 500), dtype=dtype).reshape((50, 500))
        self.e_d = self.e.reshape(self.d.shape)
        self.dflat = np.arange((50 * 500), dtype=dtype)

    def time_memcpy(self, typename):
        self.d[...] = self.e_d

    def time_cont_assign(self, typename):
        self.d[...] = 1

    def time_strided_copy(self, typename):
        self.d[...] = self.e.T

    def time_strided_assign(self, typename):
        self.dflat[::2] = 2


class CopyTo(Benchmark):
    def setup(self):
        self.d = np.ones(50000)
        self.e = self.d.copy()
        self.m = (self.d == 1)
        self.im = (~ self.m)
        self.m8 = self.m.copy()
        self.m8[::8] = (~ self.m[::8])
        self.im8 = (~ self.m8)

    def time_copyto(self):
        np.copyto(self.d, self.e)

    def time_copyto_sparse(self):
        np.copyto(self.d, self.e, where=self.m)

    def time_copyto_dense(self):
        np.copyto(self.d, self.e, where=self.im)

    def time_copyto_8_sparse(self):
        np.copyto(self.d, self.e, where=self.m8)

    def time_copyto_8_dense(self):
        np.copyto(self.d, self.e, where=self.im8)


class Savez(Benchmark):
    def time_vb_savez_squares(self):
        np.savez('tmp.npz', squares)
