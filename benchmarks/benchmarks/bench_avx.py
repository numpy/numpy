from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np

avx_ufuncs = ['sqrt',
              'absolute',
              'reciprocal',
              'square',
              'rint',
              'floor',
              'ceil' ,
              'trunc']
stride = [1, 2, 4]
dtype  = ['f', 'd']

class AVX_UFunc(Benchmark):
    params = [avx_ufuncs, stride, dtype]
    param_names = ['avx_based_ufunc', 'stride', 'dtype']
    timeout = 10

    def setup(self, ufuncname, stride, dtype):
        np.seterr(all='ignore')
        try:
            self.f = getattr(np, ufuncname)
        except AttributeError:
            raise NotImplementedError()
        N = 10000
        self.arr = np.ones(stride*N, dtype)

    def time_ufunc(self, ufuncname, stride, dtype):
        self.f(self.arr[::stride])

