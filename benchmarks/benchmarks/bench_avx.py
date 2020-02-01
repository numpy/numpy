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

avx_bfuncs = ['maximum',
              'minimum']

class AVX_BFunc(Benchmark):

    params = [avx_bfuncs, dtype, stride]
    param_names = ['avx_based_bfunc', 'dtype', 'stride']
    timeout = 10

    def setup(self, ufuncname, dtype, stride):
        np.seterr(all='ignore')
        try:
            self.f = getattr(np, ufuncname)
        except AttributeError:
            raise NotImplementedError()
        N = 10000
        self.arr1 = np.array(np.random.rand(stride*N), dtype=dtype)
        self.arr2 = np.array(np.random.rand(stride*N), dtype=dtype)

    def time_ufunc(self, ufuncname, dtype, stride):
        self.f(self.arr1[::stride], self.arr2[::stride])
