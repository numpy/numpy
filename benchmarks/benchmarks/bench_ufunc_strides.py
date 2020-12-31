from .common import Benchmark

import numpy as np

unary_ufuncs = ['sin',
              'cos',
              'exp',
              'log',
              'sqrt',
              'absolute',
              'reciprocal',
              'square',
              'rint',
              'floor',
              'ceil' ,
              'trunc',
              'frexp',
              'isnan',
              'isfinite',
              'isinf',
              'signbit']
stride = [1, 2, 4]
stride_out = [1, 2, 4]
dtype  = ['f', 'd']

class Unary(Benchmark):
    params = [unary_ufuncs, stride, stride_out, dtype]
    param_names = ['ufunc', 'stride_in', 'stride_out', 'dtype']
    timeout = 10

    def setup(self, ufuncname, stride, stride_out, dtype):
        np.seterr(all='ignore')
        try:
            self.f = getattr(np, ufuncname)
        except AttributeError:
            raise NotImplementedError()
        N = 10000
        self.arr = np.ones(stride*N, dtype)
        self.arr_out = np.empty(stride_out*N, dtype)

    def time_ufunc(self, ufuncname, stride, stride_out, dtype):
        self.f(self.arr[::stride], self.arr_out[::stride_out])

class AVX_UFunc_log(Benchmark):
    params = [stride, dtype]
    param_names = ['stride', 'dtype']
    timeout = 10

    def setup(self, stride, dtype):
        np.seterr(all='ignore')
        N = 10000
        self.arr = np.array(np.random.random_sample(stride*N), dtype=dtype)

    def time_log(self, stride, dtype):
        np.log(self.arr[::stride])

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

class AVX_ldexp(Benchmark):

    params = [dtype, stride]
    param_names = ['dtype', 'stride']
    timeout = 10

    def setup(self, dtype, stride):
        np.seterr(all='ignore')
        self.f = getattr(np, 'ldexp')
        N = 10000
        self.arr1 = np.array(np.random.rand(stride*N), dtype=dtype)
        self.arr2 = np.array(np.random.rand(stride*N), dtype='i')

    def time_ufunc(self, dtype, stride):
        self.f(self.arr1[::stride], self.arr2[::stride])

cmplx_bfuncs = ['add',
                'subtract',
                'multiply',
                'divide']
cmplxstride = [1, 2, 4]
cmplxdtype  = ['F', 'D']

class AVX_cmplx_arithmetic(Benchmark):
    params = [cmplx_bfuncs, cmplxstride, cmplxdtype]
    param_names = ['bfunc', 'stride', 'dtype']
    timeout = 10

    def setup(self, bfuncname, stride, dtype):
        np.seterr(all='ignore')
        try:
            self.f = getattr(np, bfuncname)
        except AttributeError:
            raise NotImplementedError()
        N = 10000
        self.arr1 = np.ones(stride*N, dtype)
        self.arr2 = np.ones(stride*N, dtype)

    def time_ufunc(self, bfuncname, stride, dtype):
        self.f(self.arr1[::stride], self.arr2[::stride])

cmplx_ufuncs = ['reciprocal',
                'absolute',
                'square',
                'conjugate']

class AVX_cmplx_funcs(Benchmark):
    params = [cmplx_ufuncs, cmplxstride, cmplxdtype]
    param_names = ['bfunc', 'stride', 'dtype']
    timeout = 10

    def setup(self, bfuncname, stride, dtype):
        np.seterr(all='ignore')
        try:
            self.f = getattr(np, bfuncname)
        except AttributeError:
            raise NotImplementedError()
        N = 10000
        self.arr1 = np.ones(stride*N, dtype)

    def time_ufunc(self, bfuncname, stride, dtype):
        self.f(self.arr1[::stride])

class Mandelbrot(Benchmark):
    def f(self,z):
        return np.abs(z) < 4.0

    def g(self,z,c):
        return np.sum(np.multiply(z,z) + c)

    def mandelbrot_numpy(self, c, maxiter):
        output = np.zeros(c.shape, np.int)
        z = np.empty(c.shape, np.complex64)
        for it in range(maxiter):
            notdone = self.f(z)
            output[notdone] = it
            z[notdone] = self.g(z[notdone],c[notdone])
        output[output == maxiter-1] = 0
        return output

    def mandelbrot_set(self,xmin,xmax,ymin,ymax,width,height,maxiter):
        r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
        r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
        c = r1 + r2[:,None]*1j
        n3 = self.mandelbrot_numpy(c,maxiter)
        return (r1,r2,n3.T)

    def time_mandel(self):
        self.mandelbrot_set(-0.74877,-0.74872,0.06505,0.06510,1000,1000,2048)

class LogisticRegression(Benchmark):
    param_names = ['dtype']
    params = [np.float32, np.float64]

    timeout = 1000
    def train(self, max_epoch):
        for epoch in range(max_epoch):
            z = np.matmul(self.X_train, self.W)
            A = 1 / (1 + np.exp(-z)) # sigmoid(z)
            loss = -np.mean(self.Y_train * np.log(A) + (1-self.Y_train) * np.log(1-A))
            dz = A - self.Y_train
            dw = (1/self.size) * np.matmul(self.X_train.T, dz)
            self.W = self.W - self.alpha*dw

    def setup(self, dtype):
        np.random.seed(42)
        self.size = 250
        features = 16
        self.X_train = np.random.rand(self.size,features).astype(dtype)
        self.Y_train = np.random.choice(2,self.size).astype(dtype)
        # Initialize weights
        self.W = np.zeros((features,1), dtype=dtype)
        self.b = np.zeros((1,1), dtype=dtype)
        self.alpha = 0.1

    def time_train(self, dtype):
        self.train(1000)
