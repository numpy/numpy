from __future__ import absolute_import, division, print_function

from .common import Benchmark, get_squares_

import numpy as np


ufuncs = ['abs', 'absolute', 'add', 'arccos', 'arccosh', 'arcsin',
          'arcsinh', 'arctan', 'arctan2', 'arctanh', 'bitwise_and',
          'bitwise_not', 'bitwise_or', 'bitwise_xor', 'cbrt', 'ceil',
          'conj', 'conjugate', 'copysign', 'cos', 'cosh', 'deg2rad',
          'degrees', 'divide', 'equal', 'exp', 'exp2', 'expm1',
          'fabs', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod',
          'frexp', 'greater', 'greater_equal', 'hypot', 'invert',
          'isfinite', 'isinf', 'isnan', 'ldexp', 'left_shift', 'less',
          'less_equal', 'log', 'log10', 'log1p', 'log2', 'logaddexp',
          'logaddexp2', 'logical_and', 'logical_not', 'logical_or',
          'logical_xor', 'maximum', 'minimum', 'mod', 'modf',
          'multiply', 'negative', 'nextafter', 'not_equal', 'power',
          'rad2deg', 'radians', 'reciprocal', 'remainder',
          'right_shift', 'rint', 'sign', 'signbit', 'sin', 'sinh',
          'spacing', 'sqrt', 'square', 'subtract', 'tan', 'tanh',
          'true_divide', 'trunc']

for name in dir(np):
    if isinstance(getattr(np, name, None), np.ufunc) and name not in ufuncs:
        print("Missing ufunc %r" % (name,))


class Broadcast(Benchmark):
    def setup(self):
        self.d = np.ones((50000, 100), dtype=np.float64)
        self.e = np.ones((100,), dtype=np.float64)

    def time_broadcast(self):
        self.d - self.e


class UFunc(Benchmark):
    params = [ufuncs]
    param_names = ['ufunc']
    timeout = 10

    def setup(self, ufuncname):
        np.seterr(all='ignore')
        try:
            self.f = getattr(np, ufuncname)
        except AttributeError:
            raise NotImplementedError()
        self.args = []
        for t, a in get_squares_().items():
            arg = (a,) * self.f.nin
            try:
                self.f(*arg)
            except TypeError:
                continue
            self.args.append(arg)

    def time_ufunc_types(self, ufuncname):
        [self.f(*arg) for arg in self.args]


class Custom(Benchmark):
    def setup(self):
        self.b = np.ones(20000, dtype=np.bool)

    def time_nonzero(self):
        np.nonzero(self.b)

    def time_count_nonzero(self):
        np.count_nonzero(self.b)

    def time_not_bool(self):
        (~self.b)

    def time_and_bool(self):
        (self.b & self.b)

    def time_or_bool(self):
        (self.b | self.b)


class CustomScalar(Benchmark):
    params = [np.float32, np.float64]
    param_names = ['dtype']

    def setup(self, dtype):
        self.d = np.ones(20000, dtype=dtype)

    def time_add_scalar2(self, dtype):
        np.add(self.d, 1)

    def time_divide_scalar2(self, dtype):
        np.divide(self.d, 1)

    def time_divide_scalar2_inplace(self, dtype):
        np.divide(self.d, 1, out=self.d)

    def time_less_than_scalar2(self, dtype):
        (self.d < 1)


class Scalar(Benchmark):
    def setup(self):
        self.x = np.asarray(1.0)
        self.y = np.asarray((1.0 + 1j))
        self.z = complex(1.0, 1.0)

    def time_add_scalar(self):
        (self.x + self.x)

    def time_add_scalar_conv(self):
        (self.x + 1.0)

    def time_add_scalar_conv_complex(self):
        (self.y + self.z)
