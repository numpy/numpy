import numpy as np

from .common import Benchmark, get_data

UFUNCS = [obj for obj in np._core.umath.__dict__.values() if
          isinstance(obj, np.ufunc)]
UFUNCS_UNARY = [uf for uf in UFUNCS if "O->O" in uf.types]

class _AbstractBinary(Benchmark):
    params = []
    param_names = ['ufunc', 'stride_in0', 'stride_in1', 'stride_out', 'dtype']
    timeout = 10
    arrlen = 10000
    data_finite = True
    data_denormal = False
    data_zeros = False

    def setup(self, ufunc, stride_in0, stride_in1, stride_out, dtype):
        ufunc_insig = f'{dtype}{dtype}->'
        if ufunc_insig + dtype not in ufunc.types:
            for st_sig in (ufunc_insig, dtype):
                test = [sig for sig in ufunc.types if sig.startswith(st_sig)]
                if test:
                    break
            if not test:
                raise NotImplementedError(
                    f"Ufunc {ufunc} doesn't support "
                    f"binary input of dtype {dtype}"
                ) from None
            tin, tout = test[0].split('->')
        else:
            tin = dtype + dtype
            tout = dtype

        self.ufunc_args = []
        for i, (dt, stride) in enumerate(zip(tin, (stride_in0, stride_in1))):
            self.ufunc_args += [get_data(
                self.arrlen * stride, dt, i,
                zeros=self.data_zeros,
                finite=self.data_finite,
                denormal=self.data_denormal,
            )[::stride]]
        for dt in tout:
            self.ufunc_args += [
                np.empty(stride_out * self.arrlen, dt)[::stride_out]
            ]

        np.seterr(all='ignore')

    def time_binary(self, ufunc, stride_in0, stride_in1, stride_out,
             dtype):
        ufunc(*self.ufunc_args)

    def time_binary_scalar_in0(self, ufunc, stride_in0, stride_in1,
                        stride_out, dtype):
        ufunc(self.ufunc_args[0][0], *self.ufunc_args[1:])

    def time_binary_scalar_in1(self, ufunc, stride_in0, stride_in1,
                        stride_out, dtype):
        ufunc(self.ufunc_args[0], self.ufunc_args[1][0], *self.ufunc_args[2:])

class _AbstractUnary(Benchmark):
    params = []
    param_names = ['ufunc', 'stride_in', 'stride_out', 'dtype']
    timeout = 10
    arrlen = 10000
    data_finite = True
    data_denormal = False
    data_zeros = False

    def setup(self, ufunc, stride_in, stride_out, dtype):
        arr_in = get_data(
            stride_in * self.arrlen, dtype,
            zeros=self.data_zeros,
            finite=self.data_finite,
            denormal=self.data_denormal,
        )
        self.ufunc_args = [arr_in[::stride_in]]

        ufunc_insig = f'{dtype}->'
        if ufunc_insig + dtype not in ufunc.types:
            test = [sig for sig in ufunc.types if sig.startswith(ufunc_insig)]
            if not test:
                raise NotImplementedError(
                    f"Ufunc {ufunc} doesn't support "
                    f"unary input of dtype {dtype}"
                ) from None
            tout = test[0].split('->')[1]
        else:
            tout = dtype

        for dt in tout:
            self.ufunc_args += [
                np.empty(stride_out * self.arrlen, dt)[::stride_out]
            ]

        np.seterr(all='ignore')

    def time_unary(self, ufunc, stride_in, stride_out, dtype):
        ufunc(*self.ufunc_args)

class UnaryFP(_AbstractUnary):
    params = [[uf for uf in UFUNCS_UNARY
                   if uf not in (np.invert, np.bitwise_count)],
              [1, 4],
              [1, 2],
              ['e', 'f', 'd']]

    def setup(self, ufunc, stride_in, stride_out, dtype):
        _AbstractUnary.setup(self, ufunc, stride_in, stride_out, dtype)
        if (ufunc.__name__ == 'arccosh'):
            self.ufunc_args[0] += 1.0

class UnaryFPSpecial(UnaryFP):
    data_finite = False
    data_denormal = True
    data_zeros = True

class BinaryFP(_AbstractBinary):
    params = [
        [np.maximum, np.minimum, np.fmax, np.fmin, np.ldexp],
        [1, 2], [1, 4], [1, 2, 4], ['f', 'd']
    ]

class BinaryFPSpecial(BinaryFP):
    data_finite = False
    data_denormal = True
    data_zeros = True

class BinaryComplex(_AbstractBinary):
    params = [
        [np.add, np.subtract, np.multiply, np.divide],
        [1, 2, 4], [1, 2, 4], [1, 2, 4],
        ['F', 'D']
    ]

class UnaryComplex(_AbstractUnary):
    params = [
        [np.reciprocal, np.absolute, np.square, np.conjugate],
        [1, 2, 4], [1, 2, 4], ['F', 'D']
    ]

class BinaryInt(_AbstractBinary):
    arrlen = 100000
    params = [
        [np.maximum, np.minimum],
        [1, 2], [1, 2], [1, 2],
        ['b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q']
    ]

class BinaryIntContig(_AbstractBinary):
    params = [
        [getattr(np, uf) for uf in (
            'add', 'subtract', 'multiply', 'bitwise_and', 'bitwise_or',
            'bitwise_xor', 'logical_and', 'logical_or', 'logical_xor',
            'right_shift', 'left_shift',
        )],
        [1], [1], [1],
        ['b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q']
    ]

class UnaryIntContig(_AbstractUnary):
    arrlen = 100000
    params = [
        [getattr(np, uf) for uf in (
            'positive', 'square', 'reciprocal', 'conjugate', 'logical_not',
            'invert', 'isnan', 'isinf', 'isfinite',
            'absolute', 'sign', 'bitwise_count'
        )],
        [1], [1],
        ['b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q']
    ]

class Mandelbrot(Benchmark):
    def f(self, z):
        return np.abs(z) < 4.0

    def g(self, z, c):
        return np.sum(np.multiply(z, z) + c)

    def mandelbrot_numpy(self, c, maxiter):
        output = np.zeros(c.shape, np.int32)
        z = np.empty(c.shape, np.complex64)
        for it in range(maxiter):
            notdone = self.f(z)
            output[notdone] = it
            z[notdone] = self.g(z[notdone], c[notdone])
        output[output == maxiter - 1] = 0
        return output

    def mandelbrot_set(self, xmin, xmax, ymin, ymax, width, height, maxiter):
        r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
        r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
        c = r1 + r2[:, None] * 1j
        n3 = self.mandelbrot_numpy(c, maxiter)
        return (r1, r2, n3.T)

    def time_mandel(self):
        self.mandelbrot_set(-0.74877, -0.74872, 0.06505, 0.06510, 1000, 1000, 2048)

class LogisticRegression(Benchmark):
    param_names = ['dtype']
    params = [np.float32, np.float64]

    timeout = 1000

    def train(self, max_epoch):
        for epoch in range(max_epoch):
            z = np.matmul(self.X_train, self.W)
            A = 1 / (1 + np.exp(-z))  # sigmoid(z)
            Y_train = self.Y_train
            loss = -np.mean(Y_train * np.log(A) + (1 - Y_train) * np.log(1 - A))
            dz = A - Y_train
            dw = (1 / self.size) * np.matmul(self.X_train.T, dz)
            self.W = self.W - self.alpha * dw

    def setup(self, dtype):
        np.random.seed(42)
        self.size = 250
        features = 16
        self.X_train = np.random.rand(self.size, features).astype(dtype)
        self.Y_train = np.random.choice(2, self.size).astype(dtype)
        # Initialize weights
        self.W = np.zeros((features, 1), dtype=dtype)
        self.b = np.zeros((1, 1), dtype=dtype)
        self.alpha = 0.1

    def time_train(self, dtype):
        self.train(1000)
