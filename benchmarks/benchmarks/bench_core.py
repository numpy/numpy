import numpy as np

from .common import Benchmark


class Core(Benchmark):
    def setup(self):
        self.l100 = range(100)
        self.l50 = range(50)
        self.float_l1000 = [float(i) for i in range(1000)]
        self.float64_l1000 = [np.float64(i) for i in range(1000)]
        self.int_l1000 = list(range(1000))
        self.l = [np.arange(1000), np.arange(1000)]
        self.l_view = [memoryview(a) for a in self.l]
        self.l10x10 = np.ones((10, 10))
        self.float64_dtype = np.dtype(np.float64)

    def time_array_1(self):
        np.array(1)

    def time_array_empty(self):
        np.array([])

    def time_array_l1(self):
        np.array([1])

    def time_array_l100(self):
        np.array(self.l100)

    def time_array_float_l1000(self):
        np.array(self.float_l1000)

    def time_array_float_l1000_dtype(self):
        np.array(self.float_l1000, dtype=self.float64_dtype)

    def time_array_float64_l1000(self):
        np.array(self.float64_l1000)

    def time_array_int_l1000(self):
        np.array(self.int_l1000)

    def time_array_l(self):
        np.array(self.l)

    def time_array_l_view(self):
        np.array(self.l_view)

    def time_can_cast(self):
        np.can_cast(self.l10x10, self.float64_dtype)

    def time_can_cast_same_kind(self):
        np.can_cast(self.l10x10, self.float64_dtype, casting="same_kind")

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

    def time_empty_like(self):
        np.empty_like(self.l10x10)

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

    def time_triu_indices_500(self):
        np.triu_indices(500)

    def time_tril_indices_500(self):
        np.tril_indices(500)


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
    params = [[50, 1000, int(1e5)],
              [10, 100, 1000, int(1e4)],
              ['valid', 'same', 'full']]
    param_names = ['size1', 'size2', 'mode']

    def setup(self, size1, size2, mode):
        self.x1 = np.linspace(0, 1, num=size1)
        self.x2 = np.cos(np.linspace(0, 2 * np.pi, num=size2))

    def time_correlate(self, size1, size2, mode):
        np.correlate(self.x1, self.x2, mode=mode)

    def time_convolve(self, size1, size2, mode):
        np.convolve(self.x1, self.x2, mode=mode)


class CountNonzero(Benchmark):
    param_names = ['numaxes', 'size', 'dtype']
    params = [
        [1, 2, 3],
        [100, 10000, 1000000],
        [bool, np.int8, np.int16, np.int32, np.int64, np.float32,
         np.float64, str, object]
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


class Nonzero(Benchmark):
    params = [
        [bool, np.uint8, np.uint64, np.int64, np.float32, np.float64],
        [(1_000_000,), (1000, 1000), (100, ), (2, )]
    ]
    param_names = ["dtype", "shape"]

    def setup(self, dtype, size):
        self.x = np.random.randint(0, 3, size=size).astype(dtype)
        self.x_sparse = np.zeros(size).astype(dtype)
        self.x_sparse[1] = 1
        self.x_sparse[-1] = 1
        self.x_dense = np.ones(size).astype(dtype)

    def time_nonzero(self, dtype, size):
        np.nonzero(self.x)

    def time_nonzero_sparse(self, dtype, size):
        np.nonzero(self.x_sparse)

    def time_nonzero_dense(self, dtype, size):
        np.nonzero(self.x_dense)


class PackBits(Benchmark):
    param_names = ['dtype']
    params = [[bool, np.uintp]]

    def setup(self, dtype):
        self.d = np.ones(10000, dtype=dtype)
        self.d2 = np.ones((200, 1000), dtype=dtype)

    def time_packbits(self, dtype):
        np.packbits(self.d)

    def time_packbits_little(self, dtype):
        np.packbits(self.d, bitorder="little")

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

    def time_unpackbits_little(self):
        np.unpackbits(self.d, bitorder="little")

    def time_unpackbits_axis0(self):
        np.unpackbits(self.d2, axis=0)

    def time_unpackbits_axis1(self):
        np.unpackbits(self.d2, axis=1)

    def time_unpackbits_axis1_little(self):
        np.unpackbits(self.d2, bitorder="little", axis=1)


class Indices(Benchmark):
    def time_indices(self):
        np.indices((1000, 500))


class StatsMethods(Benchmark):
    params = [['int64', 'uint64', 'float32', 'float64',
               'complex64', 'bool_'],
              [100, 10000]]
    param_names = ['dtype', 'size']

    def setup(self, dtype, size):
        self.data = np.ones(size, dtype=dtype)
        if dtype.startswith('complex'):
            self.data = np.random.randn(size) + 1j * np.random.randn(size)

    def time_min(self, dtype, size):
        self.data.min()

    def time_max(self, dtype, size):
        self.data.max()

    def time_mean(self, dtype, size):
        self.data.mean()

    def time_std(self, dtype, size):
        self.data.std()

    def time_prod(self, dtype, size):
        self.data.prod()

    def time_var(self, dtype, size):
        self.data.var()

    def time_sum(self, dtype, size):
        self.data.sum()


class NumPyChar(Benchmark):
    def setup(self):
        self.A = np.array([100 * 'x', 100 * 'y'])
        self.B = np.array(1000 * ['aa'])

        self.C = np.array([100 * 'x' + 'z', 100 * 'y' + 'z' + 'y', 100 * 'x'])
        self.D = np.array(1000 * ['ab'] + 1000 * ['ac'])

    def time_isalpha_small_list_big_string(self):
        np.char.isalpha(self.A)

    def time_isalpha_big_list_small_string(self):
        np.char.isalpha(self.B)

    def time_add_small_list_big_string(self):
        np.char.add(self.A, self.A)

    def time_add_big_list_small_string(self):
        np.char.add(self.B, self.B)

    def time_find_small_list_big_string(self):
        np.char.find(self.C, 'z')

    def time_find_big_list_small_string(self):
        np.char.find(self.D, 'b')

    def time_startswith_small_list_big_string(self):
        np.char.startswith(self.A, 'x')

    def time_startswith_big_list_small_string(self):
        np.char.startswith(self.B, 'a')
