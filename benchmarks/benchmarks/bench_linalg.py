import numpy as np

from .common import TYPES1, Benchmark, get_indexes_rand, get_squares_


class Eindot(Benchmark):
    def setup(self):
        self.a = np.arange(60000.0).reshape(150, 400)
        self.ac = self.a.copy()
        self.at = self.a.T
        self.atc = self.a.T.copy()
        self.b = np.arange(240000.0).reshape(400, 600)
        self.c = np.arange(600)
        self.d = np.arange(400)

        self.a3 = np.arange(480000.).reshape(60, 80, 100)
        self.b3 = np.arange(192000.).reshape(80, 60, 40)

    def time_dot_a_b(self):
        np.dot(self.a, self.b)

    def time_dot_d_dot_b_c(self):
        np.dot(self.d, np.dot(self.b, self.c))

    def time_dot_trans_a_at(self):
        np.dot(self.a, self.at)

    def time_dot_trans_a_atc(self):
        np.dot(self.a, self.atc)

    def time_dot_trans_at_a(self):
        np.dot(self.at, self.a)

    def time_dot_trans_atc_a(self):
        np.dot(self.atc, self.a)

    def time_einsum_i_ij_j(self):
        np.einsum('i,ij,j', self.d, self.b, self.c)

    def time_einsum_ij_jk_a_b(self):
        np.einsum('ij,jk', self.a, self.b)

    def time_einsum_ijk_jil_kl(self):
        np.einsum('ijk,jil->kl', self.a3, self.b3)

    def time_inner_trans_a_a(self):
        np.inner(self.a, self.a)

    def time_inner_trans_a_ac(self):
        np.inner(self.a, self.ac)

    def time_matmul_a_b(self):
        np.matmul(self.a, self.b)

    def time_matmul_d_matmul_b_c(self):
        np.matmul(self.d, np.matmul(self.b, self.c))

    def time_matmul_trans_a_at(self):
        np.matmul(self.a, self.at)

    def time_matmul_trans_a_atc(self):
        np.matmul(self.a, self.atc)

    def time_matmul_trans_at_a(self):
        np.matmul(self.at, self.a)

    def time_matmul_trans_atc_a(self):
        np.matmul(self.atc, self.a)

    def time_tensordot_a_b_axes_1_0_0_1(self):
        np.tensordot(self.a3, self.b3, axes=([1, 0], [0, 1]))


class Linalg(Benchmark):
    params = sorted(set(TYPES1) - {'float16'})
    param_names = ['dtype']

    def setup(self, typename):
        np.seterr(all='ignore')
        self.a = get_squares_()[typename]

    def time_svd(self, typename):
        np.linalg.svd(self.a)

    def time_pinv(self, typename):
        np.linalg.pinv(self.a)

    def time_det(self, typename):
        np.linalg.det(self.a)


class LinalgNorm(Benchmark):
    params = TYPES1
    param_names = ['dtype']

    def setup(self, typename):
        self.a = get_squares_()[typename]

    def time_norm(self, typename):
        np.linalg.norm(self.a)


class LinalgSmallArrays(Benchmark):
    """ Test overhead of linalg methods for small arrays """
    def setup(self):
        self.array_3_3 = np.eye(3) + np.arange(9.).reshape((3, 3))
        self.array_3 = np.arange(3.)
        self.array_5 = np.arange(5.)
        self.array_5_5 = np.reshape(np.arange(25.), (5, 5))

    def time_norm_small_array(self):
        np.linalg.norm(self.array_5)

    def time_det_small_array(self):
        np.linalg.det(self.array_5_5)

    def time_det_3x3(self):
        np.linalg.det(self.array_3_3)

    def time_solve_3x3(self):
        np.linalg.solve(self.array_3_3, self.array_3)

    def time_eig_3x3(self):
        np.linalg.eig(self.array_3_3)


class Lstsq(Benchmark):
    def setup(self):
        self.a = get_squares_()['float64']
        self.b = get_indexes_rand()[:100].astype(np.float64)

    def time_numpy_linalg_lstsq_a__b_float64(self):
        np.linalg.lstsq(self.a, self.b, rcond=-1)

class Einsum(Benchmark):
    param_names = ['dtype']
    params = [[np.float32, np.float64]]

    def setup(self, dtype):
        self.one_dim_small = np.arange(600, dtype=dtype)
        self.one_dim = np.arange(3000, dtype=dtype)
        self.one_dim_big = np.arange(480000, dtype=dtype)
        self.two_dim_small = np.arange(1200, dtype=dtype).reshape(30, 40)
        self.two_dim = np.arange(240000, dtype=dtype).reshape(400, 600)
        self.three_dim_small = np.arange(10000, dtype=dtype).reshape(10, 100, 10)
        self.three_dim = np.arange(24000, dtype=dtype).reshape(20, 30, 40)
        # non_contiguous arrays
        self.non_contiguous_dim1_small = np.arange(1, 80, 2, dtype=dtype)
        self.non_contiguous_dim1 = np.arange(1, 4000, 2, dtype=dtype)
        self.non_contiguous_dim2 = np.arange(1, 2400, 2, dtype=dtype).reshape(30, 40)
        self.non_contiguous_dim3 = np.arange(1, 48000, 2, dtype=dtype).reshape(20, 30, 40)

    # outer(a,b): trigger sum_of_products_contig_stride0_outcontig_two
    def time_einsum_outer(self, dtype):
        np.einsum("i,j", self.one_dim, self.one_dim, optimize=True)

    # multiply(a, b):trigger sum_of_products_contig_two
    def time_einsum_multiply(self, dtype):
        np.einsum("..., ...", self.two_dim_small, self.three_dim, optimize=True)

    # sum and multiply:trigger sum_of_products_contig_stride0_outstride0_two
    def time_einsum_sum_mul(self, dtype):
        np.einsum(",i...->", 300, self.three_dim_small, optimize=True)

    # sum and multiply:trigger sum_of_products_stride0_contig_outstride0_two
    def time_einsum_sum_mul2(self, dtype):
        np.einsum("i...,->", self.three_dim_small, 300, optimize=True)

    # scalar mul: trigger sum_of_products_stride0_contig_outcontig_two
    def time_einsum_mul(self, dtype):
        np.einsum("i,->i", self.one_dim_big, 300, optimize=True)

    # trigger contig_contig_outstride0_two
    def time_einsum_contig_contig(self, dtype):
        np.einsum("ji,i->", self.two_dim, self.one_dim_small, optimize=True)

    # trigger sum_of_products_contig_outstride0_one
    def time_einsum_contig_outstride0(self, dtype):
        np.einsum("i->", self.one_dim_big, optimize=True)

    # outer(a,b): non_contiguous arrays
    def time_einsum_noncon_outer(self, dtype):
        np.einsum("i,j", self.non_contiguous_dim1, self.non_contiguous_dim1, optimize=True)

    # multiply(a, b):non_contiguous arrays
    def time_einsum_noncon_multiply(self, dtype):
        np.einsum("..., ...", self.non_contiguous_dim2, self.non_contiguous_dim3, optimize=True)

    # sum and multiply:non_contiguous arrays
    def time_einsum_noncon_sum_mul(self, dtype):
        np.einsum(",i...->", 300, self.non_contiguous_dim3, optimize=True)

    # sum and multiply:non_contiguous arrays
    def time_einsum_noncon_sum_mul2(self, dtype):
        np.einsum("i...,->", self.non_contiguous_dim3, 300, optimize=True)

    # scalar mul: non_contiguous arrays
    def time_einsum_noncon_mul(self, dtype):
        np.einsum("i,->i", self.non_contiguous_dim1, 300, optimize=True)

    # contig_contig_outstride0_two: non_contiguous arrays
    def time_einsum_noncon_contig_contig(self, dtype):
        np.einsum("ji,i->", self.non_contiguous_dim2, self.non_contiguous_dim1_small, optimize=True)

    # sum_of_products_contig_outstride0_one: non_contiguous arrays
    def time_einsum_noncon_contig_outstride0(self, dtype):
        np.einsum("i->", self.non_contiguous_dim1, optimize=True)


class LinAlgTransposeVdot(Benchmark):
    # Smaller for speed
    # , (128, 128), (256, 256), (512, 512),
    # (1024, 1024)
    params = [[(16, 16), (32, 32),
               (64, 64)], TYPES1]
    param_names = ['shape', 'npdtypes']

    def setup(self, shape, npdtypes):
        self.xarg = np.random.uniform(-1, 1, np.dot(*shape)).reshape(shape)
        self.xarg = self.xarg.astype(npdtypes)
        self.x2arg = np.random.uniform(-1, 1, np.dot(*shape)).reshape(shape)
        self.x2arg = self.x2arg.astype(npdtypes)
        if npdtypes.startswith('complex'):
            self.xarg += self.xarg.T * 1j
            self.x2arg += self.x2arg.T * 1j

    def time_transpose(self, shape, npdtypes):
        np.transpose(self.xarg)

    def time_vdot(self, shape, npdtypes):
        np.vdot(self.xarg, self.x2arg)


class MatmulStrided(Benchmark):
    # some interesting points selected from
    # https://github.com/numpy/numpy/pull/23752#issuecomment-2629521597
    # (m, p, n, batch_size)
    args = [
        (2, 2, 2, 1), (2, 2, 2, 10), (5, 5, 5, 1), (5, 5, 5, 10),
        (10, 10, 10, 1), (10, 10, 10, 10), (20, 20, 20, 1), (20, 20, 20, 10),
        (50, 50, 50, 1), (50, 50, 50, 10),
        (150, 150, 100, 1), (150, 150, 100, 10),
        (400, 400, 100, 1), (400, 400, 100, 10)
    ]

    param_names = ['configuration']

    def __init__(self):
        self.args_map = {
            'matmul_m%03d_p%03d_n%03d_bs%02d' % arg: arg for arg in self.args
        }

        self.params = [list(self.args_map.keys())]

    def setup(self, configuration):
        m, p, n, batch_size = self.args_map[configuration]

        self.a1raw = np.random.rand(batch_size * m * 2 * n).reshape(
            (batch_size, m, 2 * n)
        )

        self.a1 = self.a1raw[:, :, ::2]

        self.a2 = np.random.rand(batch_size * n * p).reshape(
            (batch_size, n, p)
        )

    def time_matmul(self, configuration):
        return np.matmul(self.a1, self.a2)
