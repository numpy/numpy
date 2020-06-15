from .common import Benchmark, get_squares_, get_indexes_rand, TYPES1

import numpy as np


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
    params = [['svd', 'pinv', 'det', 'norm'],
              TYPES1]
    param_names = ['op', 'type']

    def setup(self, op, typename):
        np.seterr(all='ignore')

        self.func = getattr(np.linalg, op)

        if op == 'cholesky':
            # we need a positive definite
            self.a = np.dot(get_squares_()[typename],
                            get_squares_()[typename].T)
        else:
            self.a = get_squares_()[typename]

        # check that dtype is supported at all
        try:
            self.func(self.a[:2, :2])
        except TypeError:
            raise NotImplementedError()

    def time_op(self, op, typename):
        self.func(self.a)


class Lstsq(Benchmark):
    def setup(self):
        self.a = get_squares_()['float64']
        self.b = get_indexes_rand()[:100].astype(np.float64)

    def time_numpy_linalg_lstsq_a__b_float64(self):
        np.linalg.lstsq(self.a, self.b, rcond=-1)

class Einsum(Benchmark):
    param_names = ['dtype']
    params = [[np.float64]]
    def setup(self, dtype):
        self.a = np.arange(2900, dtype=dtype)
        self.b = np.arange(3000, dtype=dtype)
        self.c = np.arange(24000, dtype=dtype).reshape(20, 30, 40)
        self.c1 = np.arange(1200, dtype=dtype).reshape(30, 40)
        self.d = np.arange(10000, dtype=dtype).reshape(10,100,10)

    #outer(a,b): trigger sum_of_products_contig_stride0_outcontig_two
    def time_einsum_outer(self, dtype):
        np.einsum("i,j", self.a, self.b, optimize=True)

    # multiply(a, b):trigger sum_of_products_contig_two
    def time_einsum_multiply(self, dtype):
        np.einsum("..., ...", self.c1, self.c , optimize=True)
    
    # sum and multiply:trigger sum_of_products_contig_stride0_outstride0_two
    def time_einsum_sum_mul(self, dtype):
        np.einsum(",i...->", 300, self.d, optimize=True)

    # sum and multiply:trigger sum_of_products_stride0_contig_outstride0_two
    def time_einsum_sum_mul2(self, dtype):
        np.einsum("i...,->", self.d, 300, optimize=True)