from __future__ import absolute_import, division, print_function

from .common import Benchmark, squares_, indexes_rand

import numpy as np

class Eindot(Benchmark):
    def setup(self):
        self.a = np.arange(60000.0).reshape(150, 400)
        self.b = np.arange(240000.0).reshape(400, 600)
        self.c = np.arange(600)
        self.d = np.arange(400)

        self.a3 = np.arange(480000.).reshape(60, 80, 100)
        self.b3 = np.arange(192000.).reshape(80, 60, 40)

    def time_einsum_ij_jk_a_b(self):
        np.einsum('ij,jk', self.a, self.b)

    def time_dot_a_b(self):
        np.dot(self.a, self.b)

    def time_einsum_i_ij_j(self):
        np.einsum('i,ij,j', self.d, self.b, self.c)

    def time_dot_d_dot_b_c(self):
        np.dot(self.d, np.dot(self.b, self.c))

    def time_einsum_ijk_jil_kl(self):
        np.einsum('ijk,jil->kl', self.a3, self.b3)

    def time_tensordot_a_b_axes_1_0_0_1(self):
        np.tensordot(self.a3, self.b3, axes=([1, 0], [0, 1]))


class Linalg(Benchmark):
    params = [['svd', 'pinv', 'det', 'norm'],
              list(squares_.keys())]
    param_names = ['op', 'type']

    def setup(self, op, typename):
        np.seterr(all='ignore')

        self.func = getattr(np.linalg, op)

        if op == 'cholesky':
            # we need a positive definite
            self.a = np.dot(squares_[typename],
                            squares_[typename].T)
        else:
            self.a = squares_[typename]

        # check that dtype is supported at all
        try:
            self.func(self.a[:2, :2])
        except TypeError:
            raise NotImplementedError()

    def time_op(self, op, typename):
        self.func(self.a)


class Lstsq(Benchmark):
    def setup(self):
        self.a = squares_['float64']
        self.b = indexes_rand[:100].astype(np.float64)

    def time_numpy_linalg_lstsq_a__b_float64(self):
        np.linalg.lstsq(self.a, self.b)
