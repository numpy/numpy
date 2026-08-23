import numpy as np

from .common import Benchmark


class Block(Benchmark):
    params = [1, 10, 100]
    param_names = ['size']

    def setup(self, n):
        self.a_2d = np.ones((2 * n, 2 * n))
        self.b_1d = np.ones(2 * n)
        self.b_2d = 2 * self.a_2d

        self.a = np.ones(3 * n)
        self.b = np.ones(3 * n)

        self.one_2d = np.ones((1 * n, 3 * n))
        self.two_2d = np.ones((1 * n, 3 * n))
        self.three_2d = np.ones((1 * n, 6 * n))
        self.four_1d = np.ones(6 * n)
        self.five_0d = np.ones(1 * n)
        self.six_1d = np.ones(5 * n)
        # avoid np.zeros's lazy allocation that might cause
        # page faults during benchmark
        self.zero_2d = np.full((2 * n, 6 * n), 0)

        self.one = np.ones(3 * n)
        self.two = 2 * np.ones((3, 3 * n))
        self.three = 3 * np.ones(3 * n)
        self.four = 4 * np.ones(3 * n)
        self.five = 5 * np.ones(1 * n)
        self.six = 6 * np.ones(5 * n)
        # avoid np.zeros's lazy allocation that might cause
        # page faults during benchmark
        self.zero = np.full((2 * n, 6 * n), 0)

    def time_block_simple_row_wise(self, n):
        np.block([self.a_2d, self.b_2d])

    def time_block_simple_column_wise(self, n):
        np.block([[self.a_2d], [self.b_2d]])

    def time_block_complicated(self, n):
        np.block([[self.one_2d, self.two_2d],
                  [self.three_2d],
                  [self.four_1d],
                  [self.five_0d, self.six_1d],
                  [self.zero_2d]])

    def time_nested(self, n):
        np.block([
            [
                np.block([
                   [self.one],
                   [self.three],
                   [self.four]
                ]),
                self.two
            ],
            [self.five, self.six],
            [self.zero]
        ])

    def time_no_lists(self, n):
        np.block(1)
        np.block(np.eye(3 * n))


class Block2D(Benchmark):
    params = [[(16, 16), (64, 64), (256, 256), (1024, 1024)],
              ['uint8', 'uint16', 'uint32', 'uint64'],
              [(2, 2), (4, 4)]]
    param_names = ['shape', 'dtype', 'n_chunks']

    def setup(self, shape, dtype, n_chunks):

        self.block_list = [
             [np.full(shape=[s // n_chunk for s, n_chunk in zip(shape, n_chunks)],
                     fill_value=1, dtype=dtype) for _ in range(n_chunks[1])]
            for _ in range(n_chunks[0])
        ]

    def time_block2d(self, shape, dtype, n_chunks):
        np.block(self.block_list)


class Block3D(Benchmark):
    """This benchmark concatenates an array of size ``(5n)^3``"""
    # Having copy as a `mode` of the block3D
    # allows us to directly compare the benchmark of block
    # to that of a direct memory copy into new buffers with
    # the ASV framework.
    # block and copy will be plotted on the same graph
    # as opposed to being displayed as separate benchmarks
    params = [[1, 10, 100],
              ['block', 'copy']]
    param_names = ['n', 'mode']

    def setup(self, n, mode):
        # Slow setup method: hence separated from the others above
        self.a000 = np.ones((2 * n, 2 * n, 2 * n), int) * 1

        self.a100 = np.ones((3 * n, 2 * n, 2 * n), int) * 2
        self.a010 = np.ones((2 * n, 3 * n, 2 * n), int) * 3
        self.a001 = np.ones((2 * n, 2 * n, 3 * n), int) * 4

        self.a011 = np.ones((2 * n, 3 * n, 3 * n), int) * 5
        self.a101 = np.ones((3 * n, 2 * n, 3 * n), int) * 6
        self.a110 = np.ones((3 * n, 3 * n, 2 * n), int) * 7

        self.a111 = np.ones((3 * n, 3 * n, 3 * n), int) * 8

        self.block = [
            [
                [self.a000, self.a001],
                [self.a010, self.a011],
            ],
            [
                [self.a100, self.a101],
                [self.a110, self.a111],
            ]
        ]
        self.arr_list = [a
                         for two_d in self.block
                         for one_d in two_d
                         for a in one_d]

    def time_3d(self, n, mode):
        if mode == 'block':
            np.block(self.block)
        else:  # mode == 'copy'
            [arr.copy() for arr in self.arr_list]

    # Retain old benchmark name for backward compat
    time_3d.benchmark_name = "bench_shape_base.Block.time_3d"


class Kron(Benchmark):
    """Benchmarks for Kronecker product of two arrays"""

    def setup(self):
        self.large_arr = np.random.random((10,) * 4)
        self.large_mat = np.asmatrix(np.random.random((100, 100)))
        self.scalar = 7

    def time_arr_kron(self):
        np.kron(self.large_arr, self.large_arr)

    def time_scalar_kron(self):
        np.kron(self.large_arr, self.scalar)

    def time_mat_kron(self):
        np.kron(self.large_mat, self.large_mat)

class AtLeast1D(Benchmark):
    """Benchmarks for np.atleast_1d"""

    def setup(self):
        self.x = np.array([1, 2, 3])
        self.zero_d = np.float64(1.)

    def time_atleast_1d(self):
        np.atleast_1d(self.x, self.x, self.x)

    def time_atleast_1d_reshape(self):
        np.atleast_1d(self.zero_d, self.zero_d, self.zero_d)

    def time_atleast_1d_single_argument(self):
        np.atleast_1d(self.x)
