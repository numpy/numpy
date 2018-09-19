from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np


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
        self.zero_2d = np.zeros((2 * n, 6 * n))

        self.one = np.ones(3 * n)
        self.two = 2 * np.ones((3, 3 * n))
        self.three = 3 * np.ones(3 * n)
        self.four = 4 * np.ones(3 * n)
        self.five = 5 * np.ones(1 * n)
        self.six = 6 * np.ones(5 * n)
        self.zero = np.zeros((2 * n, 6 * n))

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
    params = [[(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)],
              ['uint8', 'uint16', 'uint32', 'uint64'],
              [(2, 2), (4, 4)]]
    param_names = ['shape', 'dtype', 'n_chunks']

    def setup(self, shape, dtype, n_chunks):

        self.block_list = [
             [np.full(shape=[s//n_chunk for s, n_chunk in zip(shape, n_chunks)],
                     fill_value=1, dtype=dtype) for _ in range(n_chunks[1])]
            for _ in range(n_chunks[0])
        ]

    def time_block2d(self, shape, dtype, n_chunks):
        np.block(self.block_list)


class Block3D(Benchmark):
    params = [1, 10, 100]
    param_names = ['size']

    def setup(self, n):
        # Slow setup method: hence separated from the others above
        self.a000 = np.ones((2 * n, 2 * n, 2 * n), int) * 1

        self.a100 = np.ones((3 * n, 2 * n, 2 * n), int) * 2
        self.a010 = np.ones((2 * n, 3 * n, 2 * n), int) * 3
        self.a001 = np.ones((2 * n, 2 * n, 3 * n), int) * 4

        self.a011 = np.ones((2 * n, 3 * n, 3 * n), int) * 5
        self.a101 = np.ones((3 * n, 2 * n, 3 * n), int) * 6
        self.a110 = np.ones((3 * n, 3 * n, 2 * n), int) * 7

        self.a111 = np.ones((3 * n, 3 * n, 3 * n), int) * 8

    def time_3d(self, n):
        np.block([
            [
                [self.a000, self.a001],
                [self.a010, self.a011],
            ],
            [
                [self.a100, self.a101],
                [self.a110, self.a111],
            ]
        ])

    # Retain old benchmark name for backward compat
    time_3d.benchmark_name = "bench_shape_base.Block.time_3d"
