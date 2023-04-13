from .common import Benchmark, TYPES1

import numpy as np


class MeshGrid(Benchmark):
    """ Benchmark meshgrid generation
    """
    params = [[16, 32],
              [2, 3, 4],
              ['ij', 'xy'], TYPES1]
    param_names = ['size', 'ndims', 'ind', 'ndtype']
    timeout = 10

    def setup(self, size, ndims, ind, ndtype):
        self.grid_dims = [(np.random.ranf(size)).astype(ndtype) for
                          x in range(ndims)]

    def time_meshgrid(self, size, ndims, ind, ndtype):
        np.meshgrid(*self.grid_dims, indexing=ind)


class Create(Benchmark):
    """ Benchmark for creation functions
    """
    # (64, 64), (128, 128), (256, 256)
    # , (512, 512), (1024, 1024)
    params = [[16, 32, 128, 256, 512,
               (16, 16), (32, 32)],
              ['C', 'F'],
              TYPES1]
    param_names = ['shape', 'order', 'npdtypes']
    timeout = 10

    def setup(self, shape, order, npdtypes):
        values = get_squares_()
        self.xarg = values.get(npdtypes)[0]

    def time_full(self, shape, order, npdtypes):
        np.full(shape, self.xarg[1], dtype=npdtypes, order=order)

    def time_full_like(self, shape, order, npdtypes):
        np.full_like(self.xarg, self.xarg[0], order=order)

    def time_ones(self, shape, order, npdtypes):
        np.ones(shape, dtype=npdtypes, order=order)

    def time_ones_like(self, shape, order, npdtypes):
        np.ones_like(self.xarg, order=order)

    def time_zeros(self, shape, order, npdtypes):
        np.zeros(shape, dtype=npdtypes, order=order)

    def time_zeros_like(self, shape, order, npdtypes):
        np.zeros_like(self.xarg, order=order)

    def time_empty(self, shape, order, npdtypes):
        np.empty(shape, dtype=npdtypes, order=order)

    def time_empty_like(self, shape, order, npdtypes):
        np.empty_like(self.xarg, order=order)


class UfuncsFromDLP(Benchmark):
    """ Benchmark for creation functions
    """
    params = [[16, 32, (16, 16),
               (32, 32), (64, 64)],
              TYPES1]
    param_names = ['shape', 'npdtypes']
    timeout = 10

    def setup(self, shape, npdtypes):
        if npdtypes in ['longdouble', 'clongdouble']:
            raise NotImplementedError(
                'Only IEEE dtypes are supported')
        values = get_squares_()
        self.xarg = values.get(npdtypes)[0]

    def time_from_dlpack(self, shape, npdtypes):
        np.from_dlpack(self.xarg)
