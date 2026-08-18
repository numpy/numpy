import numpy as np

from .common import Benchmark


class SearchSorted(Benchmark):
    params = [
        [100, 10_000, 1_000_000, 100_000_000],  # array sizes
        [1, 10, 100_000],                       # number of query elements
        ['ordered', 'random'],                  # query order
        [False, True],                          # use sorter
        [42, 18122022],                         # seed
    ]
    param_names = ['array_size', 'n_queries', 'query_order', 'use_sorter', 'seed']

    def setup(self, array_size, n_queries, query_order, use_sorter, seed):
        self.arr = np.arange(array_size, dtype=np.int32)

        rng = np.random.default_rng(seed)

        low = -array_size // 10
        high = array_size + array_size // 10

        self.queries = rng.integers(low, high, size=n_queries, dtype=np.int32)
        if query_order == 'ordered':
            self.queries.sort()

        if use_sorter:
            rng.shuffle(self.arr)
            self.sorter = self.arr.argsort()
        else:
            self.sorter = None

    def time_searchsorted(self, array_size, n_queries, query_order, use_sorter, seed):
        np.searchsorted(self.arr, self.queries, sorter=self.sorter)


class SearchSortedNd(Benchmark):
    params = [
        [(10, 10_000), (10_000, 10)],  # (batch, length of the searched axis)
        [1, 100],                      # number of query elements per batch
        ['ordered', 'random'],         # query order
        [False, True],                 # use sorter
        [-1, 0],                       # axis
        [42, 18122022],                # seed
    ]
    param_names = ['shape', 'n_queries', 'query_order', 'use_sorter', 'axis',
                   'seed']

    def setup(self, shape, n_queries, query_order, use_sorter, axis, seed):
        batch, length = shape
        rng = np.random.default_rng(seed)

        low = -length // 10
        high = length + length // 10

        arr = rng.integers(0, length, size=(batch, length), dtype=np.int32)
        self.queries = rng.integers(low, high, size=(batch, n_queries),
                                    dtype=np.int32)
        if query_order == 'ordered':
            self.queries.sort(axis=-1)

        if use_sorter:
            self.sorter = arr.argsort(axis=-1)
        else:
            arr.sort(axis=-1)
            self.sorter = None

        # for axis=0 the searched sequence has to live on the first axis
        if axis == -1:
            self.arr = arr
        else:
            self.arr = np.ascontiguousarray(arr.T)
            if self.sorter is not None:
                self.sorter = np.ascontiguousarray(self.sorter.T)

    def time_searchsorted_nd(self, shape, n_queries, query_order, use_sorter,
                             axis, seed):
        np.searchsorted(self.arr, self.queries, sorter=self.sorter, axis=axis)
