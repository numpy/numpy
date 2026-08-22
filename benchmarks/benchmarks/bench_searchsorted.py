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


class SearchSortedListNeedle(Benchmark):
    # gh-31495: the needle is a python list, so the runtime is dominated by
    # dtype discovery/conversion of `v`, which must only happen once.
    params = [[1_000, 1_000_000]]
    param_names = ['n_queries']

    def setup(self, n_queries):
        self.arr = np.arange(100)
        self.queries = list(range(n_queries))

    def time_searchsorted_list(self, n_queries):
        self.arr.searchsorted(self.queries)
