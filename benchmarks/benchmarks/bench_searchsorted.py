import numpy as np

from .common import Benchmark


class SearchSortedInt64(Benchmark):
    # Benchmark for np.searchsorted with int64 arrays
    params = [
        # 1B u64 is 8gb
        [100, 10_000, 1_000_000, 1_000_000_000],       # array sizes
        [1, 2, 100, 100_000],   # number of query elements
        ['ordered', 'random'],        # query order
        [42, 18122022]
    ]
    param_names = ['array_size', 'n_queries', 'query_order', 'seed']

    def setup(self, array_size, n_queries, query_order, seed):
        self.arr = np.arange(array_size, dtype=np.int64)

        rng = np.random.default_rng(seed)

        low = -array_size // 10
        high = array_size + array_size // 10
        self.queries = rng.integers(low, high, size=n_queries, dtype=np.int64)

        # Generate queries
        if query_order == 'ordered':
            self.queries.sort()

    def time_searchsorted(self, array_size, n_queries, query_order, seed):
        np.searchsorted(self.arr, self.queries)
