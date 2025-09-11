from itertools import product

import numpy as np

from .common import Benchmark


class NdindexBenchmark(Benchmark):
    """
    Benchmark comparing numpy.ndindex() and itertools.product()
    for different multi-dimensional shapes.
    """

    # Fix: Define each dimension separately, not as tuples
    # ASV will pass each parameter list element to setup()
    params = [
        [(10, 10), (20, 20), (50, 50), (10, 10, 10), (20, 30, 40), (50, 60, 90)]
    ]
    param_names = ["shape"]

    def setup(self, shape):
        """Setup method called before each benchmark run."""
        # Access ndindex through NumPy's main namespace
        self.ndindex = np.ndindex

    def time_ndindex(self, shape):
        """
        Measure time taken by np.ndindex.
        It creates an iterator that goes over each index.
        """
        for _ in self.ndindex(*shape):
            pass  # Just loop through, no work inside

    def time_itertools_product(self, shape):
        """
        Measure time taken by itertools.product.
        Same goal: iterate over all index positions.
        """
        for _ in product(*(range(s) for s in shape)):
            pass

    def peakmem_ndindex(self, shape):
        """
        Measure peak memory used when fully consuming
        np.ndindex iterator by converting it to a list.
        """
        return list(self.ndindex(*shape))

    def peakmem_itertools_product(self, shape):
        """
        Measure peak memory used when fully consuming
        itertools.product iterator by converting it to a list.
        """
        return list(product(*(range(s) for s in shape)))
