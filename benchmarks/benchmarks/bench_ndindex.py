from itertools import product

from numpy.lib import ndindex


class NdindexBenchmark:
    """
    Benchmark comparing numpy.ndindex() and itertools.product()
    for different multi-dimensional shapes.
    """

    # These are the different input shapes to test.
    # Each tuple in this list will be passed as the 'shape' parameter
    # to the benchmark methods.
    params = [
        (10, 10), (20, 20), (50, 50),        # 2D shapes
        (10, 10, 10), (20, 30, 40),          # 3D shapes
        (50, 60, 90)                         # bigger 3D shape
    ]
    param_names = ["shape"]  # Tells ASV what this param means in its output

    def time_ndindex(self, shape):
        """
        Measure time taken by np.ndindex.
        It creates an iterator that goes over each index.
        """
        for _ in ndindex(*shape):
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
        return list(ndindex(*shape))

    def peakmem_itertools_product(self, shape):
        """
        Measure peak memory used when fully consuming
        itertools.product iterator by converting it to a list.
        """
        return list(product(*(range(s) for s in shape)))
