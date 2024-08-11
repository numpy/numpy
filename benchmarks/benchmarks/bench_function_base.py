from .common import Benchmark

import numpy as np

try:
    # SkipNotImplemented is available since 6.0
    from asv_runner.benchmarks.mark import SkipNotImplemented
except ImportError:
    SkipNotImplemented = NotImplementedError


class Linspace(Benchmark):
    """
    Benchmark for numpy's linspace function.
    """
    def setup(self):
        """Sets up the data for the benchmarks."""
        self.d = np.array([1, 2, 3])

    def time_linspace_scalar(self):
        """Benchmark for linspace with scalar inputs."""
        np.linspace(0, 10, 2)

    def time_linspace_array(self):
        """Benchmark for linspace with array inputs."""
        np.linspace(self.d, 10, 10)


class Histogram1D(Benchmark):
    """
    Benchmark for 1d histogram computations using numpy.
    """
    def setup(self):
        """Sets up the data for the benchmarks."""
        self.d = np.linspace(0, 100, 100000)

    def time_full_coverage(self):
        """Benchmark for histogram covering the full range."""
        np.histogram(self.d, 200, (0, 100))

    def time_small_coverage(self):
        """Benchmark for histogram covering a small range."""
        np.histogram(self.d, 200, (50, 51))

    def time_fine_binning(self):
        """Benchmark for histogram with fine binning."""
        np.histogram(self.d, 10000, (0, 100))


class Histogram2D(Benchmark):
    """
    Benchmark for 2D histogram computations using numpy.
    """
    def setup(self):
        self.d = np.linspace(0, 100, 200000).reshape((-1, 2))

    def time_full_coverage(self):
        """Benchmark for 2D histogram covering the full range."""
        np.histogramdd(self.d, (200, 200), ((0, 100), (0, 100)))

    def time_small_coverage(self):
        """Benchmark for 2D histogram covering a small range."""
        np.histogramdd(self.d, (200, 200), ((50, 51), (50, 51)))

    def time_fine_binning(self):
        """Benchmark for 2D histogram with fine binning."""
        np.histogramdd(self.d, (10000, 10000), ((0, 100), (0, 100)))


class Bincount(Benchmark):
    """
    Benchmark for numpy's bincount function.
    """
    def setup(self):
        """Sets up the data for the benchmarks."""
        self.d = np.arange(80000, dtype=np.intp)
        self.e = self.d.astype(np.float64)

    def time_bincount(self):
        """Benchmark for basic bincount."""
        np.bincount(self.d)

    def time_weights(self):
        """Benchmark for bincount with weights."""
        np.bincount(self.d, weights=self.e)


class Mean(Benchmark):
    """
    Benchmark for computing the mean of arrays.
    """
    param_names = ['size']
    params = [[1, 10, 100_000]]

    def setup(self, size):
        """Sets up the data for the benchmarks."""
        self.array = np.arange(2*size).reshape(2, size)

    def time_mean(self, size):
        """Benchmark for mean calculation."""
        np.mean(self.array)

    def time_mean_axis(self, size):
        """Benchmark for mean calculation along axis 1."""
        np.mean(self.array, axis=1)


class Median(Benchmark):
    """
    Benchmark for computing the median of arrays.
    """
    def setup(self):
        """Sets up the data for the benchmarks."""
        self.e = np.arange(10000, dtype=np.float32)
        self.o = np.arange(10001, dtype=np.float32)
        self.tall = np.random.random((10000, 20))
        self.wide = np.random.random((20, 10000))

    def time_even(self):
        """Benchmark for median of an even-length array."""
        np.median(self.e)

    def time_odd(self):
        """Benchmark for median of an odd-length array."""
        np.median(self.o)

    def time_even_inplace(self):
        """Benchmark for in-place median of an even-length array."""
        np.median(self.e, overwrite_input=True)

    def time_odd_inplace(self):
        """Benchmark for in-place median of an odd-length array."""
        np.median(self.o, overwrite_input=True)

    def time_even_small(self):
        """Benchmark for median of a small even-length array."""
        np.median(self.e[:500], overwrite_input=True)

    def time_odd_small(self):
        """Benchmark for median of a small odd-length array."""
        np.median(self.o[:500], overwrite_input=True)

    def time_tall(self):
        """Benchmark for median calculation along axis -1 of a tall array."""
        np.median(self.tall, axis=-1)

    def time_wide(self):
        """Benchmark for median calculation along axis 0 of a wide array."""
        np.median(self.wide, axis=0)


class Percentile(Benchmark):
    """
    Benchmark for computing percentiles.
    """
    def setup(self):
        """Sets up the data for the benchmarks."""
        self.e = np.arange(10000, dtype=np.float32)
        self.o = np.arange(21, dtype=np.float32)

    def time_quartile(self):
        """Benchmark for calculating the first and third quartiles."""
        np.percentile(self.e, [25, 75])

    def time_percentile(self):
        """Benchmark for calculating multiple percentiles."""
        np.percentile(self.e, [25, 35, 55, 65, 75])

    def time_percentile_small(self):
        """Benchmark for calculating percentiles on a small array."""
        np.percentile(self.o, [25, 75])


class Select(Benchmark):
    """
    Benchmark for numpy's select function.
    """
    def setup(self):
        """Sets up the data for the benchmarks"""
        self.d = np.arange(20000)
        self.e = self.d.copy()  # Copy of the original array
        self.cond = [(self.d > 4), (self.d < 2)]
        self.cond_large = [(self.d > 4), (self.d < 2)] * 10

    def time_select(self):
        """Benchmark for np.select with the defined conditions."""
        np.select(self.cond, [self.d, self.e])

    def time_select_larger(self):
        """Benchmark for np.select with a larger set of conditions."""
        np.select(self.cond_large, ([self.d, self.e] * 10))


def memoize(f):
    """Memoization decorater to cache fuction results.
    This helps avoid recomputing results for the same input.
    Parameters: 
    - f, The function to be memoized.
    """
    _memoized = {}

    def wrapped(*args):
        # Check if the result is already computed and cached.
        if args not in _memoized:
            _memoized[args] = f(*args)

        return _memoized[args].copy()

    return f


class SortGenerator:
    """
    A class for generating arrays used in sorting benchmarks.
    """
    # The size of the unsorted area in the "random unsorted area"
    AREA_SIZE = 100
    # The size of the "partially ordered" sub-arrays
    BUBBLE_SIZE = 100

    @staticmethod
    @memoize
    def random(size, dtype, rnd):
        """
        Returns a randomly-shuffled array.
        Parameters:
        - size: The size of the array to generate.
        - dtype: The data type of the array.
        - rnd: Random state for reproducibility.
        """
        arr = np.arange(size, dtype=dtype)
        rnd = np.random.RandomState(1792364059)
        np.random.shuffle(arr)
        rnd.shuffle(arr)
        return arr

    @staticmethod
    @memoize
    def ordered(size, dtype):
        """
        Returns an ordered array.
        Parameters:
        - size: The size of the array to generate.
        - dtypeL The data type of the array.
        """
        return np.arange(size, dtype=dtype)

    @staticmethod
    @memoize
    def reversed(size, dtype):
        """
        Returns an array, of a specified size, that's in descending order.
        Parameters:
        - size: The number of elements in the array.
        - dtype: The desired data type of the array elements.
        Returns:
        - ndarray: An array of integers in descending order.
        """
        dtype = np.dtype(dtype)
        try:
            with np.errstate(over="raise"):
                res = dtype.type(size-1)
        except (OverflowError, FloatingPointError):
            raise SkipNotImplemented("Cannot construct arange for this size.")

        return np.arange(size-1, -1, -1, dtype=dtype)

    @staticmethod
    @memoize
    def uniform(size, dtype):
        """
        Returns an array that has the same value everywhere.
        Parameters:
        - size: The number of elements in the array.
        - dtype: The desired data type of the array elements.
        """
        return np.ones(size, dtype=dtype)

    @staticmethod
    @memoize
    def sorted_block(size, dtype, block_size):
        """
        Returns an array with blocks that are all sorted.
        Parameters:
        - size (int): The total number of elements in the array.
        - dtype (str or dtype): The desired data type of the array elements.
        - block_size (int): The size of each sorted block within the array.
        Returns:
        ndarray: An array where elements are arranged in sorted blocks.
        """
        a = np.arange(size, dtype=dtype)
        b = []
        if size < block_size:
            return a
        block_num = size // block_size
        for i in range(block_num):
            b.extend(a[i::block_num])
        return np.array(b)


class Sort(Benchmark):
    """
    This benchmark tests sorting performance with several
    different types of arrays that are likely to appear in
    real-world applications.
    """
    params = [
        # In NumPy 1.17 and newer, 'merge' can be one of several
        # stable sorts, it isn't necessarily merge sort.
        ['quick', 'merge', 'heap'],
        ['float64', 'int64', 'float32', 'uint32', 'int32', 'int16', 'float16'],
        [
            ('random',),
            ('ordered',),
            ('reversed',),
            ('uniform',),
            ('sorted_block', 10),
            ('sorted_block', 100),
            ('sorted_block', 1000),
        ],
    ]
    param_names = ['kind', 'dtype', 'array_type']

    # The size of the benchmarked arrays.
    ARRAY_SIZE = 10000

    def setup(self, kind, dtype, array_type):
        rnd = np.random.RandomState(507582308)
        array_class = array_type[0]
        self.arr = getattr(SortGenerator, array_class)(self.ARRAY_SIZE, dtype, *array_type[1:], rnd)

    def time_sort(self, kind, dtype, array_type):
        # Using np.sort(...) instead of arr.sort(...) because it makes a copy.
        # This is important because the data is prepared once per benchmark, 
        # but used across multiple runs.
        np.sort(self.arr, kind=kind)

    def time_argsort(self, kind, dtype, array_type):
        np.argsort(self.arr, kind=kind)


class Partition(Benchmark):
    """
    Benchmark tests the performance of partitioning operations
    on arrays of different types.
    """
    params = [
        ['float64', 'int64', 'float32', 'int32', 'int16', 'float16'],
        [
            ('random',),
            ('ordered',),
            ('reversed',),
            ('uniform',),
            ('sorted_block', 10),
            ('sorted_block', 100),
            ('sorted_block', 1000),
        ],
        [10, 100, 1000],
    ]
    param_names = ['dtype', 'array_type', 'k']

    # The size of the benchmarked arrays.
    ARRAY_SIZE = 100000

    def setup(self, dtype, array_type, k):
        """Sets up the data for the benchmarks."""
        rnd = np.random.seed(2136297818)
        array_class = array_type[0]
        self.arr = getattr(SortGenerator, array_class)(
            self.ARRAY_SIZE, dtype, *array_type[1:], rnd)

    def time_partition(self, dtype, array_type, k):
        """Measures the time taken to partition the array."""
        temp = np.partition(self.arr, k)

    def time_argpartition(self, dtype, array_type, k):
        """Measures the time taken to perform argpartition on the array."""
        temp = np.argpartition(self.arr, k)


class SortWorst(Benchmark):
    """
    Benchmark tests the sorting performance on worst-case scenarios
    for the quicksort algorithm
    """
    def setup(self):
        """Sets up the data for the benchmarks."""
        # quicksort median of 3 worst case
        self.worst = np.arange(1000000)
        x = self.worst
        while x.size > 3:
            mid = x.size // 2
            x[mid], x[-2] = x[-2], x[mid]
            x = x[:-2]

    def time_sort_worst(self):
        """Measures the time taken to sort the worst-case array."""
        np.sort(self.worst)

    # Retain old benchmark name for backward compatibility
    time_sort_worst.benchmark_name = "bench_function_base.Sort.time_sort_worst"


class Where(Benchmark):
    """
    Benchmark tests the performace of the np.where function
    under various conditions and array types."""
    def setup(self):
        """Sets up the data for the benchmarks."""
        self.d = np.arange(20000)
        self.d_o = self.d.astype(object)
        self.e = self.d.copy()
        self.e_o = self.d_o.copy()
        self.cond = (self.d > 5000)
        size = 1024 * 1024 // 8
        rnd_array = np.random.rand(size)
        self.rand_cond_01 = rnd_array > 0.01
        self.rand_cond_20 = rnd_array > 0.20
        self.rand_cond_30 = rnd_array > 0.30
        self.rand_cond_40 = rnd_array > 0.40
        self.rand_cond_50 = rnd_array > 0.50
        self.all_zeros = np.zeros(size, dtype=bool)
        self.all_ones = np.ones(size, dtype=bool)
        self.rep_zeros_2 = np.arange(size) % 2 == 0
        self.rep_zeros_4 = np.arange(size) % 4 == 0
        self.rep_zeros_8 = np.arange(size) % 8 == 0
        self.rep_ones_2 = np.arange(size) % 2 > 0
        self.rep_ones_4 = np.arange(size) % 4 > 0
        self.rep_ones_8 = np.arange(size) % 8 > 0

    def time_1(self):
        np.where(self.cond)

    def time_2(self):
        np.where(self.cond, self.d, self.e)

    def time_2_object(self):
        # object and byteswapped arrays have a
        # special slow path in the where internals
        np.where(self.cond, self.d_o, self.e_o)

    def time_2_broadcast(self):
        np.where(self.cond, self.d, 0)

    def time_all_zeros(self):
        np.where(self.all_zeros)

    def time_random_01_percent(self):
        np.where(self.rand_cond_01)

    def time_random_20_percent(self):
        np.where(self.rand_cond_20)

    def time_random_30_percent(self):
        np.where(self.rand_cond_30)

    def time_random_40_percent(self):
        np.where(self.rand_cond_40)

    def time_random_50_percent(self):
        np.where(self.rand_cond_50)

    def time_all_ones(self):
        np.where(self.all_ones)

    def time_interleaved_zeros_x2(self):
        np.where(self.rep_zeros_2)

    def time_interleaved_zeros_x4(self):
        np.where(self.rep_zeros_4)

    def time_interleaved_zeros_x8(self):
        np.where(self.rep_zeros_8)

    def time_interleaved_ones_x2(self):
        np.where(self.rep_ones_2)

    def time_interleaved_ones_x4(self):
        np.where(self.rep_ones_4)

    def time_interleaved_ones_x8(self):
        np.where(self.rep_ones_8)
