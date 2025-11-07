"""Benchmarks for `numpy.lib`."""

import string

import numpy as np

from .common import Benchmark


class Pad(Benchmark):
    """Benchmarks for `numpy.pad`.

    When benchmarking the pad function it is useful to cover scenarios where
    the ratio between the size of the input array and the output array differs
    significantly (original area vs. padded area). This allows to evaluate for
    which scenario a padding algorithm is optimized. Furthermore involving
    large range of array sizes ensures that the effects of CPU-bound caching is
    visible.

    The table below shows the sizes of the arrays involved in this benchmark:

    +-----------------+----------+-----------+-----------+-----------------+
    | shape           | original | padded: 1 | padded: 8 | padded: (0, 32) |
    +=================+==========+===========+===========+=================+
    | (2 ** 22,)      | 32 MiB   | 32.0 MiB  | 32.0 MiB  | 32.0 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (1024, 1024)    | 8 MiB    | 8.03 MiB  | 8.25 MiB  | 8.51 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (256, 256, 1)   | 256 KiB  | 786 KiB   | 5.08 MiB  | 11.6 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (4, 4, 4, 4)    | 2 KiB    | 10.1 KiB  | 1.22 MiB  | 12.8 MiB        |
    +-----------------+----------+-----------+-----------+-----------------+
    | (1, 1, 1, 1, 1) | 8 B      | 1.90 MiB  | 10.8 MiB  | 299 MiB         |
    +-----------------+----------+-----------+-----------+-----------------+
    """

    param_names = ["shape", "pad_width", "mode"]
    params = [
        # Shape of the input arrays
        [(2 ** 22,), (1024, 1024), (256, 128, 1),
         (4, 4, 4, 4), (1, 1, 1, 1, 1)],
        # Tested pad widths
        [1, 8, (0, 32)],
        # Tested modes: mean, median, minimum & maximum use the same code path
        #               reflect & symmetric share a lot of their code path
        ["constant", "edge", "linear_ramp", "mean", "reflect", "wrap"],
    ]

    def setup(self, shape, pad_width, mode):
        # Make sure to fill the array to make the OS page fault
        # in the setup phase and not the timed phase
        self.array = np.full(shape, fill_value=1, dtype=np.float64)

    def time_pad(self, shape, pad_width, mode):
        np.pad(self.array, pad_width, mode)


class Nan(Benchmark):
    """Benchmarks for nan functions"""

    param_names = ["array_size", "percent_nans"]
    params = [
            # sizes of the 1D arrays
            [200, int(2e5)],
            # percent of np.nan in arrays
            [0, 0.1, 2., 50., 90.],
            ]

    def setup(self, array_size, percent_nans):
        rnd = np.random.RandomState(1819780348)
        # produce a randomly shuffled array with the
        # approximate desired percentage np.nan content
        base_array = rnd.uniform(size=array_size)
        base_array[base_array < percent_nans / 100.] = np.nan
        self.arr = base_array

    def time_nanmin(self, array_size, percent_nans):
        np.nanmin(self.arr)

    def time_nanmax(self, array_size, percent_nans):
        np.nanmax(self.arr)

    def time_nanargmin(self, array_size, percent_nans):
        np.nanargmin(self.arr)

    def time_nanargmax(self, array_size, percent_nans):
        np.nanargmax(self.arr)

    def time_nansum(self, array_size, percent_nans):
        np.nansum(self.arr)

    def time_nanprod(self, array_size, percent_nans):
        np.nanprod(self.arr)

    def time_nancumsum(self, array_size, percent_nans):
        np.nancumsum(self.arr)

    def time_nancumprod(self, array_size, percent_nans):
        np.nancumprod(self.arr)

    def time_nanmean(self, array_size, percent_nans):
        np.nanmean(self.arr)

    def time_nanvar(self, array_size, percent_nans):
        np.nanvar(self.arr)

    def time_nanstd(self, array_size, percent_nans):
        np.nanstd(self.arr)

    def time_nanmedian(self, array_size, percent_nans):
        np.nanmedian(self.arr)

    def time_nanquantile(self, array_size, percent_nans):
        np.nanquantile(self.arr, q=0.2)

    def time_nanpercentile(self, array_size, percent_nans):
        np.nanpercentile(self.arr, q=50)


class Unique(Benchmark):
    """Benchmark for np.unique with np.nan values."""

    param_names = ["array_size", "percent_nans", "percent_unique_values", "dtype"]
    params = [
        # sizes of the 1D arrays
        [200, int(2e5)],
        # percent of np.nan in arrays
        [0.0, 10., 90.],
        # percent of unique values in arrays
        [0.2, 20.],
        # dtypes of the arrays
        [np.float64, np.complex128, np.dtypes.StringDType(na_object=np.nan)],
    ]

    def setup(self, array_size, percent_nans, percent_unique_values, dtype):
        rng = np.random.default_rng(123)
        # produce a randomly shuffled array with the
        # approximate desired percentage np.nan content
        unique_values_size = max(int(percent_unique_values / 100. * array_size), 2)
        match dtype:
            case np.float64:
                unique_array = rng.uniform(size=unique_values_size).astype(dtype)
            case np.complex128:
                unique_array = np.array(
                    [
                        complex(*rng.uniform(size=2))
                        for _ in range(unique_values_size)
                    ],
                    dtype=dtype,
                )
            case np.dtypes.StringDType():
                chars = string.ascii_letters + string.digits
                unique_array = np.array(
                    [
                        ''.join(rng.choice(list(chars), size=rng.integers(4, 8)))
                        for _ in range(unique_values_size)
                    ],
                    dtype=dtype,
                )
            case _:
                raise ValueError(f"Unsupported dtype {dtype}")

        base_array = np.resize(unique_array, array_size)
        rng.shuffle(base_array)
        # insert nans in random places
        n_nan = int(percent_nans / 100. * array_size)
        nan_indices = rng.choice(np.arange(array_size), size=n_nan, replace=False)
        base_array[nan_indices] = np.nan
        self.arr = base_array

    def time_unique_values(self, array_size, percent_nans,
                           percent_unique_values, dtype):
        np.unique(self.arr, return_index=False,
                  return_inverse=False, return_counts=False)

    def time_unique_counts(self, array_size, percent_nans,
                           percent_unique_values, dtype):
        np.unique(self.arr, return_index=False,
                  return_inverse=False, return_counts=True,)

    def time_unique_inverse(self, array_size, percent_nans,
                            percent_unique_values, dtype):
        np.unique(self.arr, return_index=False,
                  return_inverse=True, return_counts=False)

    def time_unique_all(self, array_size, percent_nans,
                        percent_unique_values, dtype):
        np.unique(self.arr, return_index=True,
                  return_inverse=True, return_counts=True)


class Isin(Benchmark):
    """Benchmarks for `numpy.isin`."""

    param_names = ["size", "highest_element"]
    params = [
        [10, 100000, 3000000],
        [10, 10000, int(1e8)]
    ]

    def setup(self, size, highest_element):
        self.array = np.random.randint(
                low=0, high=highest_element, size=size)
        self.in_array = np.random.randint(
                low=0, high=highest_element, size=size)

    def time_isin(self, size, highest_element):
        np.isin(self.array, self.in_array)
