import numpy as np

from .common import Benchmark


class DatetimeAsString(Benchmark):
    """ISO string formatting from datetime64 — exercises set_datetimestruct_days
    on every element for every output unit.
    """
    params = [
        [10_000, 1_000_000],
        ['datetime64[D]', 'datetime64[s]', 'datetime64[ms]',
         'datetime64[us]', 'datetime64[ns]'],
    ]
    param_names = ['size', 'dtype']

    def setup(self, size, dtype):
        rng = np.random.default_rng(0xD47E)
        # Span +-50 years around 1970, generate at day resolution then cast.
        days = rng.integers(-18262, 18262, size=size).astype('datetime64[D]')
        self.arr = days.astype(dtype)

    def time_datetime_as_string(self, size, dtype):
        np.datetime_as_string(self.arr)


class DatetimeAstypeCoarser(Benchmark):
    """Cast datetime64 to a coarser calendar unit (Y/M/W).
    Y and M require year/month extraction, which is the days->ymd hot path.
    W requires only day-floor arithmetic and is included as a control.
    """
    params = [
        [1_000_000],
        ['datetime64[D]', 'datetime64[s]', 'datetime64[us]', 'datetime64[ns]'],
        ['datetime64[Y]', 'datetime64[M]', 'datetime64[W]'],
    ]
    param_names = ['size', 'src', 'dst']

    def setup(self, size, src, dst):
        rng = np.random.default_rng(0xD47E)
        days = rng.integers(-18262, 18262, size=size).astype('datetime64[D]')
        self.arr = days.astype(src)

    def time_astype(self, size, src, dst):
        self.arr.astype(dst)


class DatetimeToObject(Benchmark):
    """Convert datetime64 to a Python datetime.date / datetime.datetime array.
    Each element goes through set_datetimestruct_days; Python object creation
    is a fixed overhead on top.
    """
    params = [
        [100_000],
        ['datetime64[D]', 'datetime64[s]', 'datetime64[us]', 'datetime64[ns]'],
    ]
    param_names = ['size', 'dtype']

    def setup(self, size, dtype):
        rng = np.random.default_rng(0xD47E)
        days = rng.integers(-18262, 18262, size=size).astype('datetime64[D]')
        self.arr = days.astype(dtype)

    def time_astype_object(self, size, dtype):
        self.arr.astype(object)


class DatetimeWideRange(Benchmark):
    """Same as DatetimeAsString but spanning a wider date range
    (~+-2700 years from epoch) to stress the leap-cycle code paths.
    """
    params = [[1_000_000], ['datetime64[D]', 'datetime64[us]']]
    param_names = ['size', 'dtype']

    def setup(self, size, dtype):
        rng = np.random.default_rng(0xD47E)
        days = rng.integers(-1_000_000, 1_000_000,
                            size=size).astype('datetime64[D]')
        self.arr = days.astype(dtype)

    def time_datetime_as_string(self, size, dtype):
        np.datetime_as_string(self.arr)
