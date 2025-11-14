import numpy as np

from .common import Benchmark


class MA(Benchmark):
    def setup(self):
        self.l100 = range(100)
        self.t100 = ([True] * 100)

    def time_masked_array(self):
        np.ma.masked_array()

    def time_masked_array_l100(self):
        np.ma.masked_array(self.l100)

    def time_masked_array_l100_t100(self):
        np.ma.masked_array(self.l100, self.t100)

class MACreation(Benchmark):
    param_names = ['data', 'mask']
    params = [[10, 100, 1000],
              [True, False, None]]

    def time_ma_creations(self, data, mask):
        np.ma.array(data=np.zeros(int(data)), mask=mask)


class Indexing(Benchmark):
    param_names = ['masked', 'ndim', 'size']
    params = [[True, False],
              [1, 2],
              [10, 100, 1000]]

    def setup(self, masked, ndim, size):
        x = np.arange(size**ndim).reshape(ndim * (size,))

        if masked:
            self.m = np.ma.array(x, mask=x % 2 == 0)
        else:
            self.m = np.ma.array(x)

        self.idx_scalar = (size // 2,) * ndim
        self.idx_0d = (size // 2,) * ndim + (Ellipsis,)
        self.idx_1d = (size // 2,) * (ndim - 1)

    def time_scalar(self, masked, ndim, size):
        self.m[self.idx_scalar]

    def time_0d(self, masked, ndim, size):
        self.m[self.idx_0d]

    def time_1d(self, masked, ndim, size):
        self.m[self.idx_1d]


class UFunc(Benchmark):
    param_names = ['a_masked', 'b_masked', 'size']
    params = [[True, False],
              [True, False],
              [10, 100, 1000]]

    def setup(self, a_masked, b_masked, size):
        x = np.arange(size).astype(np.uint8)

        self.a_scalar = np.ma.masked if a_masked else 5
        self.b_scalar = np.ma.masked if b_masked else 3

        self.a_1d = np.ma.array(x, mask=x % 2 == 0 if a_masked else np.ma.nomask)
        self.b_1d = np.ma.array(x, mask=x % 3 == 0 if b_masked else np.ma.nomask)

        self.a_2d = self.a_1d.reshape(1, -1)
        self.b_2d = self.a_1d.reshape(-1, 1)

    def time_scalar(self, a_masked, b_masked, size):
        np.ma.add(self.a_scalar, self.b_scalar)

    def time_scalar_1d(self, a_masked, b_masked, size):
        np.ma.add(self.a_scalar, self.b_1d)

    def time_1d(self, a_masked, b_masked, size):
        np.ma.add(self.a_1d, self.b_1d)

    def time_2d(self, a_masked, b_masked, size):
        # broadcasting happens this time
        np.ma.add(self.a_2d, self.b_2d)


class Concatenate(Benchmark):
    param_names = ['mode', 'n']
    params = [
        ['ndarray', 'unmasked',
         'ndarray+masked', 'unmasked+masked',
         'masked'],
        [2, 100, 2000]
    ]

    def setup(self, mode, n):
        # avoid np.zeros's lazy allocation that cause page faults during benchmark.
        # np.fill will cause pagefaults to happen during setup.
        normal = np.full((n, n), 0, int)
        unmasked = np.ma.zeros((n, n), int)
        masked = np.ma.array(normal, mask=True)

        mode_parts = mode.split('+')
        base = mode_parts[0]
        promote = 'masked' in mode_parts[1:]

        if base == 'ndarray':
            args = 10 * (normal,)
        elif base == 'unmasked':
            args = 10 * (unmasked,)
        else:
            args = 10 * (masked,)

        if promote:
            args = args[:-1] + (masked,)

        self.args = args

    def time_it(self, mode, n):
        np.ma.concatenate(self.args)


class MAFunctions1v(Benchmark):
    param_names = ['mtype', 'func', 'msize']
    params = [['np', 'np.ma'],
              ['sin', 'log', 'sqrt'],
              ['small', 'big']]

    def setup(self, mtype, func, msize):
        xs = 2.0 + np.random.uniform(-1, 1, 6).reshape(2, 3)
        m1 = [[True, False, False], [False, False, True]]
        xl = 2.0 + np.random.uniform(-1, 1, 100 * 100).reshape(100, 100)
        maskx = xl > 2.8
        self.nmxs = np.ma.array(xs, mask=m1)
        self.nmxl = np.ma.array(xl, mask=maskx)

    def time_functions_1v(self, mtype, func, msize):
        # fun = {'np.ma.sin': np.ma.sin, 'np.sin': np.sin}[func]
        fun = eval(f"{mtype}.{func}")
        if msize == 'small':
            fun(self.nmxs)
        elif msize == 'big':
            fun(self.nmxl)


class MAMethod0v(Benchmark):
    param_names = ['method', 'msize']
    params = [['ravel', 'transpose', 'compressed', 'conjugate'],
              ['small', 'big']]

    def setup(self, method, msize):
        xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        m1 = [[True, False, False], [False, False, True]]
        xl = np.random.uniform(-1, 1, 100 * 100).reshape(100, 100)
        maskx = xl > 0.8
        self.nmxs = np.ma.array(xs, mask=m1)
        self.nmxl = np.ma.array(xl, mask=maskx)

    def time_methods_0v(self, method, msize):
        if msize == 'small':
            mdat = self.nmxs
        elif msize == 'big':
            mdat = self.nmxl
        getattr(mdat, method)()


class MAFunctions2v(Benchmark):
    param_names = ['mtype', 'func', 'msize']
    params = [['np', 'np.ma'],
              ['multiply', 'divide', 'power'],
              ['small', 'big']]

    def setup(self, mtype, func, msize):
        # Small arrays
        xs = 2.0 + np.random.uniform(-1, 1, 6).reshape(2, 3)
        ys = 2.0 + np.random.uniform(-1, 1, 6).reshape(2, 3)
        m1 = [[True, False, False], [False, False, True]]
        m2 = [[True, False, True], [False, False, True]]
        self.nmxs = np.ma.array(xs, mask=m1)
        self.nmys = np.ma.array(ys, mask=m2)
        # Big arrays
        xl = 2.0 + np.random.uniform(-1, 1, 100 * 100).reshape(100, 100)
        yl = 2.0 + np.random.uniform(-1, 1, 100 * 100).reshape(100, 100)
        maskx = xl > 2.8
        masky = yl < 1.8
        self.nmxl = np.ma.array(xl, mask=maskx)
        self.nmyl = np.ma.array(yl, mask=masky)

    def time_functions_2v(self, mtype, func, msize):
        fun = eval(f"{mtype}.{func}")
        if msize == 'small':
            fun(self.nmxs, self.nmys)
        elif msize == 'big':
            fun(self.nmxl, self.nmyl)


class MAMethodGetItem(Benchmark):
    param_names = ['margs', 'msize']
    params = [[0, (0, 0), [0, -1]],
              ['small', 'big']]

    def setup(self, margs, msize):
        xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        m1 = [[True, False, False], [False, False, True]]
        xl = np.random.uniform(-1, 1, 100 * 100).reshape(100, 100)
        maskx = xl > 0.8
        self.nmxs = np.ma.array(xs, mask=m1)
        self.nmxl = np.ma.array(xl, mask=maskx)

    def time_methods_getitem(self, margs, msize):
        if msize == 'small':
            mdat = self.nmxs
        elif msize == 'big':
            mdat = self.nmxl
        mdat.__getitem__(margs)


class MAMethodSetItem(Benchmark):
    param_names = ['margs', 'mset', 'msize']
    params = [[0, (0, 0), (-1, 0)],
              [17, np.ma.masked],
              ['small', 'big']]

    def setup(self, margs, mset, msize):
        xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        m1 = [[True, False, False], [False, False, True]]
        xl = np.random.uniform(-1, 1, 100 * 100).reshape(100, 100)
        maskx = xl > 0.8
        self.nmxs = np.ma.array(xs, mask=m1)
        self.nmxl = np.ma.array(xl, mask=maskx)

    def time_methods_setitem(self, margs, mset, msize):
        if msize == 'small':
            mdat = self.nmxs
        elif msize == 'big':
            mdat = self.nmxl
        mdat.__setitem__(margs, mset)


class Where(Benchmark):
    param_names = ['mtype', 'msize']
    params = [['np', 'np.ma'],
              ['small', 'big']]

    def setup(self, mtype, msize):
        # Small arrays
        xs = np.random.uniform(-1, 1, 6).reshape(2, 3)
        ys = np.random.uniform(-1, 1, 6).reshape(2, 3)
        m1 = [[True, False, False], [False, False, True]]
        m2 = [[True, False, True], [False, False, True]]
        self.nmxs = np.ma.array(xs, mask=m1)
        self.nmys = np.ma.array(ys, mask=m2)
        # Big arrays
        xl = np.random.uniform(-1, 1, 100 * 100).reshape(100, 100)
        yl = np.random.uniform(-1, 1, 100 * 100).reshape(100, 100)
        maskx = xl > 0.8
        masky = yl < -0.8
        self.nmxl = np.ma.array(xl, mask=maskx)
        self.nmyl = np.ma.array(yl, mask=masky)

    def time_where(self, mtype, msize):
        fun = eval(f"{mtype}.where")
        if msize == 'small':
            fun(self.nmxs > 2, self.nmxs, self.nmys)
        elif msize == 'big':
            fun(self.nmxl > 2, self.nmxl, self.nmyl)


class Cov(Benchmark):
    param_names = ["size"]
    params = [["small", "large"]]

    def setup(self, size):
        # Set the proportion of masked values.
        prop_mask = 0.2
        # Set up a "small" array with 10 vars and 10 obs.
        rng = np.random.default_rng()
        data = rng.random((10, 10), dtype=np.float32)
        self.small = np.ma.array(data, mask=(data <= prop_mask))
        # Set up a "large" array with 100 vars and 100 obs.
        data = rng.random((100, 100), dtype=np.float32)
        self.large = np.ma.array(data, mask=(data <= prop_mask))

    def time_cov(self, size):
        if size == "small":
            np.ma.cov(self.small)
        if size == "large":
            np.ma.cov(self.large)


class Corrcoef(Benchmark):
    param_names = ["size"]
    params = [["small", "large"]]

    def setup(self, size):
        # Set the proportion of masked values.
        prop_mask = 0.2
        # Set up a "small" array with 10 vars and 10 obs.
        rng = np.random.default_rng()
        data = rng.random((10, 10), dtype=np.float32)
        self.small = np.ma.array(data, mask=(data <= prop_mask))
        # Set up a "large" array with 100 vars and 100 obs.
        data = rng.random((100, 100), dtype=np.float32)
        self.large = np.ma.array(data, mask=(data <= prop_mask))

    def time_corrcoef(self, size):
        if size == "small":
            np.ma.corrcoef(self.small)
        if size == "large":
            np.ma.corrcoef(self.large)
