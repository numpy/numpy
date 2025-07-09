import shutil
from os.path import join as pjoin
from tempfile import mkdtemp

import numpy as np
from numpy import array, float32, memmap

from .common import TYPES1, Benchmark, get_indexes_, get_indexes_rand_, get_square_


class Indexing(Benchmark):
    params = [TYPES1 + ["object", "O,i"],
              ["indexes_", "indexes_rand_"],
              ['I', ':,I', 'np.ix_(I, I)'],
              ['', '=1']]
    param_names = ['dtype', 'indexes', 'sel', 'op']

    def setup(self, dtype, indexes, sel, op):
        sel = sel.replace('I', indexes)

        ns = {'a': get_square_(dtype),
              'np': np,
              'indexes_': get_indexes_(),
              'indexes_rand_': get_indexes_rand_()}

        code = "def run():\n    a[%s]%s"
        code = code % (sel, op)

        exec(code, ns)
        self.func = ns['run']

    def time_op(self, dtype, indexes, sel, op):
        self.func()


class IndexingWith1DArr(Benchmark):
    # Benchmark similar to the take one
    params = [
        [(1000,), (1000, 1), (1000, 2), (2, 1000, 1), (1000, 3)],
        TYPES1 + ["O", "i,O"]]
    param_names = ["shape", "dtype"]

    def setup(self, shape, dtype):
        self.arr = np.ones(shape, dtype)
        self.index = np.arange(1000)
        # index the second dimension:
        if len(shape) == 3:
            self.index = (slice(None), self.index)

    def time_getitem_ordered(self, shape, dtype):
        self.arr[self.index]

    def time_setitem_ordered(self, shape, dtype):
        self.arr[self.index] = 0


class ScalarIndexing(Benchmark):
    params = [[0, 1, 2]]
    param_names = ["ndim"]

    def setup(self, ndim):
        self.array = np.ones((5,) * ndim)

    def time_index(self, ndim):
        # time indexing.
        arr = self.array
        indx = (1,) * ndim
        for i in range(100):
            arr[indx]

    def time_assign(self, ndim):
        # time assignment from a python scalar
        arr = self.array
        indx = (1,) * ndim
        for i in range(100):
            arr[indx] = 5.

    def time_assign_cast(self, ndim):
        # time an assignment which may use a cast operation
        arr = self.array
        indx = (1,) * ndim
        val = np.int16(43)
        for i in range(100):
            arr[indx] = val


class IndexingSeparate(Benchmark):
    def setup(self):
        self.tmp_dir = mkdtemp()
        self.fp = memmap(pjoin(self.tmp_dir, 'tmp.dat'),
                         dtype=float32, mode='w+', shape=(50, 60))
        self.indexes = array([3, 4, 6, 10, 20])

    def teardown(self):
        del self.fp
        shutil.rmtree(self.tmp_dir)

    def time_mmap_slicing(self):
        for i in range(1000):
            self.fp[5:10]

    def time_mmap_fancy_indexing(self):
        for i in range(1000):
            self.fp[self.indexes]


class IndexingStructured0D(Benchmark):
    def setup(self):
        self.dt = np.dtype([('a', 'f4', 256)])

        self.A = np.zeros((), self.dt)
        self.B = self.A.copy()

        self.a = np.zeros(1, self.dt)[0]
        self.b = self.a.copy()

    def time_array_slice(self):
        self.B['a'][:] = self.A['a']

    def time_array_all(self):
        self.B['a'] = self.A['a']

    def time_scalar_slice(self):
        self.b['a'][:] = self.a['a']

    def time_scalar_all(self):
        self.b['a'] = self.a['a']


class FlatIterIndexing(Benchmark):
    def setup(self):
        self.a = np.ones((200, 50000))
        self.m_all = np.repeat(True, 200 * 50000)
        self.m_half = np.copy(self.m_all)
        self.m_half[::2] = False
        self.m_none = np.repeat(False, 200 * 50000)

    def time_flat_bool_index_none(self):
        self.a.flat[self.m_none]

    def time_flat_bool_index_half(self):
        self.a.flat[self.m_half]

    def time_flat_bool_index_all(self):
        self.a.flat[self.m_all]
