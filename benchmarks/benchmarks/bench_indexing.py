from __future__ import absolute_import, division, print_function

from .common import Benchmark, get_squares_, get_indexes_, get_indexes_rand_

from os.path import join as pjoin
import shutil
import sys
import six
from numpy import memmap, float32, array
import numpy as np
from tempfile import mkdtemp


class Indexing(Benchmark):
    params = [["indexes_", "indexes_rand_"],
              ['I', ':,I', 'np.ix_(I, I)'],
              ['', '=1']]
    param_names = ['indexes', 'sel', 'op']

    def setup(self, indexes, sel, op):
        sel = sel.replace('I', indexes)

        ns = {'squares_': get_squares_(),
              'np': np,
              'indexes_': get_indexes_(),
              'indexes_rand_': get_indexes_rand_()}

        if sys.version_info[0] >= 3:
            code = "def run():\n    for a in squares_.values(): a[%s]%s"
        else:
            code = "def run():\n    for a in squares_.itervalues(): a[%s]%s"
        code = code % (sel, op)

        six.exec_(code, ns)
        self.func = ns['run']

    def time_op(self, indexes, sel, op):
        self.func()


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
