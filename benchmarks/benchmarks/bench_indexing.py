from __future__ import absolute_import, division, print_function

from .common import Benchmark, squares_, indexes_, indexes_rand_

import sys
import six
from numpy import memmap, float32, array
import numpy as np


class Indexing(Benchmark):
    params = [["indexes_", "indexes_rand_"],
              ['I', ':,I', 'np.ix_(I, I)'],
              ['', '=1']]
    param_names = ['indexes', 'sel', 'op']

    def setup(self, indexes, sel, op):
        sel = sel.replace('I', indexes)

        ns = {'squares_': squares_,
              'np': np,
              'indexes_': indexes_,
              'indexes_rand_': indexes_rand_}

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
        self.fp = memmap('tmp.dat', dtype=float32, mode='w+', shape=(50, 60))
        self.indexes = array([3, 4, 6, 10, 20])

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
