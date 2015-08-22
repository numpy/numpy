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
            code = "def run():\n    for a in squares_.values(): a[%s]%s" % (sel, op)
        else:
            code = "def run():\n    for a in squares_.itervalues(): a[%s]%s" % (sel, op)

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
