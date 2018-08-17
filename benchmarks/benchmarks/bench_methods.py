from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np 

class FieldHandling(Benchmark):

    # there's some kind of reference counting
    # issue in C version of _newnames()
    # so restrict to a single iteration
    # with many repeats (and NO warmup)
    # so that self.r may be defined in setup
    # only and its reference isn't messed with
    repeat = 40
    number = 1
    warmup_time = 0.0

    params = [(1, 3, 5, 10, 20, 200)]
    param_names = 'num_fields'

    def setup(self, num_fields):
        num_fields = int(num_fields)
        self.x = np.array([21, 32, 14])
        self.arrays = []

        for field_num in range(num_fields):
            self.arrays.append(self.x)

        self.names = ','.join(np.array(['field' + str(val) for
                                val in range(num_fields)]))

        self.order = self.names.split(',')[::-1]

        self.r = np.rec.fromarrays(self.arrays,
                                   names=self.names)

    def time_sort_array_fields(self, num_fields):
        # time sorting recarray by field
        # why do I get "unknown field name" ValueError
        # from C code version when self.r is defined
        # above vs. here?? Ref counting issue again??
        self.r.sort(order=self.order)
