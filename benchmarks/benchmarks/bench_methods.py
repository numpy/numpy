from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy as np 

class FieldHandling(Benchmark):

    repeat = 1
    number = 1

    def setup(self):
        self.x1 = np.array([21, 32, 14])
        self.x2 = np.array(['my', 'first', 'name'])
        self.x3 = np.array([3.1, 4.5, 6.2])
        self.names = 'id,word,number'
        self.r = np.rec.fromarrays([self.x1, 
                                    self.x2,
                                    self.x3],
                                    names=self.names)

    def time_sort_array_fields(self):
        # time sorting recarray by field
        # why do I get "unknown field name" ValueError
        # from C code version when self.r is defined
        # above vs. here?? Ref counting issue again??
        self.r.sort(order=['word'])
