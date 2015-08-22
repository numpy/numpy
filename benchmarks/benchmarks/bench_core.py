from __future__ import absolute_import, division, print_function

from .common import Benchmark

import numpy


class Core(Benchmark):
    def setup(self):
        self.l100 = range(100)
        self.l50 = range(50)
        self.l = [numpy.arange(1000), numpy.arange(1000)]
        self.l10x10 = numpy.ones((10, 10))

    def time_array_1(self):
        numpy.array(1)

    def time_array_empty(self):
        numpy.array([])

    def time_array_l1(self):
        numpy.array([1])

    def time_array_l100(self):
        numpy.array(self.l100)

    def time_array_l(self):
        numpy.array(self.l)

    def time_vstack_l(self):
        numpy.vstack(self.l)

    def time_hstack_l(self):
        numpy.hstack(self.l)

    def time_dstack_l(self):
        numpy.dstack(self.l)

    def time_arange_100(self):
        numpy.arange(100)

    def time_zeros_100(self):
        numpy.zeros(100)

    def time_ones_100(self):
        numpy.ones(100)

    def time_empty_100(self):
        numpy.empty(100)

    def time_eye_100(self):
        numpy.eye(100)

    def time_identity_100(self):
        numpy.identity(100)

    def time_eye_3000(self):
        numpy.eye(3000)

    def time_identity_3000(self):
        numpy.identity(3000)

    def time_diag_l100(self):
        numpy.diag(self.l100)

    def time_diagflat_l100(self):
        numpy.diagflat(self.l100)

    def time_diagflat_l50_l50(self):
        numpy.diagflat([self.l50, self.l50])

    def time_triu_l10x10(self):
        numpy.triu(self.l10x10)

    def time_tril_l10x10(self):
        numpy.tril(self.l10x10)


class MA(Benchmark):
    def setup(self):
        self.l100 = range(100)
        self.t100 = ([True] * 100)

    def time_masked_array(self):
        numpy.ma.masked_array()

    def time_masked_array_l100(self):
        numpy.ma.masked_array(self.l100)

    def time_masked_array_l100_t100(self):
        numpy.ma.masked_array(self.l100, self.t100)
