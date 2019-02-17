from __future__ import absolute_import, division, print_function

from subprocess import call
from sys import executable
from timeit import default_timer

from .common import Benchmark


class Import(Benchmark):
    timer = default_timer

    def execute(self, command):
        call((executable, '-c', command))

    def time_numpy(self):
        self.execute('import numpy')

    def time_numpy_inspect(self):
        # What are the savings from avoiding to import the inspect module?
        self.execute('import numpy, inspect')

    def time_fft(self):
        self.execute('from numpy import fft')

    def time_linalg(self):
        self.execute('from numpy import linalg')

    def time_ma(self):
        self.execute('from numpy import ma')

    def time_matlib(self):
        self.execute('from numpy import matlib')

    def time_random(self):
        self.execute('from numpy import random')
