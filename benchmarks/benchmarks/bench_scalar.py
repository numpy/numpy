from random import randint
from .common import Benchmark, TYPES1

import numpy as np


class ScalarMath(Benchmark):
    # Test scalar math, note that each of these is run repeatedly to offset
    # the function call overhead to some degree.
    params = [TYPES1]
    param_names = ["type"]
    def setup(self, typename):
        self.num = np.dtype(typename).type(2)

    def time_addition(self, typename):
        n = self.num
        res = n + n + n + n + n + n + n + n + n + n

    def time_addition_pyint(self, typename):
        n = self.num
        res = n + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1

    def time_multiplication(self, typename):
        n = self.num
        res = n * n * n * n * n * n * n * n * n * n

    def time_power_of_two(self, typename):
        n = self.num
        res = n**2, n**2, n**2, n**2, n**2, n**2, n**2, n**2, n**2, n**2

    def time_abs(self, typename):
        n = self.num
        res = abs(abs(abs(abs(abs(abs(abs(abs(abs(abs(n))))))))))

    def time_compare(self, typename):
        n = self.num
        res = [n == randint(-128, 127) for _ in range(10)]

    def time_compare_types(self, typename):
        n1 = self.num
        for type_lhs in TYPES1:
            n2 = np.dtype(type_lhs).type(randint(-128, 127))
            res = [n1 == n2 for _ in range(10)]
