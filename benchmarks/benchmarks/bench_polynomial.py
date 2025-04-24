from .common import Benchmark

import numpy as np


class Polynomial(Benchmark):

    def setup(self):
        self.polynomial_degree2 = np.polynomial.Polynomial(np.array([1, 2]))
        self.array3 = np.linspace(0, 1, 3)
        self.array1000 = np.linspace(0, 1, 10_000)
        self.float64 = np.float64(1.0)

    def time_polynomial_evaluation_scalar(self):
        self.polynomial_degree2(self.float64)

    def time_polynomial_evaluation_python_float(self):
        self.polynomial_degree2(1.0)

    def time_polynomial_evaluation_array_3(self):
        self.polynomial_degree2(self.array3)

    def time_polynomial_evaluation_array_1000(self):
        self.polynomial_degree2(self.array1000)

    def time_polynomial_addition(self):
        _ = self.polynomial_degree2 + self.polynomial_degree2
