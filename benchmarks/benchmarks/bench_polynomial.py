import numpy as np

from .common import Benchmark


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


class Polyroots(Benchmark):
    """polyroots() takes an explicit, numerically stable formula for
    degree-2 (quadratic) polynomials instead of the general
    companion-matrix eigenvalue path used for every other degree.
    time_quadratic_* here is the case that formula covers;
    time_cubic_* uses the general eigenvalue path on a similarly
    small problem, for a same-scale comparison."""

    params = [[np.float64, np.float32, np.complex128, np.complex64]]
    param_names = ['dtype']

    def setup(self, dtype):
        # x^2 - 3x + 2 = (x-1)(x-2): real, distinct roots
        self.quadratic_real = np.array([2, -3, 1], dtype=dtype)
        # x^2 + 1: a complex-conjugate pair
        self.quadratic_complex = np.array([1, 0, 1], dtype=dtype)
        # (x-1)(x-2)(x-3): smallest degree that still uses the
        # general eigenvalue path, for a same-scale comparison
        self.cubic = np.array([-6, 11, -6, 1], dtype=dtype)

    def time_quadratic_real_roots(self, dtype):
        np.polynomial.polynomial.polyroots(self.quadratic_real)

    def time_quadratic_complex_roots(self, dtype):
        np.polynomial.polynomial.polyroots(self.quadratic_complex)

    def time_cubic(self, dtype):
        np.polynomial.polynomial.polyroots(self.cubic)
