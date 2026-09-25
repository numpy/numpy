import numpy as np

from .common import TYPES1, Benchmark


class ScalarMath(Benchmark):
    # Test scalar math, note that each of these is run repeatedly to offset
    # the function call overhead to some degree.
    params = [TYPES1]
    param_names = ["type"]

    def setup(self, typename):
        self.num = np.dtype(typename).type(2)
        self.int32 = np.int32(2)
        self.int32arr = np.array(2, dtype=np.int32)

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

    def time_add_int32_other(self, typename):
        # Some mixed cases are fast, some are slow, this documents these
        # differences.  (When writing, it was fast if the type of the result
        # is one of the inputs.)
        int32 = self.int32
        other = self.num
        int32 + other
        int32 + other
        int32 + other
        int32 + other
        int32 + other

    def time_add_int32arr_and_other(self, typename):
        # `arr + scalar` hits the normal ufunc (array) paths.
        int32 = self.int32arr
        other = self.num
        int32 + other
        int32 + other
        int32 + other
        int32 + other
        int32 + other

    def time_add_other_and_int32arr(self, typename):
        # `scalar + arr` at some point hit scalar paths in some cases, and
        # these paths could be optimized more easily
        int32 = self.int32arr
        other = self.num
        other + int32
        other + int32
        other + int32
        other + int32
        other + int32


class ScalarStr(Benchmark):
    # Test scalar to str conversion
    params = [TYPES1]
    param_names = ["type"]

    def setup(self, typename):
        self.a = np.array([100] * 100, dtype=typename)

    def time_str_repr(self, typename):
        res = [str(x) for x in self.a]
