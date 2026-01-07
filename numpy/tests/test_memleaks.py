import pytest

import numpy as np

psleak = pytest.importorskip("psleak")
LT = psleak.LeakTest

api = {
    # array creation
    "zeros": LT(np.zeros, 10),
    "ones": LT(np.ones, 10),
    "empty": LT(np.empty, 10),
    "full": LT(np.full, 10, 7),
    "arange": LT(np.arange, 10),
    "linspace": LT(np.linspace, 0, 1, 10),
    "eye": LT(np.eye, 3),
    "identity": LT(np.identity, 3),
    "diag": LT(np.diag, [1, 2, 3]),
    "copy": LT(np.copy, np.arange(5)),
    # dtype / casting
    "astype": LT(np.ndarray.astype, np.arange(5), np.float64),
    # basic math / ufuncs
    "add": LT(np.add, 1, 2),
    "subtract": LT(np.subtract, 5, 3),
    "multiply": LT(np.multiply, 3, 4),
    "divide": LT(np.divide, 10, 2),
    "mod": LT(np.mod, 10, 3),
    "power": LT(np.power, 2, 3),
    "sqrt": LT(np.sqrt, 16.0),
    "exp": LT(np.exp, 1.5),
    "log": LT(np.log, 10.0),
    "log2": LT(np.log2, 8.0),
    "log10": LT(np.log10, 100.0),
    "sin": LT(np.sin, 0.5),
    "cos": LT(np.cos, 0.5),
    "tan": LT(np.tan, 0.5),
    "arcsin": LT(np.arcsin, 0.5),
    "arccos": LT(np.arccos, 0.5),
    "arctan": LT(np.arctan, 0.5),
    # comparisons / boolean
    "greater": LT(np.greater, 3, 2),
    "greater_equal": LT(np.greater_equal, 2, 2),
    "less": LT(np.less, 1, 2),
    "less_equal": LT(np.less_equal, 2, 2),
    "equal": LT(np.equal, 3, 3),
    "not_equal": LT(np.not_equal, 3, 2),
    "logical_and": LT(np.logical_and, True, False),
    "logical_or": LT(np.logical_or, True, False),
    "logical_not": LT(np.logical_not, True),
    "logical_xor": LT(np.logical_xor, True, False),
    # reductions
    "sum": LT(np.sum, [1, 2, 3, 4]),
    "mean": LT(np.mean, [1, 2, 3, 4]),
    "max": LT(np.max, [1, 2, 3, 4]),
    "min": LT(np.min, [1, 2, 3, 4]),
    "argmax": LT(np.argmax, [1, 3, 2]),
    "argmin": LT(np.argmin, [1, 3, 2]),
    "prod": LT(np.prod, [1, 2, 3]),
    "std": LT(np.std, [1, 2, 3, 4]),
    "var": LT(np.var, [1, 2, 3, 4]),
    # shape manipulation
    "reshape": LT(np.reshape, np.arange(9), (3, 3)),
    "transpose": LT(np.transpose, np.arange(9).reshape(3, 3)),
    "ravel": LT(np.ravel, np.arange(9).reshape(3, 3)),
    "squeeze": LT(np.squeeze, np.zeros((1, 3, 1))),
    "expand_dims": LT(np.expand_dims, np.arange(3), 0),
    "flatten": LT(np.ndarray.flatten, np.arange(9).reshape(3, 3)),
    # indexing / selection
    "take": LT(np.take, [10, 20, 30], [0, 2]),
    "where": LT(np.where, [True, False, True], [1, 2, 3], [4, 5, 6]),
    "clip": LT(np.clip, [1, 2, 3], 1, 2),
    "nonzero": LT(np.nonzero, [0, 1, 0, 2]),
    "argwhere": LT(np.argwhere, [[0, 1], [2, 3]]),
    # linear algebra
    "dot": LT(np.dot, [1, 2], [3, 4]),
    "matmul": LT(np.matmul, [[1, 2], [3, 4]], [[5, 6], [7, 8]]),
    "norm": LT(np.linalg.norm, [3.0, 4.0]),
    "det": LT(np.linalg.det, [[1, 2], [3, 4]]),
    "inv": LT(np.linalg.inv, [[1, 0], [0, 1]]),
    "eig": LT(np.linalg.eig, [[1, 2], [3, 4]]),
    # stacking / concatenation
    "concatenate": LT(np.concatenate, ([1, 2], [3, 4])),
    "stack": LT(np.stack, ([1, 2], [3, 4])),
    "vstack": LT(np.vstack, ([1, 2], [3, 4])),
    "hstack": LT(np.hstack, ([1, 2], [3, 4])),
    "column_stack": LT(np.column_stack, ([1, 2], [3, 4])),
    "row_stack": LT(np.vstack, ([1, 2], [3, 4])),
    # sorting / unique
    "sort": LT(np.sort, [3, 1, 2]),
    "argsort": LT(np.argsort, [3, 1, 2]),
    "unique": LT(np.unique, [1, 2, 2, 3]),
    # FFTs
    "fft": LT(np.fft.fft, [1, 2, 3, 4]),
    "ifft": LT(np.fft.ifft, [1, 2, 3, 4]),
    "fft2": LT(np.fft.fft2, [[1, 2], [3, 4]]),
    "ifft2": LT(np.fft.ifft2, [[1, 2], [3, 4]]),
    "fftn": LT(np.fft.fftn, [[1, 2], [3, 4]]),
    "ifftn": LT(np.fft.ifftn, [[1, 2], [3, 4]]),
    # random number generation
    "rand": LT(np.random.rand, 5),
    "randn": LT(np.random.randn, 5),
    "random": LT(np.random.random, 5),
    "randint": LT(np.random.randint, 0, 10, 5),
}


class TestNumpyLeaks(psleak.MemoryLeakTestCase):
    verbosity = 1
    checkers = psleak.Checkers.exclude("gcgarbage")

    @classmethod
    def auto_generate(cls):
        return api
