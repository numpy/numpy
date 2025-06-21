import numpy as np
from numpy.f2py import testutils


class TestF2Cmap(testutils.F2PyTest):
    sources = [
        testutils.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90"),
        testutils.getpath("tests", "src", "f2cmap", ".f2py_f2cmap")
    ]

    # gh-15095
    def test_gh15095(self):
        inp = np.ones(3)
        out = self.module.func1(inp)
        exp_out = 3
        assert out == exp_out
