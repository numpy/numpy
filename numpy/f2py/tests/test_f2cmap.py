import numpy as np
from numpy.f2py import _testutils


class TestF2Cmap(_testutils.F2PyTest):
    sources = [
        _testutils.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90"),
        _testutils.getpath("tests", "src", "f2cmap", ".f2py_f2cmap")
    ]

    # gh-15095
    def test_gh15095(self):
        inp = np.ones(3)
        out = self.module.func1(inp)
        exp_out = 3
        assert out == exp_out
