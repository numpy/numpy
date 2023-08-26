from . import util
import numpy as np

class TestF2Cmap(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90"),
        util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap")
    ]

    # gh-15095
    def test_long_long_map(self):
        inp = np.ones(3)
        out = self.module.func1(inp)
        exp_out = 3
        assert out == exp_out

class TestISOCmap(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "f2cmap", "iso_c_oddity.f90"),
    ]

    # gh-24553
    def test_c_double(self):
        out = self.module.coddity.c_add(1, 2)
        exp_out = 3
        assert  out == exp_out
