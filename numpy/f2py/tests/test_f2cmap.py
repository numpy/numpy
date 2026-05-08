import numpy as np

from . import util


class TestF2Cmap(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90"),
        util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap")
    ]

    # gh-15095
    def test_gh15095(self):
        inp = np.ones(3)
        out = self.module.func1(inp)
        exp_out = 3
        assert out == exp_out


class TestISOFortranEnvComplexKinds(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "f2cmap", "gh30352.f90"),
    ]

    def test_gh30352(self):
        arr = np.zeros(4, dtype=np.complex128)
        self.module.test_kind_complex.test_function(arr)
        expected = np.full(4, 1.0 - 1.0j, dtype=np.complex128)
        np.testing.assert_array_equal(arr, expected)
