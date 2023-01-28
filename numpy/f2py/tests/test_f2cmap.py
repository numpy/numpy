import sys
import pytest

from . import util
import numpy as np

pytestmark = pytest.mark.xfail(
    sys.platform == "cygwin", reason="Random fork() failures on Cygwin",
    raises=BlockingIOError
)


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
