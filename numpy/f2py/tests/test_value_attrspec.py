import os
import sys
import pytest

from . import util

pytestmark = pytest.mark.xfail(
    sys.platform == "cygwin", reason="Random fork() failures on Cygwin",
    raises=BlockingIOError
)

class TestValueAttr(util.F2PyTest):
    sources = [util.getpath("tests", "src", "value_attrspec", "gh21665.f90")]

    # gh-21665
    def test_long_long_map(self):
        inp = 2
        out = self.module.fortfuncs.square(inp)
        exp_out = 4
        assert out == exp_out
