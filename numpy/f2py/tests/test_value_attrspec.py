import pytest

from . import util


@pytest.mark.thread_unsafe(reason="f2py is thread-unsafe")
class TestValueAttr(util.F2PyTest):
    sources = [util.getpath("tests", "src", "value_attrspec", "gh21665.f90")]

    # gh-21665
    @pytest.mark.slow
    def test_gh21665(self):
        inp = 2
        out = self.module.fortfuncs.square(inp)
        exp_out = 4
        assert out == exp_out
