import pytest

from numpy.f2py import testutils


class TestValueAttr(testutils.F2PyTest):
    sources = [testutils.getpath("tests", "src", "value_attrspec", "gh21665.f90")]

    # gh-21665
    @pytest.mark.slow
    def test_gh21665(self):
        inp = 2
        out = self.module.fortfuncs.square(inp)
        exp_out = 4
        assert out == exp_out
