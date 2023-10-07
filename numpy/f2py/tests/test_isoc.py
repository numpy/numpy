from . import util
import numpy as np

class TestISOC(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "isocintrin", "isoCtests.f90"),
    ]

    # gh-24553
    def test_c_double(self):
        out = self.module.coddity.c_add(1, 2)
        exp_out = 3
        assert  out == exp_out

    # gh-9693
    def test_bindc_function(self):
        out = self.module.coddity.wat(1, 20)
        exp_out = 8
        assert  out == exp_out
