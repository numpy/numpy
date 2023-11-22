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

    # gh-25207
    def test_bindc_kinds(self):
        out = self.module.coddity.c_add_int64(1, 20)
        exp_out = 21
        assert  out == exp_out


def test_process_f2cmap_dict():
    from numpy.f2py.auxfuncs import process_f2cmap_dict
    f2cmap_all = {'integer': {'8': 'long_long'}}
    new_map = {'INTEGER': {'4': 'int'}}
    c2py_map = {'int': 'int', 'long_long': 'long'}

    expected_result = {'integer': {'8': 'long_long', '4': 'int'}}

    # Call the function
    result = process_f2cmap_dict(f2cmap_all, new_map, c2py_map)

    # Assert the result is as expected
    assert result == expected_result
