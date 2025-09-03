import pytest

import numpy as np
from numpy.testing import assert_array_equal
from numpy.f2py.crackfortran import crackfortran

from . import util


class TestData(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_stmts.f90")]

    # For gh-23276
    @pytest.mark.slow
    def test_data_stmts(self):
        scalars = {
            "i": 2,
            "j": 3,
            "x": 1.5,
            "y": 2.0,
            "pi": 3.1415926535897932384626433832795028841971693993751058209749445923078164062,
        }

        for name, expected in scalars.items():
            assert getattr(self.module.cmplxdat, name) == expected

        assert_array_equal(
            self.module.cmplxdat.medium_ref_index, np.array(1.0 + 0.0j)
        )
        assert_array_equal(self.module.cmplxdat.z, np.array([3.5, 7.0]))
        assert_array_equal(
            self.module.cmplxdat.my_array, np.array([1.0 + 2.0j, -3.0 + 4.0j])
        )
        assert_array_equal(
            self.module.cmplxdat.my_real_array, np.array([1.0, 2.0, 3.0])
        )
        assert_array_equal(
            self.module.cmplxdat.ref_index_one, np.array([13.0 + 21.0j])
        )
        assert_array_equal(
            self.module.cmplxdat.ref_index_two, np.array([-30.0 + 43.0j])
        )

    def test_crackedlines(self):
        mod = crackfortran(self.sources)
        expected = {
            "x": "1.5",
            "y": "2.0",
            "pi": "3.1415926535897932384626433832795028841971693993751058209749445923078164062d0",
            "my_real_array": "(/1.0d0, 2.0d0, 3.0d0/)",
            "ref_index_one": "(13.0d0, 21.0d0)",
            "ref_index_two": "(-30.0d0, 43.0d0)",
            "my_array": "(/(1.0d0, 2.0d0), (-3.0d0, 4.0d0)/)",
            "z": "(/3.5,  7.0/)",
        }

        for var, value in expected.items():
            assert mod[0]["vars"][var]["="] == value

class TestDataF77(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_common.f")]

    # For gh-23276
    def test_data_stmts(self):
        assert self.module.mycom.mydata == 0

    def test_crackedlines(self):
        mod = crackfortran(str(self.sources[0]))
        assert mod[0]["vars"]["mydata"]["="] == "0"


class TestDataMultiplierF77(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_multiplier.f")]

    # For gh-23276
    def test_data_stmts(self):
        expected = {
            "ivar1": 3,
            "ivar2": 3,
            "ivar3": 2,
            "ivar4": 2,
            "evar5": 0,
        }

        for name, value in expected.items():
            assert getattr(self.module.mycom, name) == value


class TestDataWithCommentsF77(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_with_comments.f")]

    # For gh-23276
    def test_data_stmts(self):
        assert_array_equal(
            self.module.mycom.mytab,
            np.array([0, 4, 0], dtype=self.module.mycom.mytab.dtype),
        )
