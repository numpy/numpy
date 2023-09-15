import os
import pytest
import numpy as np

from . import util
from numpy.f2py.crackfortran import crackfortran


class TestData(util.F2PyTest):
    sources = [util.getpath("tests", "src", "crackfortran", "data_stmts.f90")]

    # For gh-23276
    def test_data_stmts(self):
        assert self.module.cmplxdat.i == 2
        assert self.module.cmplxdat.j == 3
        assert self.module.cmplxdat.x == 1.5
        assert self.module.cmplxdat.y == 2.0
        assert self.module.cmplxdat.medium_ref_index == np.array(1.+0.j)
        assert np.all(self.module.cmplxdat.z == np.array([3.5, 7.0]))
        assert np.all(self.module.cmplxdat.my_array == np.array([ 1.+2.j, -3.+4.j]))
        assert np.all(self.module.cmplxdat.my_real_array == np.array([ 1., 2., 3.]))
        assert np.all(self.module.cmplxdat.ref_index_one == np.array([13.0 + 21.0j]))
        assert np.all(self.module.cmplxdat.ref_index_two == np.array([-30.0 + 43.0j]))

    def test_crackedlines(self):
        mod = crackfortran(self.sources)
        assert mod[0]['vars']['x']['='] == '1.5'
        assert mod[0]['vars']['y']['='] == '2.0'
        assert mod[0]['vars']['my_real_array']['='] == '(/1.0d0, 2.0d0, 3.0d0/)'
        assert mod[0]['vars']['ref_index_one']['='] == '(13.0d0, 21.0d0)'
        assert mod[0]['vars']['ref_index_two']['='] == '(-30.0d0, 43.0d0)'
        assert mod[0]['vars']['my_array']['='] == '(/(1.0d0, 2.0d0), (-3.0d0, 4.0d0)/)'
        assert mod[0]['vars']['z']['='] == '(/3.5,  7.0/)'
