import os
import pytest
import numpy as np

from . import util
from numpy.f2py.crackfortran import crackfortran


def test_crackedlines_f90():
    mod = crackfortran(util.getpath("tests", "src", "crackfortran", "data_stmts.f90"))
    assert mod[0]['vars']['x']['='] == '1.5'
    assert mod[0]['vars']['y']['='] == '2.0'
    assert mod[0]['vars']['pi']['='] == '3.1415926535897932384626433832795028841971693993751058209749445923078164062d0'
    assert mod[0]['vars']['my_real_array']['='] == '(/1.0d0, 2.0d0, 3.0d0/)'
    assert mod[0]['vars']['ref_index_one']['='] == '(13.0d0, 21.0d0)'
    assert mod[0]['vars']['ref_index_two']['='] == '(-30.0d0, 43.0d0)'
    assert mod[0]['vars']['my_array']['='] == '(/(1.0d0, 2.0d0), (-3.0d0, 4.0d0)/)'
    assert mod[0]['vars']['z']['='] == '(/3.5,  7.0/)'


def test_crackedlines_f77():
    mod = crackfortran(util.getpath("tests", "src", "crackfortran", "data_common.f"))
    print(mod[0]['vars'])
    assert mod[0]['vars']['mydata']['='] == '0'


@pytest.fixture(scope="module")
def data_test_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestData",
        sources = [util.getpath("tests", "src", "crackfortran", "data_stmts.f90")]
    )
    return spec


@pytest.mark.parametrize("_mod", ["data_test_spec"], indirect=True)
def test_gh23276_data_stmts(_mod):
    assert _mod.cmplxdat.i == 2
    assert _mod.cmplxdat.j == 3
    assert _mod.cmplxdat.x == 1.5
    assert _mod.cmplxdat.y == 2.0
    assert _mod.cmplxdat.pi == 3.1415926535897932384626433832795028841971693993751058209749445923078164062
    assert _mod.cmplxdat.medium_ref_index == np.array(1.+0.j)
    assert np.all(_mod.cmplxdat.z == np.array([3.5, 7.0]))
    assert np.all(_mod.cmplxdat.my_array == np.array([ 1.+2.j, -3.+4.j]))
    assert np.all(_mod.cmplxdat.my_real_array == np.array([ 1., 2., 3.]))
    assert np.all(_mod.cmplxdat.ref_index_one == np.array([13.0 + 21.0j]))
    assert np.all(_mod.cmplxdat.ref_index_two == np.array([-30.0 + 43.0j]))


@pytest.fixture(scope="module")
def data_f77_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestDataF77",
        sources = [util.getpath("tests", "src", "crackfortran", "data_common.f")]
    )
    return spec


# For gh-23276
@pytest.mark.parametrize("_mod", ["data_f77_spec"], indirect=True)
def test_gh23276_f77_data_stmts(_mod):
    assert _mod.mycom.mydata == 0


@pytest.fixture(scope="module")
def data_f77_multiplier_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestDataMultiplierF77",
        sources = [util.getpath("tests", "src", "crackfortran", "data_multiplier.f")]
    )
    return spec


# For gh-23276
@pytest.mark.parametrize("_mod", ["data_f77_multiplier_spec"], indirect=True)
def test_data_stmts(_mod):
    assert _mod.mycom.ivar1 == 3
    assert _mod.mycom.ivar2 == 3
    assert _mod.mycom.ivar3 == 2
    assert _mod.mycom.ivar4 == 2
    assert _mod.mycom.evar5 == 0


@pytest.fixture(scope="module")
def data_f77_comment_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestDataWithCommentsF77",
        sources = [util.getpath("tests", "src", "crackfortran", "data_with_comments.f")]
    )
    return spec


# For gh-23276
@pytest.mark.parametrize("_mod", ["data_f77_comment_spec"], indirect=True)
def test_comment_data_stmts(_mod):
    assert len(_mod.mycom.mytab) == 3
    assert _mod.mycom.mytab[0] == 0
    assert _mod.mycom.mytab[1] == 4
    assert _mod.mycom.mytab[2] == 0
