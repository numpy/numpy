from . import util
import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture(scope="module")
def isocmap_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestISOC",
        sources=[
            util.getpath("tests", "src", "isocintrin", "isoCtests.f90"),
        ],
    )
    return spec


@pytest.mark.parametrize("_mod", ["isocmap_spec"], indirect=True)
def test_c_double(_mod):
    # gh-24553
    out = _mod.coddity.c_add(1, 2)
    exp_out = 3
    assert out == exp_out


@pytest.mark.parametrize("_mod", ["isocmap_spec"], indirect=True)
def test_bindc_function(_mod):
    # gh-9693
    out = _mod.coddity.wat(1, 20)
    exp_out = 8
    assert out == exp_out


@pytest.mark.parametrize("_mod", ["isocmap_spec"], indirect=True)
def test_bindc_kinds(_mod):
    # gh-25207
    out = _mod.coddity.c_add_int64(1, 20)
    exp_out = 21
    assert out == exp_out


@pytest.mark.parametrize("_mod", ["isocmap_spec"], indirect=True)
def test_bindc_add_arr(_mod):
    # gh-25207
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    out = _mod.coddity.add_arr(a, b)
    exp_out = a * 2
    assert_allclose(out, exp_out)


def test_process_f2cmap_dict():
    from numpy.f2py.auxfuncs import process_f2cmap_dict

    f2cmap_all = {"integer": {"8": "rubbish_type"}}
    new_map = {"INTEGER": {"4": "int"}}
    c2py_map = {"int": "int", "rubbish_type": "long"}

    exp_map, exp_maptyp = ({"integer": {"8": "rubbish_type", "4": "int"}}, ["int"])

    # Call the function
    res_map, res_maptyp = process_f2cmap_dict(f2cmap_all, new_map, c2py_map)

    # Assert the result is as expected
    assert res_map == exp_map
    assert res_maptyp == exp_maptyp
