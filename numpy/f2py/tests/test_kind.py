import sys
import os
import pytest
import platform

from numpy.f2py.crackfortran import (
    _selected_int_kind_func as selected_int_kind,
    _selected_real_kind_func as selected_real_kind,
)
from . import util, conftest


@pytest.fixture(scope="module")
def kind_test_module(module_builder_factory):
    spec = util.F2PyModuleSpec(
        test_class_name="TestKind",
        sources=[util.getpath("tests", "src", "kind", "foo.f90")],
    )
    built_module = module_builder_factory(spec)
    return built_module


@pytest.mark.skipif(sys.maxsize < 2**31 + 1, reason="Fails for 32 bit machines")
def test_int(kind_test_module):
    """Test `int` kind_func for integers up to 10**40."""
    selectedintkind = kind_test_module.selectedintkind

    for i in range(40):
        assert selectedintkind(i) == selected_int_kind(
            i
        ), f"selectedintkind({i}): expected {selected_int_kind(i)!r} but got {selectedintkind(i)!r}"


def test_real(kind_test_module):
    """Test (processor-dependent) `real` kind_func for real numbers of up to 31 digits precision (extended/quadruple)."""
    selectedrealkind = kind_test_module.selectedrealkind

    for i in range(32):
        assert selectedrealkind(i) == selected_real_kind(
            i
        ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"


@pytest.mark.xfail(
    platform.machine().lower().startswith("ppc"),
    reason="Some PowerPC may not support full IEEE 754 precision",
)
def test_quad_precision(kind_test_module):
    """Test kind_func for quadruple precision [`real(16)`] of 32+ digits."""
    selectedrealkind = kind_test_module.selectedrealkind

    for i in range(32, 40):
        assert selectedrealkind(i) == selected_real_kind(
            i
        ), f"selectedrealkind({i}): expected {selected_real_kind(i)!r} but got {selectedrealkind(i)!r}"
