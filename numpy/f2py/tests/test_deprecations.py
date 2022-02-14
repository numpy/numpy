"""Test deprecation warnings."""

import pytest
from numpy.f2py.utils.deprecation_helpers \
    import deprecated_submodule, _validate_deadline


def test_auxfuncs():
    with pytest.warns(DeprecationWarning):
        import numpy.f2py.auxfuncs


def test_deprecated_submodule():
    with pytest.raises(ImportError):
        deprecated_submodule(new_module_name="numpy.f2py.auxfuncy",
                             old_parent="numpy.f2py",
                             old_child="auxfuncs",
                             deadline="1.25")
        import numpy.f2py.auxfuncs


def test_validate_deadline():
    with pytest.raises(AssertionError):
        _validate_deadline("1.3.2",
                           "numpy.f2py.blah",
                           "numpy.f2py.some.blah")
    with pytest.raises(AssertionError):
        _validate_deadline("1.02",
                           "numpy.f2py.blah",
                           "numpy.f2py.some.blah")
