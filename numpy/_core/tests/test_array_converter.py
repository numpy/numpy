"""
Tests for numpy/_core/src/multiarray/array_converter.c
"""
import pytest

import numpy as np
from numpy._core._multiarray_umath import _array_converter


def test_pyscalars_self_referencing_array_raises():
    # gh-32700
    obj_array = np.empty(2, dtype=object)
    obj_array[0] = obj_array
    obj_array[1] = [obj_array, obj_array]

    conv = _array_converter([1, 2, 3])
    with pytest.raises(TypeError, match="must be a string"):
        conv.as_arrays(pyscalars=obj_array)


@pytest.mark.parametrize("mode", [123, [], None])
def test_pyscalars_invalid_mode_type(mode):
    conv = _array_converter([1, 2, 3])
    with pytest.raises(TypeError, match="must be a string"):
        conv.as_arrays(pyscalars=mode)


def test_pyscalars_invalid_mode_string():
    conv = _array_converter([1, 2, 3])
    with pytest.raises(ValueError, match="invalid pyscalar mode"):
        conv.as_arrays(pyscalars="invalid")
