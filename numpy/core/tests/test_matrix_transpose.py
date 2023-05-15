import pytest
from hypothesis import given
from hypothesis.extra.numpy import array_shapes

import numpy as np
from numpy.testing import assert_array_equal


def test_matrix_transpose_equals_swapaxes_1d():
    arr = np.arange(48)
    assert_array_equal(arr.T, arr.mT)


def test_matrix_transpose_equals_swapaxes_2d():
    arr = np.arange(48).reshape((6, 8))
    assert_array_equal(arr.T, arr.mT)


@given(shape=array_shapes(min_dims=3))
def test_matrix_transpose_equals_swapaxes(shape):
    num_of_axes = len(shape)
    total_elements = np.prod(shape)
    arr = np.arange(total_elements).reshape(shape)
    tgt = np.swapaxes(arr, num_of_axes - 2, num_of_axes - 1)
    mT = arr.mT
    assert_array_equal(tgt, mT)
