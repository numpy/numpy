from numpy.testing import assert_raises
import numpy as np

from .. import all
from .._creation_functions import asarray
from .._dtypes import float64, int8
from .._manipulation_functions import (
        concat,
        reshape,
        stack
)


def test_concat_errors():
    assert_raises(TypeError, lambda: concat((1, 1), axis=None))
    assert_raises(TypeError, lambda: concat([asarray([1], dtype=int8), asarray([1], dtype=float64)]))


def test_stack_errors():
    assert_raises(TypeError, lambda: stack([asarray([1, 1], dtype=int8), 
                                            asarray([2, 2], dtype=float64)]))


def test_reshape_copy():
    a = asarray([1])
    b = reshape(a, (1, 1), copy=True)
    a[0] = 0
    assert all(b[0, 0] == 1)
    assert all(a[0] == 0)
    assert_raises(NotImplementedError, lambda: reshape(a, (1, 1), copy=False))

