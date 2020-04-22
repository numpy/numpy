"""
Tests related to the ``symbol`` attribute of the ABCPolyBase class.
"""

import pytest
import numpy.polynomial as poly
from numpy.polynomial._polybase import ABCPolyBase
from numpy.testing import assert_equal, assert_raises

class TestInit:
    """
    Test polynomial creation with symbol kwarg.
    """
    c = [1, 2, 3]
    def test_default_symbol(self):
        p = poly.Polynomial(self.c)
        assert_equal(p.symbol, 'x')

    def test_symbol_None(self):
        with pytest.raises(TypeError):
            p = poly.Polynomial(self.c, symbol=None)

    def test_symbol_empty(self):
        with pytest.raises(ValueError):
            p = poly.Polynomial(self.c, symbol='')
