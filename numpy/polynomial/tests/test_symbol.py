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

    @pytest.mark.parametrize(('bad_input', 'exception'), (
        ('', ValueError),
        ('3', ValueError),
        (None, TypeError),
        (1, TypeError),
    ))
    def test_symbol_bad_input(self, bad_input, exception):
        with pytest.raises(exception):
            p = poly.Polynomial(self.c, symbol=bad_input)

    @pytest.mark.parametrize('symbol', (
        'x',
        'x_1',
        'A',
        'xyz',
        'β',
    ))
    def test_valid_symbols(self, symbol):
        """
        Values for symbol that should pass input validation.
        """
        p = poly.Polynomial(self.c, symbol=symbol)
        assert_equal(p.symbol, symbol)

    def test_change_symbol(self):
        p = poly.Polynomial(self.c, symbol='y')
        # Create new polynomial from p with different symbol
        pt = poly.Polynomial(p, symbol='t')
        assert_equal(pt.symbol, 't')

class TestNumericOperations:
    """
    Test numeric operators to ensure that

        1. Polynomial objects with different symbols cannot be combined
        2. The symbol is preserved by the operation
    """
    p = poly.Polynomial([1, 2, 3], symbol='z')

    def test_neg(self):
        n = -self.p
        assert_equal(n.symbol, 'z')

    def test_add_same_symbol(self):
        other = poly.Polynomial([3, 2, 1], symbol='z')
        out = self.p + other
        assert_equal(out.symbol, 'z')
