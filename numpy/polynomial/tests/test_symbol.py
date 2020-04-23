"""
Tests related to the ``symbol`` attribute of the ABCPolyBase class.
"""

import pytest
import numpy.polynomial as poly
from numpy.core import array
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


class TestUnaryOperators:
    p = poly.Polynomial([1, 2, 3], symbol='z')

    def test_neg(self):
        n = -self.p
        assert_equal(n.symbol, 'z')

    def test_scalarmul(self):
        out = self.p * 10
        assert_equal(out.symbol, 'z')

    def test_rscalarmul(self):
        out = 10 * self.p
        assert_equal(out.symbol, 'z')

    def test_pow(self):
        out = self.p ** 3
        assert_equal(out.symbol, 'z')


@pytest.mark.parametrize('rhs',(
    poly.Polynomial([4, 5, 6], symbol='z'),
    array([4, 5, 6]),
))
class TestBinaryOperatorsSameSymbol:
    """
    Ensure symbol is preserved for numeric operations on polynomials with
    the same symbol
    """
    p = poly.Polynomial([1, 2, 3], symbol='z')

    def test_add(self, rhs):
        out = self.p + rhs
        assert_equal(out.symbol, 'z')

    def test_sub(self, rhs):
        out = self.p - rhs
        assert_equal(out.symbol, 'z')

    def test_polymul(self, rhs):
        out = self.p * rhs
        assert_equal(out.symbol, 'z')

    def test_divmod(self, rhs):
        for out in divmod(self.p, rhs):
            assert_equal(out.symbol, 'z')

    def test_radd(self, rhs):
        out = rhs + self.p
        assert_equal(out.symbol, 'z')

    def test_rsub(self, rhs):
        out = rhs - self.p
        assert_equal(out.symbol, 'z')

    def test_rmul(self, rhs):
        out = rhs * self.p
        assert_equal(out.symbol, 'z')

    def test_rdivmod(self, rhs):
        for out in divmod(rhs, self.p):
            assert_equal(out.symbol, 'z')

