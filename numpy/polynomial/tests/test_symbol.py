"""
Tests related to the ``symbol`` attribute of the ABCPolyBase class.
"""

import pytest
import numpy.polynomial as poly
from numpy.polynomial._polybase import ABCPolyBase
from numpy.testing import assert_equal

class TestInit:
    """
    Test polynomial creation with symbol kwarg.
    """
    
    def test_default_symbol(self):
        p = poly.Polynomial([1, 2, 3])
        assert_equal(p.symbol, 'x')
