from numpy.core import zeros, float64
from numpy.testing import TestCase, assert_almost_equal
from numpy.core.multiarray import inner as inner_

DECPREC = 14

class TestInner(TestCase):
    def test_vecself(self):
        """Ticket 844."""
        # Inner product of a vector with itself segfaults or give meaningless
        # result
        a = zeros(shape = (1, 80), dtype = float64)
        p = inner_(a, a)
        assert_almost_equal(p, 0, decimal = DECPREC)
