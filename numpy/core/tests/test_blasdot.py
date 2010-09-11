from numpy.core import zeros, float64
from numpy.testing import dec, TestCase, assert_almost_equal, assert_
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

try:
    import numpy.core._dotblas as _dotblas
except ImportError:
    _dotblas = None

@dec.skipif(_dotblas is None, "Numpy is not compiled with _dotblas")
def test_blasdot_used():
    from numpy.core import dot, vdot, inner, alterdot, restoredot
    assert_(dot is _dotblas.dot)
    assert_(vdot is _dotblas.vdot)
    assert_(inner is _dotblas.inner)
    assert_(alterdot is _dotblas.alterdot)
    assert_(restoredot is _dotblas.restoredot)
