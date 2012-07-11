import numpy as np
import sys
from numpy.core import zeros, float64
from numpy.testing import dec, TestCase, assert_almost_equal, assert_, \
     assert_raises, assert_array_equal, assert_allclose, assert_equal
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


def test_dot_2args():
    from numpy.core import dot

    a = np.array([[1, 2], [3, 4]], dtype=float)
    b = np.array([[1, 0], [1, 1]], dtype=float)
    c = np.array([[3, 2], [7, 4]], dtype=float)

    d = dot(a, b)
    assert_allclose(c, d)

def test_dot_3args():
    np.random.seed(22)
    f = np.random.random_sample((1024, 16))
    v = np.random.random_sample((16, 32))

    r = np.empty((1024, 32))
    for i in xrange(12):
        np.dot(f,v,r)
    assert_equal(sys.getrefcount(r), 2)
    r2 = np.dot(f,v,out=None)
    assert_array_equal(r2, r)
    assert_(r is np.dot(f,v,out=r))

    v = v[:,0].copy() # v.shape == (16,)
    r = r[:,0].copy() # r.shape == (1024,)
    r2 = np.dot(f,v)
    assert_(r is np.dot(f,v,r))
    assert_array_equal(r2, r)

def test_dot_3args_errors():
    np.random.seed(22)
    f = np.random.random_sample((1024, 16))
    v = np.random.random_sample((16, 32))

    r = np.empty((1024, 31))
    assert_raises(ValueError, np.dot, f, v, r)

    r = np.empty((1024,))
    assert_raises(ValueError, np.dot, f, v, r)

    r = np.empty((32,))
    assert_raises(ValueError, np.dot, f, v, r)

    r = np.empty((32, 1024))
    assert_raises(ValueError, np.dot, f, v, r)
    assert_raises(ValueError, np.dot, f, v, r.T)

    r = np.empty((1024, 64))
    assert_raises(ValueError, np.dot, f, v, r[:,::2])
    assert_raises(ValueError, np.dot, f, v, r[:,:32])

    r = np.empty((1024, 32), dtype=np.float32)
    assert_raises(ValueError, np.dot, f, v, r)

    r = np.empty((1024, 32), dtype=int)
    assert_raises(ValueError, np.dot, f, v, r)


