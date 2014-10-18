from __future__ import division, absolute_import, print_function

import sys
from itertools import product

import numpy as np
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
    for i in range(12):
        np.dot(f, v, r)
    assert_equal(sys.getrefcount(r), 2)
    r2 = np.dot(f, v, out=None)
    assert_array_equal(r2, r)
    assert_(r is np.dot(f, v, out=r))

    v = v[:, 0].copy() # v.shape == (16,)
    r = r[:, 0].copy() # r.shape == (1024,)
    r2 = np.dot(f, v)
    assert_(r is np.dot(f, v, r))
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
    assert_raises(ValueError, np.dot, f, v, r[:, ::2])
    assert_raises(ValueError, np.dot, f, v, r[:, :32])

    r = np.empty((1024, 32), dtype=np.float32)
    assert_raises(ValueError, np.dot, f, v, r)

    r = np.empty((1024, 32), dtype=int)
    assert_raises(ValueError, np.dot, f, v, r)

def test_dot_array_order():
    """ Test numpy dot with different order C, F

    Comparing results with multiarray dot.
    Double and single precisions array are compared using relative
    precision of 7 and 5 decimals respectively.
    Use 30 decimal when comparing exact operations like:
        (a.b)' = b'.a'
    """
    _dot = np.core.multiarray.dot
    a_dim, b_dim, c_dim = 10, 4, 7
    orders = ["C", "F"]
    dtypes_prec = {np.float64: 7, np.float32: 5}
    np.random.seed(7)

    for arr_type, prec in dtypes_prec.items():
        for a_order in orders:
            a = np.asarray(np.random.randn(a_dim, a_dim),
                dtype=arr_type, order=a_order)
            assert_array_equal(np.dot(a, a), a.dot(a))
            # (a.a)' = a'.a', note that mse~=1e-31 needs almost_equal
            assert_almost_equal(a.dot(a), a.T.dot(a.T).T, decimal=prec)

            #
            # Check with making explicit copy
            #
            a_T = a.T.copy(order=a_order)
            assert_almost_equal(a_T.dot(a_T), a.T.dot(a.T), decimal=prec)
            assert_almost_equal(a.dot(a_T), a.dot(a.T), decimal=prec)
            assert_almost_equal(a_T.dot(a), a.T.dot(a), decimal=prec)

            #
            # Compare with multiarray dot
            #
            assert_almost_equal(a.dot(a), _dot(a, a), decimal=prec)
            assert_almost_equal(a.T.dot(a), _dot(a.T, a), decimal=prec)
            assert_almost_equal(a.dot(a.T), _dot(a, a.T), decimal=prec)
            assert_almost_equal(a.T.dot(a.T), _dot(a.T, a.T), decimal=prec)
            for res in a.dot(a), a.T.dot(a), a.dot(a.T), a.T.dot(a.T):
                assert res.flags.c_contiguous

            for b_order in orders:
                b = np.asarray(np.random.randn(a_dim, b_dim),
                    dtype=arr_type, order=b_order)
                b_T = b.T.copy(order=b_order)
                assert_almost_equal(a_T.dot(b), a.T.dot(b), decimal=prec)
                assert_almost_equal(b_T.dot(a), b.T.dot(a), decimal=prec)
                # (b'.a)' = a'.b
                assert_almost_equal(b.T.dot(a), a.T.dot(b).T, decimal=prec)
                assert_almost_equal(a.dot(b), _dot(a, b), decimal=prec)
                assert_almost_equal(b.T.dot(a), _dot(b.T, a), decimal=prec)


                for c_order in orders:
                    c = np.asarray(np.random.randn(b_dim, c_dim),
                        dtype=arr_type, order=c_order)
                    c_T = c.T.copy(order=c_order)
                    assert_almost_equal(c.T.dot(b.T), c_T.dot(b_T), decimal=prec)
                    assert_almost_equal(c.T.dot(b.T).T, b.dot(c), decimal=prec)
                    assert_almost_equal(b.dot(c), _dot(b, c), decimal=prec)
                    assert_almost_equal(c.T.dot(b.T), _dot(c.T, b.T), decimal=prec)

@dec.skipif(True) # ufunc override disabled for 1.9
def test_dot_override():
    class A(object):
        def __numpy_ufunc__(self, ufunc, method, pos, inputs, **kwargs):
            return "A"

    class B(object):
        def __numpy_ufunc__(self, ufunc, method, pos, inputs, **kwargs):
            return NotImplemented

    a = A()
    b = B()
    c = np.array([[1]])

    assert_equal(np.dot(a, b), "A")
    assert_equal(c.dot(a), "A")
    assert_raises(TypeError, np.dot, b, c)
    assert_raises(TypeError, c.dot, b)


def test_npdot_segfault():
    if sys.platform != 'darwin': return
    # Test for float32 np.dot segfault
    # https://github.com/numpy/numpy/issues/4007

    def aligned_array(shape, align, dtype, order='C'):
        # Make array shape `shape` with aligned at `align` bytes
        d = dtype()
        # Make array of correct size with `align` extra bytes
        N = np.prod(shape)
        tmp = np.zeros(N * d.nbytes + align, dtype=np.uint8)
        address = tmp.__array_interface__["data"][0]
        # Find offset into array giving desired alignment
        for offset in range(align):
            if (address + offset) % align == 0: break
        tmp = tmp[offset:offset+N*d.nbytes].view(dtype=dtype)
        return tmp.reshape(shape, order=order)

    def as_aligned(arr, align, dtype, order='C'):
        # Copy `arr` into an aligned array with same shape
        aligned = aligned_array(arr.shape, align, dtype, order)
        aligned[:] = arr[:]
        return aligned

    def assert_dot_close(A, X, desired):
        assert_allclose(np.dot(A, X), desired, rtol=1e-5, atol=1e-7)

    m = aligned_array(100, 15, np.float32)
    s = aligned_array((100, 100), 15, np.float32)
    # This always segfaults when the sgemv alignment bug is present
    np.dot(s, m)
    # test the sanity of np.dot after applying patch
    for align, m, n, a_order in product(
        (15, 32),
        (10000,),
        (200, 89),
        ('C', 'F')):
        # Calculation in double precision
        A_d = np.random.rand(m, n)
        X_d = np.random.rand(n)
        desired = np.dot(A_d, X_d)
        # Calculation with aligned single precision
        A_f = as_aligned(A_d, align, np.float32, order=a_order)
        X_f = as_aligned(X_d, align, np.float32)
        assert_dot_close(A_f, X_f, desired)
        # Strided A rows
        A_d_2 = A_d[::2]
        desired = np.dot(A_d_2, X_d)
        A_f_2 = A_f[::2]
        assert_dot_close(A_f_2, X_f, desired)
        # Strided A columns, strided X vector
        A_d_22 = A_d_2[:, ::2]
        X_d_2 = X_d[::2]
        desired = np.dot(A_d_22, X_d_2)
        A_f_22 = A_f_2[:, ::2]
        X_f_2 = X_f[::2]
        assert_dot_close(A_f_22, X_f_2, desired)
        # Check the strides are as expected
        if a_order == 'F':
            assert_equal(A_f_22.strides, (8, 8 * m))
        else:
            assert_equal(A_f_22.strides, (8 * n, 8))
        assert_equal(X_f_2.strides, (8,))
        # Strides in A rows + cols only
        X_f_2c = as_aligned(X_f_2, align, np.float32)
        assert_dot_close(A_f_22, X_f_2c, desired)
        # Strides just in A cols
        A_d_12 = A_d[:, ::2]
        desired = np.dot(A_d_12, X_d_2)
        A_f_12 = A_f[:, ::2]
        assert_dot_close(A_f_12, X_f_2c, desired)
        # Strides in A cols and X
        assert_dot_close(A_f_12, X_f_2, desired)
